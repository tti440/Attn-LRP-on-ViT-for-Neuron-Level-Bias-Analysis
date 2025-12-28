# Copyright 2024, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. &
# the authors: Reduan Achtibat, Sayed Mohammad Vakilzadeh Hatefi, Maximilian Dreyer, Aakriti Jain,
# Thomas Wiegand, Sebastian Lapuschkin, Wojciech Samek. All rights reserved.
# 
# This code is based on the following work:
# 
#   'AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers. ICML 2024.'
#
# The copyright in this software is being made available under the Clear BSD License.
# No patent rights, trademark rights and/or other Intellectual Property Rights other than
# the copyrights concerning the Software are granted under this license.
# You may obtain a full copy of the License at
#     
#   https://github.com/rachtibat/LRP-eXplains-Transformers/blob/main/LICENSE
#
import sys
import torch
from warnings import warn
from lxt.efficient.rules import stop_gradient, divide_gradient, identity_rule_implicit
from torch import nn
import math
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from typing import Callable, Optional
import logging
import os
import warnings
from torch import Tensor

def check_already_patched(target_fn, new_fn):
    """
    Check if a function is already replaced by another function.
    Used to avoid redundant patching.

    Parameters
    ----------
    target_fn : function
        The function to be wrapped.
    new_fn : function
        The new function to wrap the target function.

    Returns
    -------
    bool
        True if the target function is already wrapped by the new function, False otherwise.
    """

    if target_fn.__module__ == new_fn.__module__:
        if hasattr(target_fn, '__name__'):
            warn(f"{target_fn.__name__} already patched.")
        else:
            warn(f"{target_fn} already patched.")
        return True
    return False

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

    # Normalize the attention scores to probabilities.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # Mask heads if we want to
    if attention_mask is not None:
        attn_weights = attn_weights * attention_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

def attention_tf_vit(self, hidden_states, *args, **kwargs):
    """
    For the CP-LRP variant in Hugging Face ViT, we stop the gradient flow through the 
    query and key tensors, which are directly connected to the softmax function.
    """    
    output_attentions = kwargs.get("output_attentions", False)
    head_mask = kwargs.get("head_mask", None)
    
    key = self.key(hidden_states)
    value = self.value(hidden_states)
    query = self.query(hidden_states)

    # Shared reshaping logic
    def transpose_for_scores(x, num_heads, head_dim):
        new_shape = x.size()[:-1] + (num_heads, head_dim) 
        return x.view(*new_shape).permute(0, 2, 1, 3)   

    # Apply to all three
    key_layer = transpose_for_scores(key, self.num_attention_heads, self.attention_head_size)
    value_layer = transpose_for_scores(value, self.num_attention_heads, self.attention_head_size)
    query_layer = transpose_for_scores(query, self.num_attention_heads, self.attention_head_size)

    key_layer = stop_gradient(key_layer)
    query_layer = stop_gradient(query_layer)
    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.reshape(new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

    return outputs

def attention_tf_beit(self, hidden_states, *args, **kwargs):
    head_mask = args[0]
    output_attentions = args[1]
    relative_position_bias = args[2]
    interpolate_pos_encoding = args[3]
    resolution = args[4]
    if output_attentions or head_mask is not None:
            from transformers.models.beit.modeling_beit import BeitSelfAttention
            return BeitSelfAttention.forward(
                self,
                hidden_states=hidden_states,
                head_mask=head_mask,
                output_attentions=output_attentions,
                relative_position_bias=relative_position_bias,
                interpolate_pos_encoding=interpolate_pos_encoding,
                resolution=resolution,
            )

    mixed_query_layer = self.query(hidden_states)
    key_layer = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states))
    query_layer = self.transpose_for_scores(mixed_query_layer)

    key_layer = stop_gradient(key_layer)
    query_layer = stop_gradient(query_layer)
    
    attn_bias = None
    if self.has_relative_position_bias:
            height, width = resolution
            window_size = (height // self.config.patch_size, width // self.config.patch_size)
            attn_bias = self.relative_position_bias(
                window_size, interpolate_pos_encoding, dim_size=hidden_states.shape[1]
            )

        # Add shared relative position bias if provided.
    if relative_position_bias is not None:
        if attn_bias is None:
            attn_bias = relative_position_bias
        else:
            attn_bias += relative_position_bias

    scaling = 1 / math.sqrt(self.attention_head_size)
    context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attn_bias,
            dropout_p=self.config.attention_probs_dropout_prob if self.training else 0.0,
            is_causal=False,
            scale=scaling,
    )
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)
    return context_layer, None

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")
def attention_dino(self, x: Tensor, attn_bias=None):
    hub_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
    if hub_dir not in sys.path:
        sys.path.insert(0, hub_dir)
    from dinov2.layers.attention import Attention
    if not XFORMERS_AVAILABLE:
        if attn_bias is not None:
            raise AssertionError("xFormers is required for using nested tensors")
        return Attention.forward(self, x)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

    q, k, v = unbind(qkv, 2)

    q = stop_gradient(q)
    k = stop_gradient(k)
        
    x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
    x = x.reshape([B, N, C])

    x = self.proj(x)
    x  = self.proj_drop(x)
    return x

def attention_tf_swin(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        batch_size, dim, num_channels = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        
        key_layer = stop_gradient(key_layer)
        query_layer = stop_gradient(query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in SwinModel forward() function)
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
            )
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

def patch_method(fn, module, method_name="forward", keep_original=False):
    """
    Patch a method in a module with a new function.

    Parameters
    ----------
    fn : function
        The function to replace the method with.
    module : module
        The module containing the method to be patched.
    method_name : str, optional
        The name of the method to be patched. Default is "forward".
    keep_original : bool, optional
        If True, the original method is saved in the module as f"original_{method_name}".
        Default is False.
    
    Returns
    -------
    bool
        True if the method was successfully patched, False otherwise.
    """
    
    if check_already_patched(getattr(module, method_name), fn):
        return False
    if keep_original:
        setattr(module, f'original_{method_name}', getattr(module, method_name))
    
    setattr(module, method_name, fn)
    return True


def replace_module(patched_module, original_module):
    """
    Replace all attributes of a module with the attributes of another module.

    Parameters
    ----------
    patched_module : module
        The module whose attributes will be copied to the original module.
    original_module : module
        The module whose attributes will be replaced by the patched module.
    
    Returns
    -------
    bool
        True if the module was successfully patched, False otherwise.
    """
    if original_module == patched_module:
        return False

    # Then replace all attributes
    for attr in dir(patched_module):
        if not attr.startswith('__'):  # Skip special methods
            setattr(original_module, attr, getattr(patched_module, attr))

    return True


#############################
###### AttnLRP Patches ######
#############################

def rms_norm_forward(self, hidden_states):
    """
    On normalization operations, we apply the identity rule.
    It is implemented here by stopping the gradient flow through the variance calculation,
    which is equivalent to the identity rule in a Gradient*Input framework.
    """

    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * stop_gradient(torch.rsqrt(variance + self.variance_epsilon))

    return self.weight * hidden_states.to(input_dtype)


def layer_norm_forward(self, x):
    """
    On normalization operations, we apply the identity rule.
    It is implemented here by stopping the gradient flow through the variance calculation,
    which is equivalent to the identity rule in a Gradient*Input framework.
    """

    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    std = (var + self.eps).sqrt()
    y = (x - mean) / stop_gradient(std)
    if self.weight is not None:
        y *= self.weight
    if self.bias is not None:
        y += self.bias

    return y


def gated_mlp_forward(self, x):
    """
    On the element-wise non-linear activation, we apply the identity rule and
    on the element-wise multiplication, we apply the uniform rule.
    Both rules are implemented via the Gradient*Input framework.
    """

    gate_out = self.gate_proj(x)
    gate_out = identity_rule_implicit(self.act_fn, gate_out)

    weighted = gate_out * self.up_proj(x)
    weighted = divide_gradient(weighted, 2)
    return self.down_proj(weighted)


def mlp_forward(self, x):
    """
    On the element-wise non-linear activation, we apply the identity rule,
    which is implemented via the Gradient*Input framework.
    """

    up_out = self.up_proj(x)
    up_out = identity_rule_implicit(self.act_fn, up_out)
    return self.down_proj(up_out)


def patch_attention(module):
    """
    Huggingface's transformers library provides a dictionary of all attention functions.
    We patch all of them with the same wrapper function to implement the uniform rule in
    matmul operations via the Gradient*Input framework. It is sufficient to correct the
    gradient flow later at the query, key, and value tensors.
    """
    new_forward = wrap_attention_forward(module.eager_attention_forward)
    if check_already_patched(module.eager_attention_forward, new_forward):
        return False
    else:
        module.eager_attention_forward = new_forward
    
    NEW_ATTENTION_FUNCTIONS = {}
    for key, value in module.ALL_ATTENTION_FUNCTIONS.items():
        new_forward = wrap_attention_forward(value)
        if check_already_patched(value, new_forward):
            return False
        else:
            NEW_ATTENTION_FUNCTIONS[key] = new_forward
    module.ALL_ATTENTION_FUNCTIONS = NEW_ATTENTION_FUNCTIONS
            #module.ALL_ATTENTION_FUNCTIONS[key] = new_forward
    return True


def wrap_attention_forward(forward_fn):
    def attention_forward(module, query, key, value, *args, **kwargs):

        query = divide_gradient(query, 4)
        key = divide_gradient(key, 4)
        value = divide_gradient(value, 2)

        if 'dropout' in kwargs:
            kwargs['dropout'] = 0.0
        return forward_fn(module, query, key, value, *args, **kwargs)
    return attention_forward


def non_linear_forward(self, x):
    """
    Patch the element-wise non-linear activation functions with the identity rule,
    which is implemented via the Gradient*Input framework.
    """
    return identity_rule_implicit(self.original_forward, x)


def dropout_forward(self, x):
    """
    To use gradient checkpointing in huggingface, we must set the model to 'train()' mode.
    However, this will also activate the dropout layers. We patch the dropout layers
    to set the dropout rate to zero during the forward pass.
    """
    return x



############################
###### CP-LRP Patches ######
############################

def patch_cp_attention(module):
    """
    For the CP-LRP variant, no gradient is allowed to flow through the softmax function.
    We patch all attention functions with a wrapper function that stops the gradient flow
    at the query and key tensors, which are directly connected to the softmax function.
    """
    new_forward = cp_wrap_attention_forward(module.eager_attention_forward)
    if check_already_patched(module.eager_attention_forward, new_forward):
        return False
    else:
        module.eager_attention_forward = new_forward
    
    for key, value in module.ALL_ATTENTION_FUNCTIONS.items():
        new_forward = cp_wrap_attention_forward(value)
        if check_already_patched(value, new_forward):
            return False
        else:
            module.ALL_ATTENTION_FUNCTIONS[key] = new_forward
    return True


def cp_wrap_attention_forward(forward_fn):
    def cp_attention_forward(module, query, key, value, *args, **kwargs):

        query = stop_gradient(query)
        key = stop_gradient(key)

        if 'dropout' in kwargs:
            kwargs['dropout'] = 0.0
        return forward_fn(module, query, key, value, *args, **kwargs)
    return cp_attention_forward


def cp_multi_head_attention_forward(self, query, key, value, *args, **kwargs):
    """
    For the CP-LRP variant, no gradient is allowed to flow through the softmax function.
    We patch the torch.nn.MultiheadAttention.forward such that the gradient flow
    at the query and key tensors is stopped, which are directly connected to the softmax function.
    """
    # print(f"Query shape: {query.shape}, Key shape: {key.shape}, Value shape: {value.shape}")
    query = stop_gradient(query)
    key = stop_gradient(key)
    return self.original_forward(query, key, value, *args, **kwargs)


def cp_gated_mlp_forward(self, x):
    """
    For the CP-LRP variant, no gradient is allowed to flow through the gating mechanism.
    """
    gate_out = stop_gradient(self.gate_proj(x))
    gate_out = self.act_fn(gate_out)
    
    weighted = gate_out * self.up_proj(x)
    return self.down_proj(weighted)
