from functools import partial
import torch
from torchvision.models import vision_transformer
from torch.nn import GELU, LayerNorm
import transformers
from transformers import ViTForImageClassification, BeitForImageClassification, AutoModelForImageClassification, SwinForImageClassification
from lxt.efficient.patches import patch_method, non_linear_forward, layer_norm_forward, cp_multi_head_attention_forward, attention_tf_vit, patch_attention, attention_tf_beit, \
    attention_dino , attention_tf_swin
from transformers.models.vit.modeling_vit import ViTSelfAttention
import os
import sys
hub_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
if hub_dir not in sys.path:
    sys.path.insert(0, hub_dir)

# AttnLRP outside the attention mechanism & CP-LRP inside the attention mechnism is easier to tune for gamma
cp_LRP = {
    torch.nn.modules.activation.GELU: partial(patch_method, non_linear_forward, keep_original=True),
    torch.nn.modules.normalization.LayerNorm: partial(patch_method, layer_norm_forward),
    torch.nn.modules.activation.MultiheadAttention: partial(patch_method, cp_multi_head_attention_forward, keep_original=True),
}

def get_vit_LRP():
    return {
    transformers.activations.GELUActivation: partial(patch_method, non_linear_forward, keep_original=True),
    torch.nn.modules.normalization.LayerNorm: partial(patch_method, layer_norm_forward),
    transformers.models.vit.modeling_vit.ViTSelfAttention: partial(patch_method, attention_tf_vit, keep_original=True),
}

def get_beit_LRP():
    return {
    torch.nn.modules.activation.GELU: partial(patch_method, non_linear_forward, keep_original=True),
    torch.nn.modules.normalization.LayerNorm: partial(patch_method, layer_norm_forward),
    transformers.models.beit.modeling_beit.BeitSdpaSelfAttention: partial(patch_method, attention_tf_beit, keep_original=True),
}

def get_deit_LRP():
    return {
    transformers.activations.GELUActivation: partial(patch_method, non_linear_forward, keep_original=True),
    torch.nn.modules.normalization.LayerNorm: partial(patch_method, layer_norm_forward),
    transformers.models.vit.modeling_vit.ViTSelfAttention: partial(patch_method, attention_tf_vit, keep_original=True),
}

def get_dino_LRP():
    import dinov2.layers.attention as dino_attn
    return {
        torch.nn.modules.activation.GELU: partial(patch_method, non_linear_forward, keep_original=True),
        torch.nn.modules.normalization.LayerNorm: partial(patch_method, layer_norm_forward),
        dino_attn.MemEffAttention: partial(patch_method, attention_dino, keep_original=True),
    }
    
def get_swin_LRP():
    return{
    torch.nn.modules.activation.GELU: partial(patch_method, non_linear_forward, keep_original=True),
    torch.nn.modules.normalization.LayerNorm: partial(patch_method, layer_norm_forward),
    transformers.models.swin.modeling_swin.SwinSelfAttention: partial(patch_method, attention_tf_swin, keep_original=True),
}
    
def get_mask_LRP():
    return {
        torch.nn.modules.activation.GELU: partial(patch_method, non_linear_forward, keep_original=True),
        torch.nn.modules.normalization.LayerNorm: partial(patch_method, layer_norm_forward),
        ViTSelfAttention: partial(patch_method, attention_tf_vit, keep_original=True),
    }