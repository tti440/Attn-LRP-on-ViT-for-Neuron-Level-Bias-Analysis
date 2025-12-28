import transformers.activations
import math
from lxt.explicit.core import Composite
import transformers
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
import torch.nn as nn

def hidden_relevance_hook(module, input, output):
	if isinstance(output, tuple):
		output = output[0]
	module.hidden_relevance = output.detach().cpu()
 
def save_relevance_heatmap(relevance_matrix, title, save_path):
	fig, ax = plt.subplots(figsize=(12, 6))
	im = ax.imshow(relevance_matrix.T, cmap='bwr', aspect='auto')
	plt.title(title)
	plt.xlabel("Layer")
	plt.ylabel("Neuron Index")
	plt.colorbar(im, label="CLS Token Relevance")
	plt.xticks(np.arange(len(relevance_matrix)), [f"L{i}" for i in range(len(relevance_matrix))], rotation=90)
	plt.tight_layout()
	plt.savefig(save_path)
	plt.close(fig)
	#plt.show()
 
def save_patch_relevance_heatmap(relevance_matrix, title, save_path):
	fig, ax = plt.subplots(figsize=(12, 6))
	im = ax.imshow(relevance_matrix.T, cmap='bwr', aspect='auto')
	plt.title(title)
	plt.xlabel("Layer")
	plt.ylabel("Patch Index")
	plt.colorbar(im, label="CLS Token Relevance")
	plt.xticks(np.arange(len(relevance_matrix)), [f"L{i}" for i in range(len(relevance_matrix))], rotation=90)
	plt.tight_layout()
	plt.savefig(save_path)
	plt.close(fig)
	#plt.show()

def relevance_retrieval(model_name, model, gender, image_path, target_class, score):
	outdir = f"{model_name}_relevance/{gender}/{image_path.split('_')[0]}/"
	if not os.path.exists(f"{model_name}_relevance"):
		os.makedirs(f"{model_name}_relevance")
	if not os.path.exists(f"{model_name}_relevance/{gender}"):
		os.makedirs(f"{model_name}_relevance/{gender}")
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	relevance_trace = []
	patch_relevance_trace = []
	relevance_df = {}
	relevance_intermediate=[]
	patch_mid_relevance = []
	relevance_intermediate_df = {}
	count = 0
	labels = model.config.id2label
	label = labels[target_class]
	for layer in model.vit.encoder.layer:
		relevance = layer.hidden_relevance[0].sum(0)
		patch_relevance = layer.hidden_relevance[0].sum(-1)
		scores, top5_index = relevance.topk(5)
		relevance_df[f"layer{count}"] = {
			"top5_index": top5_index.cpu().numpy().tolist(),
			"scores":scores.cpu().numpy().tolist()
		}
  
		relevance = relevance / relevance.abs().max()
		patch_relevance = patch_relevance / patch_relevance.abs().max()
  
		relevance_trace.append(relevance)
		patch_relevance_trace.append(patch_relevance)
		count += 1

	relevance_trace = torch.stack(relevance_trace)
	patch_relevance_trace = torch.stack(patch_relevance_trace)
	count = 0
 
	for layer in model.vit.encoder.layer:
		relevance = layer.intermediate.hidden_relevance[0].sum(0)
		patch_relevance = layer.intermediate.hidden_relevance[0].sum(-1)
  
		scores, top5_index = relevance.topk(5)
		relevance_intermediate_df[f"layer{count}"] = {
			"top5_index": top5_index.cpu().numpy().tolist(),
			"scores": scores.cpu().numpy().tolist()
		}
  
		relevance = relevance / relevance.abs().max()
		patch_relevance = patch_relevance / patch_relevance.abs().max()
		relevance_intermediate.append(relevance)
		patch_mid_relevance.append(patch_relevance)
		count += 1
  
	relevance_intermediate = torch.stack(relevance_intermediate)
	patch_mid_relevance = torch.stack(patch_mid_relevance)
	pred = label.split(" ")[0].replace(",", "")
	save_relevance_heatmap(relevance_trace, f"Class-Specific Relevance on CLS Token: {pred} {score*100:.4f}", f"{outdir}cls_relevance_heatmap_{label}_vit.png")
	save_relevance_heatmap(relevance_intermediate, f"Intermediate Layer Relevance on CLS Token: {pred} {score*100:.4f}", f"{outdir}intermediate_cls_relevance_heatmap_{label}_vit.png")
	save_patch_relevance_heatmap(patch_relevance_trace, f"Patch-Specific Relevance on CLS Token: {pred} {score*100:.4f}", f"{outdir}patch_cls_relevance_heatmap_{label}_vit.png")
	save_patch_relevance_heatmap(patch_mid_relevance, f"Intermediate Layer Patch-Specific Relevance on CLS Token: {pred} {score*100:.4f}", f"{outdir}intermediate_patch_cls_relevance_heatmap_{label}_vit.png")
	with open(f"{outdir}relevance_df_{label}_vit.jsonl", "w") as f:
		for layer, data in relevance_df.items():
			f.write(json.dumps({layer: data}) + "\n")

	with open(f"{outdir}relevance_intermediate_df_{label}_vit.jsonl", "w") as f:
		for layer, data in relevance_intermediate_df.items():
			f.write(json.dumps({layer: data}) + "\n")
