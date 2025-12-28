import os
import json
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
def count_neurons(image_type,  robust_check=False, sensitivity_check=False):
	if image_type == "masked":
		image_set = ["masked_male", "masked_female"]
	elif image_type == "unmasked":
		image_set = ["male", "female"]
	threshold = 0.9 if not robust_check else 0.5
	k = 5 if not sensitivity_check else 10
	male_img_count = 0
	female_img_count = 0
	male_scores = json.load(open(f"vit_relevance/{image_set[0]}/scores.json"))
	img_paths = os.listdir(os.path.join("vit_relevance", f"{image_set[0]}"))
	male_neuron_set = {}
	for img_path in img_paths:
		if ".json" in img_path:
			continue
		if male_scores[img_path] < threshold:
			continue
		full_path = os.path.join(f"vit_relevance/{image_set[0]}", img_path, f"top{k}_relevance_male_vit.json")
		data = json.load(open(full_path, "r"))
		for layer, values in data.items():
			if layer not in male_neuron_set:
				male_neuron_set[layer] = {}
			male_neuron_set[layer][img_path] = values["top5_index"]
		male_img_count += 1

	female_scores = json.load(open(f"vit_relevance/{image_set[1]}/scores.json"))
	img_paths = os.listdir(os.path.join("vit_relevance", f"{image_set[1]}"))
	female_neuron_set = {}
	for img_path in img_paths:
		if ".json" in img_path:
			continue
		if female_scores[img_path] < threshold:
			continue
		full_path = os.path.join(f"vit_relevance/{image_set[1]}", img_path, f"top{k}_relevance_female_vit.json")
		data = json.load(open(full_path, "r"))
		for layer, values in data.items():
			if layer not in female_neuron_set:
				female_neuron_set[layer] = {}
			female_neuron_set[layer][img_path] = values["top5_index"]
		female_img_count += 1

	img_paths = os.listdir(os.path.join("vit_relevance", f"{image_set[0]}"))
	male_mid_neuron_set = {}
	for img_path in img_paths:
		if ".json" in img_path:
			continue
		if male_scores[img_path] <  threshold:
			continue
		full_path = os.path.join(f"vit_relevance/{image_set[0]}", img_path, f"top{k}_intermediate_relevance_male_vit.json")
		data = json.load(open(full_path, "r"))
		for layer, values in data.items():
			if layer not in male_mid_neuron_set:
				male_mid_neuron_set[layer] = {}
			male_mid_neuron_set[layer][img_path] = values["top5_index"]

	img_paths = os.listdir(os.path.join("vit_relevance", f"{image_set[1]}"))
	female_mid_neuron_set = {}
	for img_path in img_paths:
		if ".json" in img_path:
			continue
		if female_scores[img_path] < threshold:
			continue
		full_path = os.path.join(f"vit_relevance/{image_set[1]}", img_path, f"top{k}_intermediate_relevance_female_vit.json")
		data = json.load(open(full_path, "r"))
		for layer, values in data.items():
			if layer not in female_mid_neuron_set:
				female_mid_neuron_set[layer] = {}
			female_mid_neuron_set[layer][img_path] = values["top5_index"]
   
	male_neuron_counts = {}
	for layer, neurons in male_neuron_set.items():
		neuron_count = Counter()
		for img, neuron_indices in neurons.items():
			neuron_count.update(neuron_indices)
		male_neuron_counts[layer] = neuron_count
	female_neuron_counts = {}
	for layer, neurons in female_neuron_set.items():
		neuron_count = Counter()
		for img, neuron_indices in neurons.items():
			neuron_count.update(neuron_indices)
		female_neuron_counts[layer] = neuron_count
	male_mid_neuron_counts = {}
	for layer, neurons in male_mid_neuron_set.items():
		neuron_count = Counter()
		for img, neuron_indices in neurons.items():
			neuron_count.update(neuron_indices)
		male_mid_neuron_counts[layer] = neuron_count
	female_mid_neuron_counts = {}
	for layer, neurons in female_mid_neuron_set.items():
		neuron_count = Counter()
		for img, neuron_indices in neurons.items():
			neuron_count.update(neuron_indices)
		female_mid_neuron_counts[layer] = neuron_count
	
	top_male_neuron_count = []
	top_female_neuron_count = []
	top_male_mid_neuron_count = []
	top_female_mid_neuron_count = []
	for layer, neuron_count in male_neuron_counts.items():
		top_male_neuron_count.append(neuron_count.most_common(1)[0][1]/male_img_count*100)
	for layer, neuron_count in female_neuron_counts.items():
		top_female_neuron_count.append(neuron_count.most_common(1)[0][1]/female_img_count*100)
	for layer, neuron_count in male_mid_neuron_counts.items():
		top_male_mid_neuron_count.append(neuron_count.most_common(1)[0][1]/male_img_count*100)
	for layer, neuron_count in female_mid_neuron_counts.items():
		top_female_mid_neuron_count.append(neuron_count.most_common(1)[0][1]/female_img_count*100)

	return top_male_neuron_count, top_female_neuron_count, top_male_mid_neuron_count, top_female_mid_neuron_count

def plot_consistency(robust_check=False, sensitivity_check=False):
	top_male_neuron_count, top_female_neuron_count, top_male_mid_neuron_count, top_female_mid_neuron_count = count_neurons("masked", robust_check, sensitivity_check)
	layers = np.arange(12)

	plt.style.use('seaborn-v0_8-whitegrid')
	plt.figure(figsize=(10, 6))

	plt.plot(layers, top_male_neuron_count, marker='s', linestyle='--', label='Male (Post_LayerNorm)')
	plt.plot(layers, top_female_neuron_count, marker='s', linestyle='--', label='Female (Post_LayerNorm)')
	plt.plot(layers, top_male_mid_neuron_count, marker='o', linestyle='-', label="Male (Post_GELU)")
	plt.plot(layers, top_female_mid_neuron_count, marker='o', linestyle='-', label="Female (Post_GELU)")

	plt.title('Consistency of Top-Ranked Neurons Per Layer(Masked Images)', fontsize=22, fontweight='bold')
	plt.xlabel('Transformer Layer', fontsize=18, fontweight='bold')
	plt.ylabel('Consistency (%)', fontsize=18, fontweight='bold')
	plt.xticks(layers) # Ensure a tick for every layer
	plt.ylim(0, 105) # Set y-axis from 0% to 100%
	plt.legend(
		fontsize=22,
		prop={'weight': 'bold'}
	)
	plt.tight_layout()
	if robust_check:
		plt.savefig('masked_consistency_plot_robust.png', dpi=300)
	elif sensitivity_check:
		plt.savefig('masked_consistency_plot_sensitivity.png', dpi=300)
	else:
		plt.savefig('masked_consistency_plot.png', dpi=300)
	plt.close()

	top_male_neuron_count, top_female_neuron_count, top_male_mid_neuron_count, top_female_mid_neuron_count = count_neurons("unmasked", robust_check, sensitivity_check)
	plt.figure(figsize=(10, 6))

	plt.plot(layers, top_male_neuron_count, marker='s', linestyle='--', label='Male (Post_LayerNorm)')
	plt.plot(layers, top_female_neuron_count, marker='s', linestyle='--', label='Female (Post_LayerNorm)')
	plt.plot(layers, top_male_mid_neuron_count, marker='o', linestyle='-', label="Male (Post_GELU)")
	plt.plot(layers, top_female_mid_neuron_count, marker='o', linestyle='-', label="Female (Post_GELU)")

	plt.title('Consistency of Top-Ranked Neurons Per Layer(Unmasked Images)', fontsize=22, fontweight='bold')
	plt.xlabel('Transformer Layer', fontsize=18, fontweight='bold')
	plt.ylabel('Consistency (%)', fontsize=18, fontweight='bold')
	plt.xticks(layers) # Ensure a tick for every layer
	plt.ylim(0, 105) # Set y-axis from 0% to 100%
	plt.legend(
		fontsize=22,
		prop={'weight': 'bold'}
	)
	plt.tight_layout()
	if robust_check:
		plt.savefig('unmasked_consistency_plot_robust.png', dpi=300)
	elif sensitivity_check:
		plt.savefig('unmasked_consistency_plot_sensitivity.png', dpi=300)
	else:
		plt.savefig('unmasked_consistency_plot.png', dpi=300)
	plt.close()
