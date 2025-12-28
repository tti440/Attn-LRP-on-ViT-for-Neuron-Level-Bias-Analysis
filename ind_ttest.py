import numpy as np
import pingouin as pg
import pickle
import pingouin as pg
import os
import json
import pickle
from statsmodels.stats.multitest import multipletests
import re
from collections import defaultdict
import pandas as pd

all_results = []
labels = {
	"labcoat" : 617,
	"stethoscope" : 823,
	"jeans" : 608,
	"cardigan" : 474,
	"miniskirt" : 655,
	"academicgown" : 400,
	"pajamas" : 697,
	"oscilloscope" : 688,
	"suit" : 834,
	"notePC" : 681,
	"sunscreen" : 838,
	"wig" : 903,
	"lipstick" :629,
	"bra" : 459,
	"windsor" : 906,
	"desk" : 526,
	"library" : 624
}

def ind_ttest(image_type, component, robust_check=False, sensitivity_check=False):
	if component == "final":
		comp_path = ""
	elif component == "intermediate":
		comp_path = "intermediate_"
	else:
		raise ValueError("Component must be 'final' or 'intermediate'.")

	if component == "final":
		comp_path1 = "trace"
	elif component == "intermediate":
		comp_path1 = "intermediate"
	else:
		raise ValueError("Component must be 'final' or 'intermediate'.")

	threshold = 0.9 if not robust_check else 0.5
	k = 5 if not sensitivity_check else 10
 
	if image_type == "masked":
		image_set = ["masked_male", "masked_female"]
	elif image_type == "unmasked":
		image_set = ["male", "female"]
  
	male_scores = json.load(open(f"vit_relevance/{image_set[0]}/scores.json"))
	female_scores = json.load(open(f"vit_relevance/{image_set[1]}/scores.json"))
 
	img_paths = os.listdir(os.path.join("vit_relevance", f"{image_set[0]}"))
	male_neuron_set = {}
	for img_path in img_paths:
		if ".json" in img_path:
			continue
		if male_scores[img_path] < threshold:
			continue
		full_path = os.path.join(f"vit_relevance/{image_set[0]}", img_path, f"top{k}_{comp_path}relevance_male_vit.json")
		data = json.load(open(full_path, "r"))
		for layer, values in data.items():
			if layer not in male_neuron_set:
				male_neuron_set[layer] = {}
			male_neuron_set[layer][img_path] = values["top5_index"]

	img_paths = os.listdir(os.path.join("vit_relevance", f"{image_set[1]}"))
	female_neuron_set = {}
	for img_path in img_paths:
		if ".json" in img_path:
			continue
		if female_scores[img_path] < threshold:
			continue
		full_path = os.path.join(f"vit_relevance/{image_set[1]}", img_path, f"top{k}_{comp_path}relevance_female_vit.json")
		data = json.load(open(full_path, "r"))
		for layer, values in data.items():
			if layer not in female_neuron_set:
				female_neuron_set[layer] = {}
			female_neuron_set[layer][img_path] = values["top5_index"]
	
	done = []
	for label1, _ in labels.items():
		label1_neurons = {}
		label2_neurons = {}
		for layer, data in male_neuron_set.items():
			label1_neurons[layer] = []
			for img, neurons in data.items():
				targeted_neurons = male_neuron_set[str(layer)][img]
				path1 = os.path.join(f"vit_relevance/{image_set[0]}", img, f"relevance_{comp_path1}_{label1}_vit.pkl")
				data1 = pickle.load(open(path1, "rb"))
				label1_neurons[layer].extend(data1[int(layer)][targeted_neurons])
		for layer, data in female_neuron_set.items():
			label2_neurons[layer] = []
			for img, neurons in data.items():
				targeted_neurons = female_neuron_set[str(layer)][img]
				path2 = os.path.join(f"vit_relevance/{image_set[1]}", img, f"relevance_{comp_path1}_{label1}_vit.pkl")
				data2 = pickle.load(open(path2, "rb"))
				label2_neurons[layer].extend(data2[int(layer)][targeted_neurons])

		for layer in label1_neurons.keys():
			stats_df = pg.ttest(
				label1_neurons[layer], label2_neurons[layer],
				paired=False
			)
			t_stat = stats_df['T'].iloc[0]
			p_value = stats_df['p-val'].iloc[0]
			cohen_d = stats_df['cohen-d'].iloc[0]
			all_results.append({
				"label1": label1,
				"layer": layer,
				"t_stat": t_stat,
				"p_value": p_value,
				"cohen_d": cohen_d,
				"male mean": np.mean(label1_neurons[layer]),
				"female mean": np.mean(label2_neurons[layer]),
				"group": f"{image_type} {component}"
			})
   
def ind_fdr_correction(robust_check=False, sensitivity_check=False):
	print("Running independent t-test with FDR correction...")
	for image_type in ["masked", "unmasked"]:
		print(f"Processing {image_type} images...")
		for component in ["final", "intermediate"]:
			print(f"Processing {component} component...")
			ind_ttest(image_type, component, robust_check, sensitivity_check)
	print("Independent t-test completed.")
	print(f"Total results collected: {len(all_results)}")
	assert (len(all_results)/4) == 204
	p_values = [r["p_value"] for r in all_results]
	_, q_values, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
	for i, q in enumerate(q_values):
		all_results[i]["q_value"] = q
		all_results[i]["sig_fdr"] = q < 0.05
	unmasked_final=[]
	unmasked_intermediate = []
	masked_final = []
	masked_intermediate = []
	for data in all_results:
		if data["sig_fdr"]:
			if data["group"] == "unmasked final":
				unmasked_final.append(data)
			elif data["group"] == "unmasked intermediate":
				unmasked_intermediate.append(data)
			elif data["group"] == "masked final":
				masked_final.append(data)
			elif data["group"] == "masked intermediate":
				masked_intermediate.append(data)
	print("Independent t-test is completed.")
	iter_data = {
		"unmasked_final": unmasked_final,
		"unmasked_intermediate": unmasked_intermediate,
		"masked_final": masked_final,
		"masked_intermediate": masked_intermediate
	}
	for key, data in iter_data.items():
		texts = ""
		for result in data:
			texts+=f"Label: {result['label1']}, Layer: {result['layer']}, t-statistic: {float(result['t_stat']):.2f}, p-value: {float(result['p_value']):.2e}, q-value: {float(result['q_value']):.2e}, cohen_d: {float(result['cohen_d']):.2f}\n"
		raw_text = texts  # truncated for example
		# Parse full entries
		pattern = re.compile(r"Label: (\w+), Layer: (\d+), t-statistic: ([\-\d.]+), p-value: ([\d.e\-]+), q-value: ([\d.e\-]+), cohen_d: ([\-\d.]+)")

		entries = pattern.findall(raw_text)

		# Prepare nested dictionary
		results = defaultdict(lambda: [''] * 12)

		for label1, layer, t_stat, p_val, q_val, cohen_d in entries:
			pair = f"{label1}"
			layer = int(layer)
			results[pair][layer] = f"t={float(t_stat):.2f} p={float(p_val):.2e} q={float(q_val):.2e} d={float(cohen_d):.2f}"

		# Convert to DataFrame
		df = pd.DataFrame.from_dict(results, orient='index', columns=[f"Layer {i}" for i in range(12)])
		os.makedirs("ind_ttest", exist_ok=True)
		if sensitivity_check:
			df.to_csv(f"ind_ttest/sensitivity_{key}_results.csv", index_label=True)
		elif robust_check:
			df.to_csv(f"ind_ttest/lower_{key}_results.csv", index_label=True)
		else:
			df.to_csv(f"ind_ttest/{key}_results.csv", index_label=True)
