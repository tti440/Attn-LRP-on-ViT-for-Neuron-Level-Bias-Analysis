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
def pairwise_ttest(image_type, component, robust_check=False, sensitivity_check=False):
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

	
	for img_dir in image_set:
		if ".json" in img_path:
			continue
		done = []
		for label1, _ in labels.items():
			for label2, _ in labels.items():
				if label1 == label2:
					continue
				if label2 in done:
					continue
				if "female" in img_dir:
					for layer, data in female_neuron_set.items():
						label1_neurons = []
						label2_neurons = []
						for img, neurons in data.items():
							targeted_neurons = female_neuron_set[str(layer)][img]
							path1 = os.path.join(f"vit_relevance/{img_dir}", img, f"relevance_{comp_path1}_{label1}_vit.pkl")
							path2 = os.path.join(f"vit_relevance/{img_dir}", img, f"relevance_{comp_path1}_{label2}_vit.pkl")
							data1 = pickle.load(open(path1, "rb"))
							data2 = pickle.load(open(path2, "rb"))
							label1_neurons.extend(data1[int(layer)][targeted_neurons])
							label2_neurons.extend(data2[int(layer)][targeted_neurons])

						label1_neurons = np.array(label1_neurons)
						label2_neurons = np.array(label2_neurons)
						stats_df = pg.ttest(
							label1_neurons, label2_neurons,
							paired=True
						)
						t_stat = stats_df['T'].iloc[0]
						p_value = stats_df['p-val'].iloc[0]
						cohen_d = stats_df['cohen-d'].iloc[0]
						group = img_dir.replace("_", " ") + f" {component}" + " component"
						all_results.append({
							"label1": label1,
							"label2": label2,
							"layer": layer,
							"t_stat": t_stat,
							"p_value": p_value,
							"cohen_d": cohen_d,
							"group" : group,
						})
				elif "male" in img_dir:
					for layer, data in male_neuron_set.items():
						label1_neurons = []
						label2_neurons = []
						for img, neurons in data.items():
							targeted_neurons = male_neuron_set[str(layer)][img]
							path1 = os.path.join(f"vit_relevance/{img_dir}", img, f"relevance_{comp_path1}_{label1}_vit.pkl")
							path2 = os.path.join(f"vit_relevance/{img_dir}", img, f"relevance_{comp_path1}_{label2}_vit.pkl")
							data1 = pickle.load(open(path1, "rb"))
							data2 = pickle.load(open(path2, "rb"))
							label1_neurons.extend(data1[int(layer)][targeted_neurons])
							label2_neurons.extend(data2[int(layer)][targeted_neurons])

						label1_neurons = np.array(label1_neurons)
						label2_neurons = np.array(label2_neurons)
						stats_df = pg.ttest(
							label1_neurons, label2_neurons,
							paired=True
						)
						t_stat = stats_df['T'].iloc[0]
						p_value = stats_df['p-val'].iloc[0]
						cohen_d = stats_df['cohen-d'].iloc[0]
						group = img_dir.replace("_", " ") + f" {component}" + " component"
						all_results.append({
							"label1": label1,
							"label2": label2,
							"layer": layer,
							"t_stat": t_stat,
							"p_value": p_value,
							"cohen_d": cohen_d,
							"group" : group,
						})
			done.append(label1)
   
def pairwise_fdr_correction(robust_check=False, sensitivity_check=False):
	print("Running pairwise t-test with FDR correction...")
	for image_type in ["masked", "unmasked"]:
		print(f"Processing {image_type} images...")
		for component in ["final", "intermediate"]:
			print(f"Processing {component} component...")
			pairwise_ttest(image_type, component, robust_check, sensitivity_check)
	print("Pairwise t-test completed.")
	print(f"Total results collected: {len(all_results)}")
	assert (len(all_results)/8) == 1632 # 17 labels * 16 / 2 * 12 layers
	p_values = [r["p_value"] for r in all_results]
	_, q_values, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
	for i, q in enumerate(q_values):
		all_results[i]["q_value"] = q
		all_results[i]["sig_fdr"] = q < 0.05
	male_final=[]
	male_intermediate = []
	female_final = []
	female_intermediate = []
	masked_male_final = []
	masked_male_intermediate = []
	masked_female_final = []
	masked_female_intermediate = []
	for data in all_results:
		if data["sig_fdr"]:
			if data["group"] == "male final component":
				male_final.append(data)
			elif data["group"] == "male intermediate component":
				male_intermediate.append(data)
			elif data["group"] == "female final component":
				female_final.append(data)
			elif data["group"] == "female intermediate component":
				female_intermediate.append(data)
			elif data["group"] == "masked male final component":
				masked_male_final.append(data)
			elif data["group"] == "masked male intermediate component":
				masked_male_intermediate.append(data)
			elif data["group"] == "masked female final component":
				masked_female_final.append(data)
			elif data["group"] == "masked female intermediate component":
				masked_female_intermediate.append(data)

	iter_data = {
	"male_final": male_final,
	"male_intermediate": male_intermediate,
	"female_final": female_final,
	"female_intermediate": female_intermediate,
	"masked_male_final": masked_male_final,
	"masked_male_intermediate": masked_male_intermediate,
	"masked_female_final": masked_female_final,
	"masked_female_intermediate": masked_female_intermediate
	}
	if robust_check:
		save_file = "lower_all_results.pkl"
	elif sensitivity_check:
		save_file = "sensitivity_all_results.pkl"
	else:
		save_file = "all_results.pkl"
	with open(save_file, "wb") as f:
		pickle.dump(all_results, f)
	print(f"FDR-correction is Completed")
	# ttest_rel summary for each category 
	for key, data in iter_data.items():
		texts = ""
		for result in data:
			texts+=f"Label1: {result['label1']}, Label2: {result['label2']}, Layer: {result['layer']}, t-statistic: {result['t_stat']:.4f}, p-value: {result['p_value']:.4f}, q-value: {result['q_value']:.4f}, cohen_d: {result['cohen_d']:.4f}\n"
		raw_text = texts  # truncated for example

		# Parse full entries
		pattern = re.compile(r"Label1: (\w+), Label2: (\w+), Layer: (\d+), t-statistic: ([\-\d.]+), p-value: ([\d.]+), q-value: ([\d.]+), cohen_d: ([\d.]+)")
		entries = pattern.findall(raw_text)

		# Prepare nested dictionary
		results = defaultdict(lambda: [''] * 12)

		for label1, label2, layer, t_stat, p_val, q_val, cohen_d in entries:
			pair = f"{label1} vs {label2}"
			layer = int(layer)
			results[pair][layer] = f"t={float(t_stat):.2f} p={float(p_val):.2e} q={float(q_val):.2e} d={float(cohen_d):.2f}"

		# Convert to DataFrame
		df = pd.DataFrame.from_dict(results, orient='index', columns=[f"Layer {i}" for i in range(12)])
		os.makedirs("pairwise_ttest", exist_ok=True)
		if sensitivity_check:
			df.to_csv(f"pairwise_ttest/sensitivity_{key}_results.csv", index_label=True)
		elif robust_check:
			df.to_csv(f"pairwise_ttest/lower_{key}_results.csv", index_label=True)
		else:
			df.to_csv(f"pairwise_ttest/{key}_results.csv", index_label=True)

	# Global top 50 
	global_df = pd.DataFrame(columns=["comparison", "category", "layer", "t_stat", "p_value", "q_value", "cohen_d"])
	for key, data in iter_data.items():
		# pandas DataFrame index: Comparison of Labels Columns: Category, Layer, t-statistic, p-value, q-value
		# only show 2 decimal places for t-statistic, p-value, q-value (p-value and q-value can be folowed by e-notation)
		df = pd.DataFrame(data)
		df = df[["label1", "label2", "layer", "t_stat", "p_value", "q_value", "cohen_d"]]
		df["comparison"] = df["label1"] + " vs " + df["label2"]
		df = df.drop(columns=["label1", "label2"])
		df["category"] = key
		df = df[["comparison", "category", "layer", "t_stat", "p_value", "q_value", "cohen_d"]]
		df["t_stat"] = df["t_stat"].apply(lambda x: f"{x:.2f}")
		df["p_value"] = df["p_value"].apply(lambda x: f"{x:.4f}" if x >= 0.0001 else f"{x:.2e}")
		df["q_value"] = df["q_value"].apply(lambda x: f"{x:.4f}" if x >= 0.0001 else f"{x:.2e}")
		df["cohen_d"] = df["cohen_d"].apply(lambda x: f"{x:.2f}")
		global_df = pd.concat([global_df, df], ignore_index=True)
	global_df["q_value"] = global_df["q_value"].apply(lambda x: float(x))
	global_df = global_df.sort_values(by=["q_value"], ascending=True)
	global_df["q_value"] = global_df["q_value"].apply(lambda x: f"{x:.4f}" if x >= 0.0001 else f"{x:.2e}")
	top50_global = global_df.head(50)
	top50_global.set_index("comparison", inplace=True)
	if sensitivity_check:
		top50_global.to_csv("pairwise_ttest/sensitivity_top50_global_results.csv")
	elif robust_check:
		top50_global.to_csv("pairwise_ttest/lower_top50_global_results.csv")
	else:
		top50_global.to_csv("pairwise_ttest/top50_global_results.csv")
