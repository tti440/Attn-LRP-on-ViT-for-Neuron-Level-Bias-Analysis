import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from scipy.stats import sem
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
import json
import math

def sem_plot(image_set, component, robust_check = False):
	scores = json.load(open(f"vit_relevance/{image_set}/scores.json"))
	if component == "final":
		comp_path = ""
	elif component == "intermediate":
		comp_path = "intermediate_"
	else:
		raise ValueError("Component must be 'final' or 'intermediate'.")

	# Set threshold for the robustness check
	threshold = 0.9 if not robust_check else 0.5

	img_path = os.listdir(f"relevance_similarity/{image_set}/")
	aggregated_scores = defaultdict(lambda: defaultdict(list))
	valid_image_count = 0

	print("Aggregating data from all images...")
	for img in img_path:
		if "png" in img or not os.path.isdir(f"relevance_similarity/{image_set}/{img}"):
			continue
		if scores.get(img, 0) < threshold:
			continue

		df = pd.read_json(f"relevance_similarity/{image_set}/{img}/{comp_path}total_result.json")
		df.index = [f"Layer {i}" for i in range(len(df.index))]
		if "beaker" in df.columns:
			df = df.drop(columns=["beaker"])

		for label in df.columns:
			for layer_name in df.index:
				aggregated_scores[label][layer_name].append(df.loc[layer_name, label])
		
		valid_image_count += 1

	print(f"Data aggregated from {valid_image_count} valid images.")

	labels = list(aggregated_scores.keys())
	layers = sorted(aggregated_scores[labels[0]].keys(), key=lambda x: int(x.split(' ')[1]))

	mean_matrix = np.zeros((len(labels), len(layers)))
	sem_matrix = np.zeros((len(labels), len(layers))) # New matrix for Standard Error

	print("Calculating mean and standard error...")
	for i, label in enumerate(labels):
		for j, layer in enumerate(layers):
			scores_list = aggregated_scores[label][layer]
			
			# Calculate mean
			mean_matrix[i, j] = np.mean(scores_list)
			
			# Calculate Standard Error of the Mean (SEM)
			sem_matrix[i, j] = sem(scores_list, nan_policy='omit')


	print("Generating final heatmaps...")
	plt.figure(figsize=(14,10))
	ax = plt.gca()
	sns.heatmap(
		sem_matrix, cmap="viridis",
		xticklabels=layers, yticklabels=labels, ax=ax
	)
	if component == "final":
		ax.set_title("Standard Error of the Mean (SEM)", fontsize=20, fontweight='bold')
	elif component == "intermediate":
		ax.set_title("Standard Error of the Mean at Intermediate(SEM)", fontsize=20, fontweight='bold')
	ax.set_xlabel("Transformer Layer", fontsize=16)
	ax.set_ylabel("Label", fontsize=16)
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontweight='bold', fontsize=14)
	ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold', fontsize=14)
	plt.tight_layout()
	if robust_check:
		plt.savefig(f"relevance_similarity/{image_set}/lower_AGGREGATE_{comp_path}heatmap_sem.png", dpi=300)
	else:
		plt.savefig(f"relevance_similarity/{image_set}/AGGREGATE_{comp_path}heatmap_sem.png", dpi=300)
	plt.close()
 

def heatmap_plot(image_set, component, robust_check=False):
	scores = json.load(open(f"vit_relevance/{image_set}/scores.json"))
	if component == "final":
		comp_path = ""
	elif component == "intermediate":
		comp_path = "intermediate_"
	else:
		raise ValueError("Component must be 'final' or 'intermediate'.")

	# Set threshold for the robustness check
	threshold = 0.9 if not robust_check else 0.5

	img_path = os.listdir(f"relevance_similarity/{image_set}/")
	aggregated_scores = defaultdict(lambda: defaultdict(list))
	valid_image_count = 0

	print("Aggregating data from all images...")
	for img in img_path:
		if not os.path.isdir(f"relevance_similarity/{image_set}/{img}"):
			continue
		if scores.get(img, 0) < threshold:
			continue

		df = pd.read_json(f"relevance_similarity/{image_set}/{img}/{comp_path}total_result.json")
		df.index = [f"Layer {i}" for i in range(len(df.index))]
		if "beaker" in df.columns:
			df = df.drop(columns=["beaker"])
		print(f"Image: {img} {scores[img]}")
	
		# Plot heatmap
		plt.figure(figsize=(14, 10))
		sns.heatmap(
			df.T,
			cmap="coolwarm",
			center=0,
			vmin=-1,      # Minimum color value (blue)
			vmax=1,       # Maximum color value (red)
			annot=False,
			linewidths=0.5,
			linecolor='gray'
		)
		if "female" in image_set:
			title_gender = "Female"
		else:
			title_gender = "Male"

		if component == "final":
			plt.title(f"Cosine Similarity per Layer(Labels vs. {title_gender} Reference)\n Conf Score: {math.ceil(scores[img] * 10000) / 10000}", fontsize=22, fontweight='bold')
		elif component == "intermediate":
			plt.title(f"Cosine Similarity at Intermediate(Labels vs. {title_gender} Reference)\n Conf Score: {math.ceil(scores[img] * 10000) / 10000}", fontsize=22, fontweight='bold')
		plt.xlabel("Transformer Layer", fontsize=18, fontweight='bold')
		plt.ylabel("Label", fontsize=18, fontweight='bold')
		plt.xticks(rotation=45, fontsize=16, fontweight='bold')
		plt.yticks(fontsize=16, fontweight='bold')
		plt.tight_layout()

		if robust_check:
			plt.savefig(f"relevance_similarity/{image_set}/lower_{comp_path}{img}_heatmap_with_mask.png", dpi=300)
		else:		
			plt.savefig(f"relevance_similarity/{image_set}/{comp_path}{img}_heatmap_with_mask.png", dpi=300)
		plt.close()
		# Append scores to our aggregated dictionary
		for label in df.columns:
			for layer_name in df.index:
				score = df.loc[layer_name, label]
				aggregated_scores[label][layer_name].append(score)
		
		valid_image_count += 1

	print(f"Data aggregated from {valid_image_count} valid images.")

	labels = list(aggregated_scores.keys())
	layers = sorted(aggregated_scores[labels[0]].keys(), key=lambda x: int(x.split(' ')[1]))

	mean_matrix = np.zeros((len(labels), len(layers)))
	pvalue_matrix = np.zeros((len(labels), len(layers)))

	print("Running statistical tests...")
	for i, label in enumerate(labels):
		for j, layer in enumerate(layers):
			scores_list = aggregated_scores[label][layer]
			
			# Calculate mean for the heatmap
			mean_matrix[i, j] = np.mean(scores_list)
			
			# Perform one-sample t-test against 0
			_, p_val = ttest_1samp(scores_list, popmean=0, nan_policy='omit')
			pvalue_matrix[i, j] = p_val

	print("Applying FDR correction...")
	reject, q_values_flat, _, _ = multipletests(pvalue_matrix.flatten(), alpha=0.05, method='fdr_bh')
	sig_mask = reject.reshape(pvalue_matrix.shape)

	print("Generating final heatmap...")
	plt.figure(figsize=(14, 10))
	ax = plt.gca()

	im = sns.heatmap(
		mean_matrix,
		cmap="coolwarm",
		center=0,
		vmin=-1,
		vmax=1,
		annot=False,
		linewidths=0.5,
		linecolor='gray',
		xticklabels=layers,
		yticklabels=labels,
		ax=ax,
		mask=~ sig_mask  # Mask out non-significant values
	)

	# Create the alpha layer and apply it
	# alpha_layer = sig_mask.astype(float) * 0.75 + 0.25
	# im.collections[0].set_alpha(alpha_layer)

	# --- Formatting and Saving ---
	if component == "final":
		plt.title(f"Mean Cosine Similarity(Labels vs. {title_gender} Reference)\nAggregated Across {valid_image_count} Images", fontsize=22, fontweight='bold')
	elif component == "intermediate":
		plt.title(f"Mean Cosine Similarity at Intermediate(Labels vs. {title_gender} Reference)\nAggregated Across {valid_image_count} Images", fontsize=22, fontweight='bold')
	plt.xlabel("Transformer Layer", fontsize=18, fontweight='bold')
	plt.ylabel("Label", fontsize=18, fontweight='bold')
	plt.xticks(rotation=45, ha="right", fontsize=14, fontweight='bold')
	plt.yticks(fontsize=14, fontweight='bold')
	plt.tight_layout()

	# Add a note about the significance masking in the figure corner or below
	plt.figtext(0.99, 0.01, 'Faded cells are not significant (FDR q > 0.05)', horizontalalignment='right', fontsize=10)
	if robust_check:
		plt.savefig(f"relevance_similarity/{image_set}/lower_AGGREGATE_{comp_path}heatmap_with_significance.png", dpi=300)
	else:
		plt.savefig(f"relevance_similarity/{image_set}/AGGREGATE_{comp_path}heatmap_with_significance.png", dpi=300)
	plt.close()
 
def run_heatmap(robust_check=False):
	for component in ["final", "intermediate"]:
		for image_set in ["male", "female", "masked_male", "masked_female"]:
			sem_plot(image_set, component, robust_check)
			heatmap_plot(image_set, component, robust_check)
			print(f"Heatmap for {image_set} ({component}) generated successfully.")