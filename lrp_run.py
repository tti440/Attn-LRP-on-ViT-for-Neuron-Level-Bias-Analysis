import torch
from transformers import ViTForImageClassification, ViTImageProcessorFast
import torch.nn as nn
from lxt.efficient import monkey_patch, monkey_patch_zennit
from PIL import Image
import pickle
import os
import json
from collections import defaultdict
from lrp_utils import hidden_relevance_hook, relevance_retrieval
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Monkey patching (run once)
monkey_patch("vit", verbose=False)
monkey_patch_zennit(verbose=False)

def lrp_analysis(model, labels, input_dir, output_dir_base, device="cuda", save_scores_path=None):
	"""
	Runs the LRP analysis for a given model, set of labels, and image directory.

	Args:
		model (nn.Module): The PyTorch model to analyze (already on the correct device).
		labels (dict): A dictionary of {label_name: class_index}.
		input_dir (str): Path to the directory containing input images.
		output_dir_base (str): Base path to save the relevance results.
		device (str): The device to run on (e.g., "cuda" or "cpu").
		save_scores_path (str, optional): If provided, saves classification scores to this JSON file.
	"""
	print(f"--- Running analysis for input '{input_dir}' ---")
	score_predictions = defaultdict(float)

	processor = ViTImageProcessorFast.from_pretrained("google/vit-base-patch16-224")
	model.eval()

	# Register hooks on all layers
	for layer in model.vit.encoder.layer:
		layer.intermediate.register_full_backward_hook(hidden_relevance_hook)
		layer.register_full_backward_hook(hidden_relevance_hook)
	
	for filename in os.listdir(input_dir):
		print(f"Processing {filename}...")
		image_path = os.path.join(input_dir, filename)
		image = Image.open(image_path).convert('RGB')
		input_tensor = processor(images=image, return_tensors='pt')['pixel_values'].to(device)


		for label_name, class_index in labels.items():
			model.zero_grad() # Reset gradients for each label
			
			output_logits = model(input_tensor.requires_grad_()).logits
			softmax_score = torch.nn.functional.softmax(output_logits, dim=-1)
			y = output_logits[0, class_index]
			score = softmax_score[0, class_index]	
			y.backward()
			subdir_base = output_dir_base.split("/")[-1]
			relevance_retrieval("vit", model, subdir_base, filename, class_index, score.item())
			# Extract and Process Relevance Scores 
			relevance_trace = []
			relevance_intermediate = []
			for layer in model.vit.encoder.layer:
				# Final component (Post-LayerNorm)
				relevance_final = layer.hidden_relevance[0].sum(0)
				relevance_final = relevance_final / relevance_final.abs().max()
				relevance_trace.append(relevance_final.cpu().detach())

				# Intermediate component (Post-GELU)
				relevance_inter = layer.intermediate.hidden_relevance[0].sum(0)
				relevance_inter = relevance_inter / relevance_inter.abs().max()
				relevance_intermediate.append(relevance_inter.cpu().detach())

			relevance_trace = torch.stack(relevance_trace)
			relevance_intermediate = torch.stack(relevance_intermediate)

	
			image_id = filename.split("_")[0]
			save_dir = os.path.join(output_dir_base, image_id)
			os.makedirs(save_dir, exist_ok=True)

			with open(os.path.join(save_dir, f"relevance_trace_{label_name}_vit.pkl"), "wb") as f:
				pickle.dump(relevance_trace, f)
			with open(os.path.join(save_dir, f"relevance_intermediate_{label_name}_vit.pkl"), "wb") as f:
				pickle.dump(relevance_intermediate, f)
			score_key = filename.split("_")[0]
			score_predictions[score_key] = score.item()
	
	if save_scores_path:
		with open(os.path.join(output_dir_base, save_scores_path), "w") as f:
			json.dump(score_predictions, f, indent=4)
		print(f"Confidence scores saved to {save_scores_path}")
  
def similarity_lrp(image_type, component):
	all_vectors = []
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
		"library" : 624,
		"male":1,
		"female":0
	}
	
	if image_type == "masked":
		image_set = ["masked_female", "masked_male"]
	elif image_type == "unmasked":
		image_set = ["female", "male"]
	else:
		raise ValueError("image_type must be 'masked' or 'unmasked'")
	if component == "final":
		comp_path = "trace"
	elif component == "intermediate":
		comp_path = "intermediate"
	else:
		raise ValueError("component must be 'final' or 'intermediate'")
	if component == "final":
		save_path = ""
	elif component == "intermediate":
		save_path = "intermediate_"
	else:
		raise ValueError("component must be 'final' or 'intermediate'")

	# Collect all relevance vectors
	for gender in image_set:
		skip_label = "male" if gender == "masked_female" or gender == "female" else "female"
		img_path = os.path.join("vit_relevance", gender)
		for img in os.listdir(img_path):
			if ".json" in img:
				continue
			for index, label in labels.items():
				if index == skip_label:
					continue
				full_path = os.path.join(img_path, img, f"relevance_{comp_path}_{index}_vit.pkl")
				data = pickle.load(open(full_path, "rb"))

				for layer_vector in data: 
					all_vectors.append(layer_vector)

	# Get global mean and std for normalizing lrp scores
	all_vectors = np.array(all_vectors) 
	global_mean = all_vectors.mean(axis=0)
	global_std = all_vectors.std(axis=0)
	global_std = np.where(global_std == 0, 1e-4, global_std)
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
	for img_dir in image_set:
		img_path = os.listdir(f"vit_relevance/{img_dir}")
		total_result = {}
		for img in img_path:
			if ".json" in img:
				continue
			if "female" in img_dir:
				full_path = os.path.join(f"vit_relevance/{img_dir}", img, f"relevance_{comp_path}_female_vit.pkl")
			else:
				full_path = os.path.join(f"vit_relevance/{img_dir}", img, f"relevance_{comp_path}_male_vit.pkl")
			data = pickle.load(open(full_path, "rb"))
			normalized_res_female = {}
			for i, values in enumerate(data):
				values = np.array(values)
				normalized_res_female[i] = (values - global_mean) / global_std
			for label, index in labels.items():
				full_path = os.path.join(f"vit_relevance/{img_dir}", img, f"relevance_{comp_path}_{label}_vit.pkl")
				data = pickle.load(open(full_path, "rb"))
				normalized_res_label = {}
				for layer, values in enumerate(data):
					values = np.array(values)
					normalized_res_label[layer] = (values - global_mean) / global_std
		
				cosine_similarities = {}
				for layer, values in normalized_res_label.items():
					cosine_similarities[layer] = cosine_similarity(
					normalized_res_label[layer].reshape(1, -1),
					normalized_res_female[layer].reshape(1, -1)
				)[0, 0]
				#save as json
				name = img.split(".")[0]
				output=f"relevance_similarity/{img_dir}"
				if not os.path.exists(output):
					os.makedirs(output)
				if not os.path.exists(f"relevance_similarity/{img_dir}/{name}"):
					os.makedirs(f"relevance_similarity/{img_dir}/{name}")
				cosine_similarities = {int(k): float(v) for k, v in cosine_similarities.items()}
				with open(f"relevance_similarity/{img_dir}/{name}/relevance_{save_path}similarity_{label}.json", "w") as f:
					json.dump(cosine_similarities, f, indent=4)
				total_result[label] = [value for value in cosine_similarities.values()]
			# Save the total result for the image
			with open(f"relevance_similarity/{img_dir}/{name}/{save_path}total_result.json", "w") as f:
				json.dump(total_result, f, indent=4)
		

def load_imagenet_model():
	"""Loads the pre-trained ImageNet ViT model."""
	model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
	model.to("cuda")
	return model

def load_gender_model():
	"""Loads the fine-tuned Gender Classification ViT model."""
	model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
	model.classifier = torch.nn.Linear(model.config.hidden_size, 2)
	model.load_state_dict(torch.load("vit_gender_cls_single.pth", map_location="cpu"), strict=True)
	model.to("cuda")
	return model

def run_lrp_analysis():
	# Define Labels for each analysis. This is from ImageNet Labels
	imagenet_labels = {
		"labcoat": 617, "stethoscope": 823, "jeans": 608, "cardigan": 474,
		"miniskirt": 655, "academicgown": 400, "pajamas": 697,
		"oscilloscope": 688, "suit": 834, "notePC": 681, "sunscreen": 838,
		"wig": 903, "lipstick": 629, "bra": 459, "windsor": 906,
		"desk": 526, "library": 624
	}
	
	# Note: For binary gender, you'd typically have two labels.
	# Here we define one for each analysis run, following celebA attribute .txt.
	male_gender_label = {"male": 1}
	female_gender_label = {"female": 0}

	# Define directory mappings
	# {input_dir: output_dir_base}
	dir_mappings = {
		"male": "vit_relevance/male",
		"female": "vit_relevance/female",
		"masked_male_feathered": "vit_relevance/masked_male",
		"masked_female_feathered": "vit_relevance/masked_female"
	}
	
	# Run ImageNet-based LRP Analysis
	print("="*50)
	print("STARTING IMAGENET LRP ANALYSIS")
	print("="*50)
	imagenet_model = load_imagenet_model()
	for input_d, output_d in dir_mappings.items():
		lrp_analysis(imagenet_model, imagenet_labels, input_d, output_d)

	# --- Run Gender-Classifier-based LRP Analysis ---
	print("\n" + "="*50)
	print("STARTING GENDER CLASSIFIER LRP ANALYSIS")
	print("="*50)
	gender_model = load_gender_model()
	
	# Male images
	lrp_analysis(gender_model, male_gender_label, "male", "vit_relevance/male", save_scores_path="scores.json")
	lrp_analysis(gender_model, male_gender_label, "masked_male_feathered", "vit_relevance/masked_male", save_scores_path="scores.json")
	
	# Female images
	lrp_analysis(gender_model, female_gender_label, "female", "vit_relevance/female", save_scores_path="scores.json")
	lrp_analysis(gender_model, female_gender_label, "masked_female_feathered", "vit_relevance/masked_female", save_scores_path="scores.json")

	print("\nAll analyses complete.")
	
	print("Running similarity analysis...")
	similarity_lrp("masked", "final")
	similarity_lrp("masked", "intermediate")
	similarity_lrp("unmasked", "final")
	similarity_lrp("unmasked", "intermediate")
	print("Similarity analysis complete.")

def gender_sensitive_neurons(sensitivity=False):
	if sensitivity:
		k=10
	else:
		k=5
	print("Extracting gender-sensitive neurons...")
	dirs = os.listdir("vit_relevance")
	for dir in dirs:
		img_paths = os.listdir(os.path.join("vit_relevance", dir))
		root = os.path.join("vit_relevance", dir)
		for img_path in img_paths:
			if ".json" in img_path:
				continue
			if "female" in dir:
				gender = "female"
			else:
				gender = "male"
			full_path = os.path.join("vit_relevance", dir, img_path, f"relevance_trace_{gender}_vit.pkl")
			data = pickle.load(open(full_path, "rb"))
			results = defaultdict(dict)
			for layer, values in enumerate(data):
				values = np.array(values)
				absvalues = abs(values)
				topk = np.argsort(absvalues)[-k:][::-1]
				results[layer]["top5_index"] = topk.tolist()
				results[layer]["scores"] = values[topk].tolist()
			with open(os.path.join(root, img_path, f"top{k}_relevance_{gender}_vit.json"), "w") as f:
				json.dump(results, f, indent=4)
	for dir in dirs:
		img_paths = os.listdir(os.path.join("vit_relevance", dir))
		root = os.path.join("vit_relevance", dir)
		for img_path in img_paths:
			if ".json" in img_path:
				continue
			if "female" in dir:
				gender = "female"
			else:
				gender = "male"
			full_path = os.path.join("vit_relevance", dir, img_path, f"relevance_intermediate_{gender}_vit.pkl")
			data = pickle.load(open(full_path, "rb"))
			results = defaultdict(dict)
			for layer, values in enumerate(data):
				values = np.array(values)
				absvalues = abs(values)
				topk = np.argsort(absvalues)[-k:][::-1]
				results[layer]["top5_index"] = topk.tolist()
				results[layer]["scores"] = values[topk].tolist()
			with open(os.path.join(root, img_path, f"top{k}_intermediate_relevance_{gender}_vit.json"), "w") as f:
				json.dump(results, f, indent=4)
	print("Gender-sensitive neurons extraction complete.")