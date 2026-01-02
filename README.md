# A Neuron-Level Analysis of Implicit Gender Bias in Vision Transformer

## Overview

This project provides a neuron-level analysis to dissect how implicit gender bias is encoded and applied across all components of a pre-trained Vision Transformer (ViT). By identifying gender-sensitive neurons and using Attn-LRP, we trace relevance to two key intra-layer components—the raw activations (post-GELU) and the normalized outputs (post-LayerNorm)—under both normal and context-masked conditions.

## Experiment Overview
![Experiment Overview](experiment_overview.png)

## Reproducibility Details

To ensure full reproducibility, we provide the key hyperparameters and environmental details for our experiments.

### Fine-tuning Hyperparameters

The `train.py` script was run for the model fine-tuning stage with the following settings:
* **Model:** `vit-base-patch16-224`
* **Dataset:** CelebA
* **Optimizer:** AdamW
* **Learning Rate:** 3e-4
* **Weight Decay:** 1e-4
* **Batch Size:** 128
* **Num Workers:** 8
* **Number of Epochs:** 50
* **Random Seed:** 42

### Analysis Parameters

* **Primary Confidence Threshold:** 0.9
* **Robustness Check Threshold:** 0.5
* **Primary Neuron Selection (k):** 5
* **Sensitivity Analysis Neuron Selection (k):** 10

### Computational Environment

* **Key Libraries:**
    * Python >=3.11
    * PyTorch 2.7.1+cu126
    * Transformers 4.54.1

## Requirements

The code is written in >= Python3.11 . The main dependencies can be installed via pip install -r requirements.txt.
*Note: Please ensure that you have the torch with CUDA support installed for GPU acceleration.*

Primary Libraries
- torch
- transformers
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- statsmodels
- pingouin
- opencv-python


## 1. Setup Instructions

### Step 1: Directory Structure
After unzipping this supplementary file, you will have the main project folder containing all the necessary scripts.

### Step 2: Download Datasets

**IEAT Image Dataset:**
The images for the main analysis are from the IEAT repository.
```bash
# Clone the IEAT repository
git clone [https://github.com/ryansteed/ieat.git](https://github.com/ryansteed/ieat.git)

# Create the necessary directories in your project root
mkdir -p data/male
mkdir -p data/female

# Copy the images into your project structure
cp ieat/data/experiments/gender/male/* data/male/
cp ieat/data/experiments/gender/female/* data/female/
```
You can delete the ieat repository after copying the images, as they are not needed for the analysis.

**CelebA Dataset:**
The model fine-tuning requires the CelebA dataset.

- Download the "Align&Cropped Images" (img_align_celeba.zip) from the official CelebA dataset page and unzip it at the root directory.

- Download the "Attribute Annotations" (list_attr_celeba.txt).

- Place list_attr_celeba.txt in the root directory.

### Step 3: Install Attn-LRP Dependency
#### 1. Clone the original Attn-LRP repository
git clone [https://github.com/rachtibat/LRP-eXplains-Transformers.git](https://github.com/rachtibat/LRP-eXplains-Transformers.git)

#### 2. Enter the repository directory
cd LRP-eXplains-Transformers

#### 3. IMPORTANT: Modify setup.py
##### Open setup.py in a text editor. Due to a character encoding issue, 
##### you must comment out or delete the two lines related to `long_description`.
#####

```python
# Find and comment out these lines:
 with open("README.md", "r") as fh:
     long_description = fh.read()

# And change the `long_description` argument to be empty:
 long_description="",
```
#### 4. Replace codes for hugging face ViT model
You need to replace the original lxt directoy in the LRP-eXplains-Transformers repository with the modified version that supports Hugging Face ViT models.
This the modified vaersion can be found among the source code files in this unzipped source file.

#### 5. Install the modified package in editable mode
pip install -e .

#### 6. Return to the root of your project directory
cd ..

## 2. Running the Experiments
The experiments are designed to be run in a sequence.

### Step 1: Data Preprocessing (Face Feathering)
Run the face_feathering.ipynb Jupyter notebook. This script reads the raw images from the data/male and data/female directories, applies the segmentation masks from your annotation files (e.g., male_annotations.json), and saves the feathered, context-free images to the masked_male_feathered/ and masked_female_feathered/ directories.

### Step 2: Model Fine-tuning
Run the training script to fine-tune the ViT model for gender classification on CelebA if you do not have "vit_gender_cls_single.pth".

```bash

python train.py
```
This will train the model and save the best-performing weights as vit_gender_cls_single.pth in the root directory. This model file is required for the subsequent LRP analysis.

### Step 3: Run Full Analysis Pipeline
The experiment.py script is the main entry point to run all analyses, including LRP relevance calculation, neuron consistency checks, heatmaps, and all statistical tests.

You can choose which version of the analysis to run by modifying the boolean flags within the script:

To run the primary analysis (for main paper results):

```bash
python experiment.py --robust_check
```
To run the robustness check (with 0.5 threshold):

```bash
python experiment.py --sensitivity_check
```
To run the sensitivity analysis (with top-10 neurons):

```bash
python experiment.py 
```
This will run the main analysis

--------------------------------------------------------------------------------
File Descriptions

train.py: Script for fine-tuning the ViT model on CelebA for gender classification.

face_feathering.ipynb: Jupyter notebook for data preprocessing (masking and feathering).

lrp_run.py: Contains the core logic for running Attn-LRP on images to generate relevance scores.

consistency.py: Calculates and plots the consistency of top-ranked neurons.

heatmap.py: Generates the aggregate cosine similarity heatmaps with significance masking.

pairwise_ttest.py: Performs the paired t-tests and FDR correction.

ind_ttest.py: Performs the independent t-tests and FDR correction.

experiment.py: The main script to orchestrate and run the full analysis pipeline.

result.ipynb: Jupyter notebook for visualizing the results of the LRP analysis.

pairwise ttest results: male_final_results.csv, female_final_results.csv, masked_female_final_results.csv, masked_male_final_results.csv, male_intermediate_results.csv, female_intermediate_results.csv, masked_female_intermediate_results.csv, masked_male_intermediate_results.csv

independent ttest results: masked_final_results.csv, masked_intermediate_results.csv, unmasked_final_results.csv, unmasked_intermediate_results.csv

consistency plot: masked_consistency_plot.png, unmasked_consistency_plot.png

