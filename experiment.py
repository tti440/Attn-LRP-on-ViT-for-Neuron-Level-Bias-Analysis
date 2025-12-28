from lrp_run import run_lrp_analysis, gender_sensitive_neurons
from consistency import plot_consistency
from heatmap import run_heatmap
from pairwise_ttest import pairwise_fdr_correction
from ind_ttest import ind_fdr_correction
import argparse

def experiment(robust_check, sensitivity_check):
	print("Starting experiment...")
	run_lrp_analysis()
	gender_sensitive_neurons(sensitivity=sensitivity_check)
	plot_consistency(robust_check=robust_check, sensitivity_check=sensitivity_check)
	run_heatmap(robust_check=robust_check)
	pairwise_fdr_correction(robust_check=robust_check, sensitivity_check=sensitivity_check)
	ind_fdr_correction(robust_check=robust_check, sensitivity_check=sensitivity_check)
	print("Experiment completed successfully.")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run the experiment.")
	parser.add_argument("--robust", action="store_true", help="Enable robust check.")
	parser.add_argument("--sensitivity", action="store_true", help="Enable sensitivity check.")
	args = parser.parse_args()
	experiment(robust_check=args.robust, sensitivity_check=args.sensitivity)