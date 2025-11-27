"""
validations.py

This script performs statistical validation tests on model predictions.
It supports:
1. McNemar's Test: To compare two models (e.g., Baseline vs. BERT).
2. Bootstrap Confidence Intervals: To estimate the true accuracy range of a model.

Usage:
    # For McNemar's Test:
    python src/postprocessing/validations.py --task mcnemar --run_a "cnn_baseline.pt" --run_b "bert_full_finetune_seed123.pt"

    # For Bootstrap CI:
    python src/postprocessing/validations.py --task bootstrap --run_name "bert_full_finetune_seed123.pt"
"""

import numpy as np
import os
import argparse
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def load_predictions(run_name, base_dir):
    """Loads probabilities and labels from the .npz file."""
    # Construct path to the test outputs (we only care about test set for validation)
    # Note: run_probability_generation.py saves files as "{run_name}_test_outputs.npz"
    file_path = os.path.join(base_dir, "outputs", "probabilities", f"{run_name}_test_outputs.npz")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find probability file: {file_path}")
        
    data = np.load(file_path)
    # Convert probabilities to binary predictions (threshold 0.5)
    preds = (data['probs'] > 0.5).astype(int)
    return preds, data['labels']

def run_mcnemar(args, base_dir):
    print(f"\n--- Running McNemar's Test ---")
    print(f"Model A (Baseline): {args.run_a}")
    print(f"Model B (Challenger): {args.run_b}")

    # Load data
    preds_a, labels_a = load_predictions(args.run_a, base_dir)
    preds_b, labels_b = load_predictions(args.run_b, base_dir)

    # Sanity check: Labels must be the same
    assert np.array_equal(labels_a, labels_b), "Test sets do not match! Labels are different."
    true_labels = labels_a

    # Identify Correctness boolean arrays
    # True if correct, False if wrong
    correct_a = (preds_a == true_labels)
    correct_b = (preds_b == true_labels)

    # Build 2x2 Contingency Table
    # Table layout:
    #             | Model B Correct | Model B Wrong
    # ---------------------------------------------
    # Model A Cor |      Yes/Yes    |    Yes/No
    # Model A Wrg |      No/Yes     |     No/No
    
    # We care specifically about the Discordant pairs:
    # B = Model A Correct, Model B Wrong (Model B is worse here)
    # C = Model A Wrong, Model B Correct (Model B is better here)
    
    # Calculate cells
    both_correct = np.sum(correct_a & correct_b)
    a_corr_b_wrong = np.sum(correct_a & ~correct_b) # Cell B
    a_wrong_b_corr = np.sum(~correct_a & correct_b) # Cell C
    both_wrong = np.sum(~correct_a & ~correct_b)

    table = [[both_correct, a_corr_b_wrong],
             [a_wrong_b_corr, both_wrong]]

    print("\nContingency Table:")
    print(f"Both Correct: {both_correct}")
    print(f"{args.run_a} Correct, {args.run_b} Wrong: {a_corr_b_wrong}")
    print(f"{args.run_a} Wrong,   {args.run_b} Correct: {a_wrong_b_corr}")
    print(f"Both Wrong:   {both_wrong}")

    # Run Test
    # exact=False uses Chi-Squared (fine for N=25,000)
    result = mcnemar(table, exact=False, correction=True)

    print(f"\nMcNemar's Statistic (Chi2): {result.statistic:.3f}")
    print(f"P-Value: {result.pvalue:.10e}") # Scientific notation for very small p-values

    if result.pvalue < 0.05:
        print("\n>> RESULT: Statistically Significant Difference (p < 0.05)")
        if a_wrong_b_corr > a_corr_b_wrong:
             print(f">> CONCLUSION: {args.run_b} is significantly BETTER than {args.run_a}.")
        else:
             print(f">> CONCLUSION: {args.run_b} is significantly WORSE than {args.run_a}.")
    else:
        print("\n>> RESULT: No Statistically Significant Difference.")

def run_bootstrap(args, base_dir):
    print(f"\n--- Running Bootstrap Confidence Interval (n={args.n_bootstraps}) ---")
    print(f"Model: {args.run_name}")

    preds, labels = load_predictions(args.run_name, base_dir)
    
    n_samples = len(labels)
    accuracies = []

    print("Resampling test set...")
    for _ in tqdm(range(args.n_bootstraps)):
        # Resample indices with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        
        # Calculate accuracy on this "bootstrap sample"
        acc = accuracy_score(labels[indices], preds[indices])
        accuracies.append(acc)

    # Calculate percentiles
    lower = np.percentile(accuracies, 2.5) * 100
    upper = np.percentile(accuracies, 97.5) * 100
    mean_acc = np.mean(accuracies) * 100

    print(f"\nMean Accuracy: {mean_acc:.2f}%")
    print(f"95% Confidence Interval: [{lower:.2f}%, {upper:.2f}%]")
    print(f"Margin of Error: +/- {(upper - lower)/2:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Statistical Validations")
    parser.add_argument("--task", type=str, required=True, choices=['mcnemar', 'bootstrap'], help="Which test to run")
    
    # Arguments for McNemar
    parser.add_argument("--run_a", type=str, help="Run name for Model A (Baseline)")
    parser.add_argument("--run_b", type=str, help="Run name for Model B (Comparison)")
    
    # Arguments for Bootstrap
    parser.add_argument("--run_name", type=str, help="Run name for single model analysis")
    parser.add_argument("--n_bootstraps", type=int, default=1000, help="Number of bootstrap iterations")

    args = parser.parse_args()
    
    # Define project root (assuming script is in src/postprocessing/)
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")

    if args.task == 'mcnemar':
        if not args.run_a or not args.run_b:
            print("Error: McNemar's test requires --run_a and --run_b")
            return
        run_mcnemar(args, base_dir)
        
    elif args.task == 'bootstrap':
        if not args.run_name:
            print("Error: Bootstrap requires --run_name")
            return
        run_bootstrap(args, base_dir)

if __name__ == "__main__":
    main()