"""
validations.py

This script performs statistical validation tests on model predictions.
It supports:
1. McNemar's Test: To compare two models (e.g., Baseline vs. BERT).
2. Bootstrap Confidence Intervals: To estimate the true accuracy range of a model.

Results are printed to the console and saved to JSON files in outputs/metrics/.

Usage:
    # For McNemar's Test:
    python src/postprocessing/validations.py --task mcnemar --run_a "cnn_baseline.pt" --run_b "bert_full_finetune_seed123.pt"

    # For Bootstrap CI:
    python src/postprocessing/validations.py --task bootstrap --run_name "bert_full_finetune_seed123.pt"
"""

import numpy as np
import os
import argparse
import json
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def load_predictions(run_name, base_dir):
    """Loads probabilities and labels from the .npz file."""
    file_path = os.path.join(base_dir, "outputs", "probabilities", f"{run_name}_test_outputs.npz")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find probability file: {file_path}")

    data = np.load(file_path)
    # Convert probabilities to binary predictions (threshold 0.5)
    preds = (data['probs'] > 0.5).astype(int)
    return preds, data['labels']

def save_metrics(metrics_dict, filename, base_dir):
    """Saves a dictionary of metrics to a JSON file."""
    metrics_dir = os.path.join(base_dir, "outputs", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    file_path = os.path.join(metrics_dir, filename)
    with open(file_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"\n[INFO] Metrics saved to {file_path}")

def run_mcnemar(args, base_dir):
    print(f"\n--- Running McNemar's Test ---")
    print(f"Model A (Baseline): {args.run_a}")
    print(f"Model B (Challenger): {args.run_b}")

    # Load data
    preds_a, labels_a = load_predictions(args.run_a, base_dir)
    preds_b, labels_b = load_predictions(args.run_b, base_dir)

    # Sanity check
    assert np.array_equal(labels_a, labels_b), "Test sets do not match! Labels are different."
    true_labels = labels_a

    # Identify Correctness
    correct_a = (preds_a == true_labels)
    correct_b = (preds_b == true_labels)

    # Calculate cells
    both_correct = int(np.sum(correct_a & correct_b))
    a_corr_b_wrong = int(np.sum(correct_a & ~correct_b)) # Cell B
    a_wrong_b_corr = int(np.sum(~correct_a & correct_b)) # Cell C
    both_wrong = int(np.sum(~correct_a & ~correct_b))

    table = [[both_correct, a_corr_b_wrong],
             [a_wrong_b_corr, both_wrong]]

    print("\nContingency Table:")
    print(f"Both Correct: {both_correct}")
    print(f"{args.run_a} Correct, {args.run_b} Wrong: {a_corr_b_wrong}")
    print(f"{args.run_a} Wrong,   {args.run_b} Correct: {a_wrong_b_corr}")
    print(f"Both Wrong:   {both_wrong}")

    # Run Test
    result = mcnemar(table, exact=False, correction=True)

    print(f"\nMcNemar's Statistic (Chi2): {result.statistic:.3f}")
    print(f"P-Value: {result.pvalue:.10e}")

    conclusion = "No Statistically Significant Difference."
    if result.pvalue < 0.05:
        print("\n>> RESULT: Statistically Significant Difference (p < 0.05)")
        if a_wrong_b_corr > a_corr_b_wrong:
             conclusion = f"{args.run_b} is significantly BETTER than {args.run_a}."
        else:
             conclusion = f"{args.run_b} is significantly WORSE than {args.run_a}."
        print(f">> CONCLUSION: {conclusion}")
    else:
        print("\n>> RESULT: No Statistically Significant Difference.")

    # Save Results
    metrics = {
        "test_type": "McNemar's Test",
        "model_a": args.run_a,
        "model_b": args.run_b,
        "contingency_table": {
            "both_correct": both_correct,
            "model_a_correct_model_b_wrong": a_corr_b_wrong,
            "model_a_wrong_model_b_correct": a_wrong_b_corr,
            "both_wrong": both_wrong
        },
        "chi2_statistic": result.statistic,
        "p_value": result.pvalue,
        "conclusion": conclusion
    }
    save_metrics(metrics, f"mcnemar_{args.run_a}_vs_{args.run_b}.json", base_dir)

def run_bootstrap(args, base_dir):
    print(f"\n--- Running Bootstrap Confidence Interval (n={args.n_bootstraps}) ---")
    print(f"Model: {args.run_name}")

    preds, labels = load_predictions(args.run_name, base_dir)

    n_samples = len(labels)
    accuracies = []

    print("Resampling test set...")
    for _ in tqdm(range(args.n_bootstraps)):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        acc = accuracy_score(labels[indices], preds[indices])
        accuracies.append(acc)

    # Calculate stats
    lower = np.percentile(accuracies, 2.5)
    upper = np.percentile(accuracies, 97.5)
    mean_acc = np.mean(accuracies)
    margin_error = (upper - lower) / 2

    print(f"\nMean Accuracy: {mean_acc*100:.2f}%")
    print(f"95% Confidence Interval: [{lower*100:.2f}%, {upper*100:.2f}%]")
    print(f"Margin of Error: +/- {margin_error*100:.2f}%")

    # Save Results
    metrics = {
        "test_type": "Bootstrap Confidence Interval",
        "model": args.run_name,
        "n_bootstraps": args.n_bootstraps,
        "mean_accuracy": mean_acc,
        "ci_lower_95": lower,
        "ci_upper_95": upper,
        "margin_of_error": margin_error
    }
    save_metrics(metrics, f"bootstrap_{args.run_name}.json", base_dir)

def main():
    parser = argparse.ArgumentParser(description="Statistical Validations")
    parser.add_argument("--task", type=str, required=True, choices=['mcnemar', 'bootstrap'], help="Which test to run")
    parser.add_argument("--run_a", type=str, help="Run name for Model A (Baseline)")
    parser.add_argument("--run_b", type=str, help="Run name for Model B (Comparison)")
    parser.add_argument("--run_name", type=str, help="Run name for single model analysis")
    parser.add_argument("--n_bootstraps", type=int, default=1000, help="Number of bootstrap iterations")

    args = parser.parse_args()

    # Define project root
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