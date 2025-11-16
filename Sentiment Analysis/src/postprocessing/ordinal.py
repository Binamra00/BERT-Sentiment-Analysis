"""
ordinal.py

This script performs the final analysis for Phase 3: Ordinal Rating Mapping.
It tests Hypothesis 3 by comparing the accuracy of star-rating predictions
derived from uncalibrated vs. calibrated probabilities.

Usage:
    python src/postprocessing/ordinal.py --run_name "bert_full_finetune_seed_123.pt"
"""

import numpy as np
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

# --- 1. Helper Functions ---

def load_data(run_name, base_dir):
    """Loads all necessary data from .npz files."""
    probs_dir = os.path.join(base_dir, "outputs", "probabilities")
    val_path = os.path.join(probs_dir, f"{run_name}_validation_outputs.npz")
    test_path = os.path.join(probs_dir, f"{run_name}_test_outputs.npz")
    
    print(f"Loading data for: {run_name}\n")
    
    try:
        val_data = np.load(val_path)
        test_data = np.load(test_path)
    except FileNotFoundError:
        print(f"Error: Probability files not found at {val_path} or {test_path}")
        print("Please ensure you have run 'run_probability_generation.py' successfully.")
        return None
    
    # Validation data (for fitting calibrator)
    val_probs = val_data['probs']
    val_labels = val_data['labels']
    
    # Test data (for final evaluation)
    test_probs = test_data['probs']
    test_labels = test_data['labels']
    test_ratings_1_10 = test_data['ratings']
    
    print(f"Loaded {len(val_probs)} validation samples.")
    print(f"Loaded {len(test_probs)} test samples (with 1-10 star ratings).")
    
    return val_probs, val_labels, test_probs, test_labels, test_ratings_1_10

def map_true_ratings(ratings_1_10):
    """
    Normalizes the 1-10 "gappy" star ratings to our 1-5 continuous scale
    as defined in the hypothesis.
    """
    # 1, 2 -> 1
    # 3, 4 -> 2
    # 7, 8 -> 4
    # 9, 10 -> 5
    rating_map = {
        1: 1, 2: 1,
        3: 2, 4: 2,
        7: 4, 8: 4,
        9: 5, 10: 5
    }
    return np.array([rating_map.get(r, 3) for r in ratings_1_10]) # Default 3 for any unknowns

def map_prob_to_rating(probs):
    """
    Linearly maps a probability [0.0, 1.0] to a star rating [1, 5].
    """
    # y = mx + b
    # m = 4 (range of 5-1)
    # b = 1 (our minimum)
    return 1 + (probs * 4)

def calculate_ordinal_metrics(y_pred, y_true):
    """Calculates all metrics for Hypothesis 3."""
    metrics = {}
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Correlation metrics
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    
    metrics['Pearson r'] = pearson_corr
    metrics['Spearman ρ'] = spearman_corr
    
    return metrics

def plot_distributions(df, plot_path):
    """
    Generates and saves a side-by-side box plot comparing
    uncalibrated and calibrated probability distributions per star rating.
    """
    print(f"\nGenerating plot and saving to {plot_path}...")
    
    # We need to "melt" the dataframe to use Seaborn's hue effectively
    df_melted = df.melt(
        id_vars=['True 1-10 Rating'],
        value_vars=['Uncalibrated Prob', 'Calibrated Prob'],
        var_name='Probability Type',
        value_name='Probability'
    )
    
    plt.figure(figsize=(14, 7))
    sns.boxplot(
        x='True 1-10 Rating',
        y='Probability',
        hue='Probability Type',
        data=df_melted,
        palette='muted'
    )
    
    plt.title('Probability Distribution vs. True Star Rating (1-10)', fontsize=16, pad=20)
    plt.xlabel('True Star Rating (from IMDb)', fontsize=12)
    plt.ylabel('Model\'s "Positive" Probability', fontsize=12)
    plt.legend(title='Probability Type', loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(plot_path, dpi=300)
    print("Plot saved successfully.")

# --- 2. Main Execution ---

def main(args):
    # --- Setup Paths ---
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..") # Project root
    plot_dir = os.path.join(base_dir, "outputs", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # --- 1. Load Data ---
    data = load_data(args.run_name, base_dir)
    if data is None:
        return
    val_probs, val_labels, test_probs, test_labels, test_ratings_1_10 = data
    
    # --- 2. Fit Calibrator ---
    print("\n--- Fitting Calibrator (Phase 2 logic) ---")
    ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    ir.fit(val_probs, val_labels)
    print("IsotonicRegression model fitted on validation data.")
    
    # --- 3. Generate Predictions for Phase 3 ---
    
    # Get calibrated probabilities
    cal_test_probs = ir.transform(test_probs)
    
    # A. Normalize the "Ground Truth" 1-10 ratings to our 1-5 scale
    true_1_5_ratings = map_true_ratings(test_ratings_1_10)
    
    # B. Map probabilities (uncalibrated) to 1-5 predictions
    pred_ratings_uncal = map_prob_to_rating(test_probs)
    
    # C. Map probabilities (calibrated) to 1-5 predictions
    pred_ratings_cal = map_prob_to_rating(cal_test_probs)
    
    # --- 4. Calculate Ordinal Metrics ---
    print("\n--- Calculating Ordinal Metrics (Phase 3) ---")
    metrics_uncal = calculate_ordinal_metrics(pred_ratings_uncal, true_1_5_ratings)
    metrics_cal = calculate_ordinal_metrics(pred_ratings_cal, true_1_5_ratings)
    
    # --- 5. Report Final Table ---
    print("\n--- Final Results (Hypothesis 3) ---")
    
    report_df = pd.DataFrame([metrics_uncal, metrics_cal], 
                             index=['Uncalibrated', 'Calibrated'])
    
    # Calculate percentage change
    change = ((report_df.loc['Calibrated'] - report_df.loc['Uncalibrated']) / 
              report_df.loc['Uncalibrated']) * 100
    report_df.loc['% Change'] = change
    
    # Formatting
    pd.set_option('display.float_format', '{:,.4f}'.format)
    
    print("Comparison of Ordinal Mapping Performance (1-5 Scale):")
    print(report_df.to_markdown(floatfmt=".4f"))
    
    # --- 6. Final Conclusion for H3 ---
    h3_confirmed = (metrics_cal['MAE'] < metrics_uncal['MAE'] and
                    metrics_cal['RMSE'] < metrics_uncal['RMSE'] and
                    metrics_cal['Pearson r'] > metrics_uncal['Pearson r'] and
                    metrics_cal['Spearman ρ'] > metrics_uncal['Spearman ρ'])
    
    if h3_confirmed:
        print("\nHypothesis 3 Confirmed: Calibrated probabilities produced more accurate ordinal ratings.")
    else:
        print("\nHypothesis 3 Not Confirmed: Calibrated probabilities did not improve all ordinal metrics.")
        
    # --- 7. Generate Visualization ---
    plot_df = pd.DataFrame({
        'Uncalibrated Prob': test_probs,
        'Calibrated Prob': cal_test_probs,
        'True 1-10 Rating': test_ratings_1_10
    })
    
    plot_path = os.path.join(plot_dir, f"{args.run_name}_ordinal_plot.png")
    plot_distributions(plot_df, plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Phase 3 Ordinal Rating Mapping analysis."
    )
    
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="The full name of the probability run to load (e.g., 'bert_full_finetune_seed123.pt')"
    )
    
    args = parser.parse_args()
    main(args)