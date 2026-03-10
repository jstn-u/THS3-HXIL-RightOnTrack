"""
main.py — Orchestrates the full T.R2_LSTM_A standalone pipeline end-to-end.

Workflow:
  1. Load and preprocess data (data_loader.py)
  2. Train T.R2_LSTM_A proposed model (tr2.py + lstm_a.py)
  3. Evaluate on Train, Val, and Test splits
  4. Save results.json to outputs/T.R2_LSTM_A_{timestamp}/

Usage:
  Change SAMPLE_FRACTION below to control data size:
    1.0  = 100% of data (full run)
    0.5  = 50% of data
    0.1  = 10% of data (quick test)
    0.01 = 1% of data (smoke test)

  Or pass it as a command-line argument:
    python main.py 1.0    # full run
    python main.py 0.1    # 10% quick test

This is a standalone evaluation. Results are intended for external
comparison against a separate project. No internal baselines or PIR.

Paper: "An instance-based transfer learning model with attention mechanism
        for freight train travel time prediction in the China–Europe railway express"
DOI: https://doi.org/10.1016/j.eswa.2024.123989
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import json
import time
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from train import run_full_training

# ─── Configuration ───────────────────────────────────────────────────────────
# Change this single value to control how much data is used.
#   1.0  = 100% of each data split (full run)
#   0.5  = 50%
#   0.1  = 10% (quick test)
#   0.01 = 1%  (smoke test)

SAMPLE_FRACTION = 0.01

# Allow override from command line: python main.py 0.1
if len(sys.argv) > 1:
    try:
        SAMPLE_FRACTION = float(sys.argv[1])
    except ValueError:
        print(f"  ⚠ Invalid sample_fraction '{sys.argv[1]}', using {SAMPLE_FRACTION}")


def print_predictions_table(actuals, predictions, split_name, max_rows=None):
    """
    Print a raw per-sample predictions table similar to RightOnTrack output.
    """
    n = len(actuals)
    display_n = n if max_rows is None else min(n, max_rows)

    print(f"\n{'='*60}")
    print(f"  {split_name} — Raw Per-Sample Predictions ({display_n} of {n})")
    print(f"{'='*60}")
    print(f"  {'Idx':>5}   {'Actual(s)':>10}   {'Pred(s)':>10}   {'Error(s)':>9}   {'Error%':>8}")
    print(f"  {'-'*50}")

    for i in range(display_n):
        actual = actuals[i]
        pred = predictions[i]
        error = pred - actual
        error_pct = (error / actual) * 100 if actual != 0 else float('inf')
        print(f"  {i:>5}   {actual:>10.2f}   {pred:>10.2f}   {error:>9.2f}   {error_pct:>7.2f}%")

    # Print summary stats
    errors = predictions[:display_n] - actuals[:display_n]
    abs_errors = np.abs(errors)
    print(f"  {'-'*50}")
    print(f"  Mean Error:     {np.mean(errors):>9.2f}s")
    print(f"  Mean Abs Error: {np.mean(abs_errors):>9.2f}s")
    print(f"  Std Error:      {np.std(errors):>9.2f}s")


def main():
    """
    Full end-to-end pipeline for T.R2_LSTM_A standalone evaluation.
    """
    print("=" * 70)
    print(f"  T.R2_LSTM_A STANDALONE PIPELINE")
    print(f"  Sample fraction: {SAMPLE_FRACTION:.0%}")
    print("=" * 70)

    total_start = time.time()

    # ─── Training + Evaluation ───────────────────────────────────────────
    results = run_full_training(sample_fraction=SAMPLE_FRACTION)

    output_dir = results["output_dir"]
    results_json = results["results_json"]

    # ─── Print Results ───────────────────────────────────────────────────
    print("\n" + json.dumps(results_json, indent=2))

    # ─── Print Raw Per-Sample Predictions ────────────────────────────────
    for split_name, key in [("Train", "train"), ("Validation", "val"), ("Test (Holdout)", "test")]:
        print_predictions_table(
            actuals=results["y_true"][key],
            predictions=results["predictions"][key],
            split_name=split_name,
            max_rows=50,
        )

    # ─── Summary ─────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE — Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print("=" * 70)

    print(f"\n  T.R2_LSTM_A Results:")
    for split in ["Train", "Val", "Test"]:
        print(f"    {split}:  R2={results_json[split]['R2']}  "
              f"MSE={results_json[split]['MSE']}  "
              f"RMSE={results_json[split]['RMSE']}  "
              f"MAE={results_json[split]['MAE']}  "
              f"MAPE={results_json[split]['MAPE']}")

    print(f"\n  Output: {os.path.join(output_dir, 'results.json')}")


if __name__ == "__main__":
    main()
