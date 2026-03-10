"""
train.py — Training loop for T.R2_LSTM_A model (Paper Section 4)

Orchestrates:
  1. Data loading and sequence construction
  2. Min-Max scaling of both X and y (Paper Equation 10)
  3. T.R2_LSTM_A model training on normalized data
  4. Inverse-transform predictions back to seconds
  5. Evaluate on Train, Val, and Test splits
  6. Save results as JSON to outputs/T.R2_LSTM_A_{timestamp}/

No baseline models are trained. Results are for external comparison only.

Usage:
  Adjust SAMPLE_FRACTION to control how much of the data to use:
    1.0 = 100% of each split (full run)
    0.1 = 10% of each split (quick test)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import time
from datetime import datetime

from data_loader import load_all_datasets, scale_data
from tr2 import TrAdaBoostR2
from evaluate import (
    compute_all_metrics, build_results_json, save_results_json
)


# ─── Configuration (Paper Table 5) ──────────────────────────────────────────

LSTM_UNITS = 32        # changed from paper -- increase from 16 (more capacity)
DROPOUT = 0.3          # changed from paper -- reduce from 0.5 (less regularization)
N_ESTIMATORS = 5       # Q: number of AdaBoost.R2 base learners per step
EPOCHS = 150            # changed from paper -- reduce from 150 (earlier stopping)
BATCH_SIZE = 128
TR2_STEPS = 5          # changed from paper -- S: TrAdaBoost iterations. reduce from 5 (less aggressive weight decay)
LEARNING_RATE = 0.001  # Adam optimizer

# ─── Data Fraction ───────────────────────────────────────────────────────────
# Controls what percentage of each data split to use.
#   1.0 = use 100% of the data (full run)
#   0.1 = use 10% of the data (quick test)
SAMPLE_FRACTION = 1.0

# Base output directory
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def create_run_output_dir():
    """
    Create a timestamped output folder: outputs/T.R2_LSTM_A_{timestamp}/
    Returns the path and the timestamp string.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_OUTPUT_DIR, f"T.R2_LSTM_A_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, timestamp


def subsample(X, y, fraction, seed=42):
    """
    Randomly subsample a fraction of the data.
    Works with 3D X arrays (trips, timesteps, features).
    """
    if fraction >= 1.0:
        return X, y
    n = len(X)
    k = max(1, int(n * fraction))
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=k, replace=False)
    idx.sort()
    return X[idx], y[idx]


def train_tr2_lstm_a(X_source, y_source, X_target, y_target,
                     X_val=None, y_val=None, input_shape=None):
    """
    Train the main T.R2_LSTM_A model — Paper's proposed method.
    Two-Stage TrAdaBoost.R2 with LSTM_A as base learner.

    All inputs (X and y) should already be normalized to [0, 1].
    """
    print(f"\n{'='*60}")
    print("Training T.R2_LSTM_A")
    print(f"{'='*60}")

    model = TrAdaBoostR2(
        n_estimators=N_ESTIMATORS,
        n_steps=TR2_STEPS,
        lstm_units=LSTM_UNITS,
        dropout=DROPOUT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        input_shape=input_shape,
        verbose=1,
    )

    start = time.time()
    model.fit(X_source, y_source, X_target, y_target, X_val, y_val)
    elapsed = time.time() - start
    print(f"\n  ✓ T.R2_LSTM_A trained in {elapsed:.1f}s")

    return model


def run_full_training(sample_fraction=SAMPLE_FRACTION):
    """
    Complete training pipeline for T.R2_LSTM_A only.

    Parameters
    ----------
    sample_fraction : float
        Fraction of each data split to use (0.0–1.0).
        1.0 = full data, 0.1 = 10%, etc.

    Returns
    -------
    dict with: metrics per split, convergence history, output_dir path
    """
    # Create timestamped output folder
    output_dir, timestamp_str = create_run_output_dir()
    timestamp_display = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n  Output folder: {output_dir}")

    # ─── Step 1: Load Data (with sequence construction) ──────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Loading Data & Building Sequences")
    print("=" * 60)
    print(f"  Sample fraction: {sample_fraction:.0%}")

    # Pass nrows proportional to sample_fraction to avoid loading
    # the full dataset into memory for smoke/quick tests.
    # Approximate row counts based on file size ratios.
    if sample_fraction >= 1.0:
        source_nrows = target_nrows = val_nrows = holdout_nrows = None
    else:
        source_nrows   = int(2_000_000 * sample_fraction)
        target_nrows   = int(700_000  * sample_fraction)
        val_nrows      = int(350_000  * sample_fraction)
        holdout_nrows  = int(350_000  * sample_fraction)

    data = load_all_datasets(
        source_nrows=source_nrows,
        target_nrows=target_nrows,
        val_nrows=val_nrows,
        holdout_nrows=holdout_nrows,
    )

    # Unpack 3D arrays: (trips, timesteps, features) and 1D targets (raw seconds)
    X_source_raw, y_source_raw, _ = data["source"]
    X_target_raw, y_target_raw, _ = data["target"]
    X_val_raw, y_val_raw, _ = data["validation"]
    X_holdout_raw, y_holdout_raw, _ = data["holdout"]

    # ─── Step 2: Subsample ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Subsampling Data")
    print("=" * 60)

    X_source_raw, y_source_raw = subsample(X_source_raw, y_source_raw, sample_fraction, seed=42)
    X_target_raw, y_target_raw = subsample(X_target_raw, y_target_raw, sample_fraction, seed=43)
    X_val_raw, y_val_raw       = subsample(X_val_raw, y_val_raw, sample_fraction, seed=44)
    X_holdout_raw, y_holdout_raw = subsample(X_holdout_raw, y_holdout_raw, sample_fraction, seed=45)

    print(f"  Source (Train): {len(y_source_raw):,} trips")
    print(f"  Target:         {len(y_target_raw):,} trips")
    print(f"  Validation:     {len(y_val_raw):,} trips")
    print(f"  Holdout (Test): {len(y_holdout_raw):,} trips")

    # ─── Step 3: Scale Features AND Target (Min-Max to [0,1] — Paper Eq. 10)
    print("\n" + "=" * 60)
    print("STEP 3: Scaling Features & Target (Min-Max → [0,1])")
    print("=" * 60)

    scaled = scale_data(
        X_source_raw, y_source_raw,
        X_target_raw, y_target_raw,
        X_val_raw, y_val_raw,
        X_holdout_raw, y_holdout_raw,
    )

    X_source = scaled["source_X"]
    X_target = scaled["target_X"]
    X_val = scaled["validation_X"]
    X_holdout = scaled["holdout_X"]

    y_source_scaled = scaled["source_y"]
    y_target_scaled = scaled["target_y"]
    y_val_scaled = scaled["validation_y"]
    y_holdout_scaled = scaled["holdout_y"]

    y_scaler = scaled["y_scaler"]

    seq_len = X_source.shape[1]
    n_features = X_source.shape[2]
    input_shape = (seq_len, n_features)

    print(f"  Sequence length: {seq_len} timesteps")
    print(f"  Features: {n_features}")
    print(f"  Input shape for LSTM: {input_shape}")
    print(f"  X range: [{X_source.min():.4f}, {X_source.max():.4f}]")
    print(f"  y range: [{y_source_scaled.min():.4f}, {y_source_scaled.max():.4f}]")

    # ─── Step 4: Train T.R2_LSTM_A (on normalized data) ─────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Training T.R2_LSTM_A")
    print("=" * 60)

    tr2_lstm_a_model = train_tr2_lstm_a(
        X_source, y_source_scaled, X_target, y_target_scaled,
        X_val=X_val, y_val=y_val_scaled, input_shape=input_shape,
    )

    # ─── Print Training Convergence Summary ──────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING CONVERGENCE SUMMARY")
    print("=" * 60)

    convergence = tr2_lstm_a_model.get_convergence_history()

    # Print per-step TrAdaBoost errors
    print(f"\n  TrAdaBoost Step Errors (target CV error per step):")
    print(f"  {'Step':>6}   {'Stage':>8}   {'Target Error':>14}")
    print(f"  {'-'*34}")
    n_steps = tr2_lstm_a_model.n_steps
    for i, err in enumerate(convergence["errors"]):
        stage = "Stage 1" if i < n_steps else "Stage 2"
        step_in_stage = (i % n_steps) + 1
        print(f"  {i+1:>6}   {stage:>8}   {err:>14.6f}")
    best_idx = tr2_lstm_a_model.best_idx_
    print(f"\n  Best step: {best_idx + 1} (error: {convergence['errors'][best_idx]:.6f})")

    # Print final epoch losses from Stage 1 and Stage 2 histories
    for stage_name, histories in [("Stage 1 (Plain LSTM)", convergence["stage1"]),
                                   ("Stage 2 (LSTM_A)", convergence["stage2"])]:
        if histories:
            print(f"\n  {stage_name} — Training Loss (last epoch per estimator):")
            print(f"  {'Estimator':>11}   {'Final Loss':>12}   {'Final MAE':>12}")
            print(f"  {'-'*40}")
            for j, h in enumerate(histories):
                final_loss = h.get("loss", [None])[-1]
                final_mae = h.get("mae", [None])[-1]
                loss_str = f"{final_loss:.6f}" if final_loss is not None else "N/A"
                mae_str = f"{final_mae:.6f}" if final_mae is not None else "N/A"
                print(f"  {j+1:>11}   {loss_str:>12}   {mae_str:>12}")

    # ─── Step 5: Predict & Inverse-Transform ─────────────────────────────
    # Model outputs normalized [0,1] predictions. We inverse-transform
    # back to raw seconds so metrics (especially MAPE) are meaningful.
    print("\n" + "=" * 60)
    print("STEP 5: Generating Predictions & Inverse-Transforming")
    print("=" * 60)

    print("  Predicting on Train (source)...")
    preds_train_scaled = tr2_lstm_a_model.predict(X_source)
    preds_train = y_scaler.inverse_transform(preds_train_scaled.reshape(-1, 1)).flatten()

    print("  Predicting on Validation...")
    preds_val_scaled = tr2_lstm_a_model.predict(X_val)
    preds_val = y_scaler.inverse_transform(preds_val_scaled.reshape(-1, 1)).flatten()

    print("  Predicting on Test (holdout)...")
    preds_test_scaled = tr2_lstm_a_model.predict(X_holdout)
    preds_test = y_scaler.inverse_transform(preds_test_scaled.reshape(-1, 1)).flatten()

    # ─── Step 6: Compute Metrics (R², RMSE, MAE, MAPE) ──────────────────
    # All metrics computed on RAW SECONDS (after inverse transform) so
    # MAPE divides by y_true in seconds giving a meaningful percentage.
    print("\n" + "=" * 60)
    print("STEP 6: Computing Metrics (on raw seconds)")
    print("=" * 60)

    train_metrics = compute_all_metrics(y_source_raw, preds_train)
    val_metrics   = compute_all_metrics(y_val_raw, preds_val)
    test_metrics  = compute_all_metrics(y_holdout_raw, preds_test)

    for label, m in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
        print(f"  {label}:")
        print(f"    R2:   {m['R2']:.4f}")
        print(f"    MSE:  {m['MSE']:.4f} seconds²")
        print(f"    RMSE: {m['RMSE']:.2f} seconds")
        print(f"    MAE:  {m['MAE']:.2f} seconds")
        print(f"    MAPE: {m['MAPE']:.2f}%")

    # ─── Step 7: Save results as JSON ────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: Saving Results")
    print("=" * 60)

    config = {
        "sample_fraction": sample_fraction,
        "sequence_length": seq_len,
        "n_features": n_features,
        "lstm_units": LSTM_UNITS,
        "dropout": DROPOUT,
        "n_estimators": N_ESTIMATORS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "tr2_steps": TR2_STEPS,
        "learning_rate": LEARNING_RATE,
        "train_trips": len(y_source_raw),
        "val_trips": len(y_val_raw),
        "test_trips": len(y_holdout_raw),
    }

    results_json = build_results_json(
        config=config,
        timestamp=timestamp_display,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )

    json_path = os.path.join(output_dir, "results.json")
    save_results_json(results_json, json_path)

    # Return everything needed by main.py
    return {
        "predictions": {"train": preds_train, "val": preds_val, "test": preds_test},
        "y_true": {"train": y_source_raw, "val": y_val_raw, "test": y_holdout_raw},
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "convergence": tr2_lstm_a_model.get_convergence_history(),
        "sample_fraction": sample_fraction,
        "output_dir": output_dir,
        "results_json": results_json,
    }


if __name__ == "__main__":
    results = run_full_training(sample_fraction=0.01)  # 1% for quick test
    print("\nTraining pipeline test complete!")
