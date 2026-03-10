"""
evaluate.py — Evaluation metrics (Paper Equations 11-14 + R²)

Implements:
  - R²   (Coefficient of Determination)
  - MAE  (Equation 11)
  - MAPE (Equation 12)
  - MSE  (Equation 13)
  - RMSE (Equation 14)
"""

import json
import numpy as np


# ─── Core Metrics (Paper Equations 11-14 + R²) ──────────────────────────────

def compute_r2(y_true, y_pred):
    """
    Coefficient of Determination (R²).
    R² = 1 - SS_res / SS_tot
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def compute_mae(y_true, y_pred):
    """
    Mean Absolute Error — Paper Equation 11
    MAE = (1/N) * Σ|y'_t - y_t|
    """
    return np.mean(np.abs(y_pred - y_true))


def compute_mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error — Paper Equation 12
    MAPE = (100/N) * Σ(|y'_t - y_t| / y_t)

    Note: Filters out y_true == 0 to avoid division by zero.
    """
    mask = y_true != 0
    return 100.0 * np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask]))


def compute_mse(y_true, y_pred):
    """
    Mean Squared Error — Paper Equation 13
    MSE = Σ(y'_t - y_t)² / N
    """
    return np.mean((y_pred - y_true) ** 2)


def compute_rmse(y_true, y_pred):
    """
    Root Mean Squared Error — Paper Equation 14
    RMSE = √(Σ(y'_t - y_t)² / N)
    """
    return np.sqrt(compute_mse(y_true, y_pred))


def compute_all_metrics(y_true, y_pred):
    """
    Compute all five regression metrics.

    Returns
    -------
    dict with keys: R2, RMSE, MAE, MAPE, MSE
    """
    return {
        "R2": compute_r2(y_true, y_pred),
        "RMSE": compute_rmse(y_true, y_pred),
        "MAE": compute_mae(y_true, y_pred),
        "MAPE": compute_mape(y_true, y_pred),
        "MSE": compute_mse(y_true, y_pred),
    }


def format_metrics_for_json(metrics):
    """
    Format a metrics dict with human-readable units for JSON output.

    Returns
    -------
    dict with formatted string values (e.g. "45.33 seconds", "44.91%")
    """
    return {
        "R2": f"{metrics['R2']:.4f}",
        "RMSE": f"{metrics['RMSE']:.2f} seconds",
        "MAE": f"{metrics['MAE']:.2f} seconds",
        "MAPE": f"{metrics['MAPE']:.2f}%",
        "MSE": f"{metrics['MSE']:.4f} seconds²",
    }


def build_results_json(config, timestamp, train_metrics, val_metrics, test_metrics):
    """
    Build the final results dict matching the desired JSON output format.

    Parameters
    ----------
    config : dict — model/training configuration
    timestamp : str — run timestamp (e.g. "2026-03-08 14:30:00")
    train_metrics : dict — raw metrics for train split
    val_metrics : dict — raw metrics for val split
    test_metrics : dict — raw metrics for test split

    Returns
    -------
    dict ready for json.dump()
    """
    return {
        "method": "T.R2_LSTM_A",
        "config": config,
        "timestamp": timestamp,
        "Train": format_metrics_for_json(train_metrics),
        "Val": format_metrics_for_json(val_metrics),
        "Test": format_metrics_for_json(test_metrics),
    }


def save_results_json(results_dict, filepath):
    """Save results dict to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"  Results saved to: {filepath}")


def format_results_table(metrics):
    """
    Format T.R2_LSTM_A results as a printable table.
    """
    lines = []
    lines.append("=" * 50)
    lines.append("T.R2_LSTM_A RESULTS")
    lines.append("=" * 50)
    lines.append(f"  R2:   {metrics['R2']:.4f}")
    lines.append(f"  MSE:  {metrics['MSE']:.4f}")
    lines.append(f"  RMSE: {metrics['RMSE']:.4f}")
    lines.append(f"  MAPE: {metrics['MAPE']:.4f}")
    lines.append(f"  MAE:  {metrics['MAE']:.4f}")
    lines.append("=" * 50)
    return "\n".join(lines)


if __name__ == "__main__":
    np.random.seed(42)
    y_true = np.random.randn(100) * 50 + 100
    y_pred = y_true + np.random.randn(100) * 5

    m = compute_all_metrics(y_true, y_pred)
    print(format_results_table(m))
    print("\nFormatted for JSON:")
    print(json.dumps(format_metrics_for_json(m), indent=2))
    print("\nEvaluation module tested successfully!")
