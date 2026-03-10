"""
data_loader.py — Data loading, feature selection, and sequence construction.

Corresponds to Paper Section 4.1 (Data Description) and Figure 9 (Data Reorganization).

Key design decisions:
- train_data.csv  → SOURCE DOMAIN (large, data-rich)
- test_data.csv   → TARGET DOMAIN TEST SET
- validation_data.csv → Validation during training
- holdout_data.csv    → Final holdout evaluation

SEQUENCE CONSTRUCTION:
  Each trip has multiple stops (trip_stop_sequence). We aggregate GPS
  observations per (trip_id, trip_stop_sequence) into one feature vector
  per stop, then group stops into a trip-level sequence. This gives
  the LSTM a proper temporal dimension — each timestep is one stop in
  the journey, and the model attends across stops.

  Target variable: travel_time_sec of the LAST stop in the sequence
  (predicting the next segment's travel time given the journey so far).

NORMALIZATION:
  Paper Equation 10 specifies Min-Max normalization to [0, 1].
  We use sklearn MinMaxScaler, fit on the source domain.
  Both X (features) and y (target) are normalized to [0, 1].
  Predictions are inverse-transformed back to seconds for metrics.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import warnings

warnings.filterwarnings("ignore")

# ─── Column configuration ───────────────────────────────────────────────────

# Columns to drop: non-useful identifiers and string columns
COLUMNS_TO_DROP = [
    "currentLoc",
    "vehicleLabel",
    "vehicleLicenceplate",
    "timestamp",
    "arrivalTime",
    "departureTime",
    "service_date",
    "originStopName",
    "destinationStopName",
    "segment",
    "slowdown_lat",
    "slowdown_lon",
]

# Additional ID columns to drop (not features, but needed for grouping first)
ID_COLS_TO_DROP_AFTER_GROUPING = [
    "vehicleID",
    "odometer",
    "originStopID",
    "destinationStopID",
    "currentStatus",
    "tripScheduleRelationship",
]

# Grouping columns — kept for sequence construction, dropped afterwards
GROUP_COL = "trip_id"
SEQUENCE_COL = "trip_stop_sequence"

TARGET_COLUMN = "travel_time_sec"

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def compute_max_sequence_length(filepath=None, percentile=95):
    """
    Compute MAX_SEQUENCE_LENGTH from the actual distribution of stops per trip.

    Uses the given percentile of the distribution (default 95th) rounded to
    the nearest integer. This avoids hardcoding a value that may not match
    the data.

    Parameters
    ----------
    filepath : str or None
        Path to the training CSV. If None, uses train_data.csv in DATA_DIR.
    percentile : int
        Percentile of the stops-per-trip distribution to use (default 95).

    Returns
    -------
    int — the computed max sequence length
    """
    if filepath is None:
        filepath = os.path.join(DATA_DIR, "train_data.csv")

    df = pd.read_csv(filepath, usecols=[GROUP_COL, SEQUENCE_COL], low_memory=False)
    stops_per_trip = df.groupby(GROUP_COL)[SEQUENCE_COL].nunique()

    seq_len = int(np.round(stops_per_trip.quantile(percentile / 100.0)))
    seq_len = max(seq_len, 2)  # at least 2 timesteps for LSTM

    print(f"  Stops per trip: min={stops_per_trip.min()}, "
          f"median={stops_per_trip.median():.0f}, "
          f"max={stops_per_trip.max()}, "
          f"p{percentile}={seq_len}")
    print(f"  MAX_SEQUENCE_LENGTH set to {seq_len} (p{percentile} of training data)")

    return seq_len


def load_and_build_sequences(filepath, max_seq_len, nrows=None):
    """
    Load a CSV, aggregate observations per (trip_id, stop), and build
    fixed-length sequences of shape (n_trips, max_seq_len, n_features).

    Each row in the raw CSV is a GPS observation. Multiple observations
    exist per (trip_id, trip_stop_sequence). We aggregate them into one
    feature vector per stop using mean for continuous and max for binary.

    The target is the travel_time_sec of the LAST stop in the sequence.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    max_seq_len : int
        Maximum number of stops per trip (pad/truncate to this).
    nrows : int or None
        Number of raw rows to load. None = all rows.

    Returns
    -------
    X : np.ndarray of shape (n_trips, max_seq_len, n_features)
    y : np.ndarray of shape (n_trips,)
    actual_lengths : np.ndarray of shape (n_trips,) — real stop count per trip
    """
    df = pd.read_csv(filepath, nrows=nrows, low_memory=False)

    # Drop string/non-feature columns (but keep trip_id & trip_stop_sequence)
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Check target exists
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in {filepath}")

    # Remove rows where target is NaN or zero
    df = df[df[TARGET_COLUMN].notna() & (df[TARGET_COLUMN] != 0)].copy()

    # Identify binary columns (for max aggregation) vs continuous (for mean)
    binary_cols = [c for c in df.columns if c.startswith("is_") or c.startswith("has_")]

    # Identify feature columns (everything except group, sequence, target, and ID cols)
    all_drop = [GROUP_COL, SEQUENCE_COL, TARGET_COLUMN] + ID_COLS_TO_DROP_AFTER_GROUPING
    feature_cols = [c for c in df.columns if c not in all_drop and df[c].dtype in ['float64', 'int64', 'float32']]

    # Build aggregation rules per (trip_id, trip_stop_sequence)
    agg_dict = {}
    for col in feature_cols:
        if col in binary_cols:
            agg_dict[col] = "max"
        else:
            agg_dict[col] = "mean"
    agg_dict[TARGET_COLUMN] = "last"  # travel time for this stop

    # Aggregate: one row per (trip_id, stop)
    grouped = df.groupby([GROUP_COL, SEQUENCE_COL]).agg(agg_dict).reset_index()

    # Sort by trip_id and stop sequence
    grouped = grouped.sort_values([GROUP_COL, SEQUENCE_COL])

    # Fill NaNs with 0 (after aggregation, some columns may have NaN)
    grouped[feature_cols] = grouped[feature_cols].fillna(0)

    # Build sequences: pad/truncate to max_seq_len
    trip_ids = grouped[GROUP_COL].unique()
    n_features = len(feature_cols)
    n_trips = len(trip_ids)

    X = np.zeros((n_trips, max_seq_len, n_features), dtype=np.float32)
    y = np.zeros(n_trips, dtype=np.float32)
    actual_lengths = np.zeros(n_trips, dtype=np.int32)

    # Pre-group into dict for O(1) lookup per trip instead of O(n) filter
    trip_groups = {tid: grp for tid, grp in grouped.groupby(GROUP_COL)}

    for i, tid in enumerate(trip_ids):
        trip_data = trip_groups[tid]
        features = trip_data[feature_cols].values
        target_vals = trip_data[TARGET_COLUMN].values

        seq_len = min(len(features), max_seq_len)
        X[i, :seq_len, :] = features[:seq_len]
        y[i] = target_vals[seq_len - 1]  # target = last stop's travel time
        actual_lengths[i] = seq_len

    return X, y, actual_lengths, feature_cols


def load_all_datasets(source_nrows=None, target_nrows=None,
                      val_nrows=None, holdout_nrows=None):
    """
    Load all four dataset splits and build sequences.

    MAX_SEQUENCE_LENGTH is computed dynamically from the 95th percentile
    of the stops-per-trip distribution in the source (training) data.

    Returns
    -------
    dict with keys: 'source', 'target', 'validation', 'holdout'
          each containing (X_3d, y, actual_lengths) tuple
    Also: 'feature_cols' — list of feature column names
          'max_seq_len' — the computed max sequence length
    """
    # Compute MAX_SEQUENCE_LENGTH from data
    print("Computing MAX_SEQUENCE_LENGTH from training data...")
    max_seq_len = compute_max_sequence_length()

    print("Loading source domain (train_data.csv)...")
    X_source, y_source, lens_source, feature_cols = load_and_build_sequences(
        os.path.join(DATA_DIR, "train_data.csv"), max_seq_len, nrows=source_nrows
    )
    print(f"  Source: {X_source.shape[0]:,} trips, seq_len={X_source.shape[1]}, features={X_source.shape[2]}")

    print("Loading target domain test (test_data.csv)...")
    X_target, y_target, lens_target, _ = load_and_build_sequences(
        os.path.join(DATA_DIR, "test_data.csv"), max_seq_len, nrows=target_nrows
    )
    print(f"  Target: {X_target.shape[0]:,} trips, seq_len={X_target.shape[1]}, features={X_target.shape[2]}")

    print("Loading validation (validation_data.csv)...")
    X_val, y_val, lens_val, _ = load_and_build_sequences(
        os.path.join(DATA_DIR, "validation_data.csv"), max_seq_len, nrows=val_nrows
    )
    print(f"  Validation: {X_val.shape[0]:,} trips, seq_len={X_val.shape[1]}, features={X_val.shape[2]}")

    print("Loading holdout (holdout_data.csv)...")
    X_holdout, y_holdout, lens_holdout, _ = load_and_build_sequences(
        os.path.join(DATA_DIR, "holdout_data.csv"), max_seq_len, nrows=holdout_nrows
    )
    print(f"  Holdout: {X_holdout.shape[0]:,} trips, seq_len={X_holdout.shape[1]}, features={X_holdout.shape[2]}")

    return {
        "source": (X_source, y_source, lens_source),
        "target": (X_target, y_target, lens_target),
        "validation": (X_val, y_val, lens_val),
        "holdout": (X_holdout, y_holdout, lens_holdout),
        "feature_cols": feature_cols,
        "max_seq_len": max_seq_len,
    }


def scale_data(X_source, y_source, X_target, y_target,
               X_val=None, y_val=None, X_holdout=None, y_holdout=None):
    """
    Min-Max normalization to [0, 1] — Paper Equation 10.

    Fit on source domain (data-rich), transform all datasets.
    Operates on 3D X arrays: (trips, timesteps, features).
    Scaling is done per-feature across all timesteps.

    Also scales y (target variable) to [0, 1] so the model's output
    range matches the input range, improving convergence.
    Predictions must be inverse-transformed back to seconds for metrics.

    Returns
    -------
    dict with scaled arrays + the fitted scalers:
        'source_X', 'target_X', 'validation_X', 'holdout_X'  — scaled X
        'source_y', 'target_y', 'validation_y', 'holdout_y'  — scaled y
        'x_scaler' — fitted MinMaxScaler for X (for inverse transform)
        'y_scaler' — fitted MinMaxScaler for y (for inverse transform)
    """
    n_trips_src, seq_len, n_features = X_source.shape

    # ─── Scale X (features) ──────────────────────────────────────────────
    # Flatten to 2D for fitting: (trips * timesteps, features)
    X_src_flat = X_source.reshape(-1, n_features)

    x_scaler = MinMaxScaler(feature_range=(0, 1))
    X_src_scaled = x_scaler.fit_transform(X_src_flat).reshape(n_trips_src, seq_len, n_features)

    result = {
        "source_X": X_src_scaled.astype(np.float32),
        "x_scaler": x_scaler,
    }

    X_tgt_flat = X_target.reshape(-1, n_features)
    result["target_X"] = x_scaler.transform(X_tgt_flat).reshape(X_target.shape).astype(np.float32)

    if X_val is not None:
        X_val_flat = X_val.reshape(-1, n_features)
        result["validation_X"] = x_scaler.transform(X_val_flat).reshape(X_val.shape).astype(np.float32)

    if X_holdout is not None:
        X_hold_flat = X_holdout.reshape(-1, n_features)
        result["holdout_X"] = x_scaler.transform(X_hold_flat).reshape(X_holdout.shape).astype(np.float32)

    # ─── Scale y (target) ────────────────────────────────────────────────
    # Fit on source y, transform all y arrays
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    result["source_y"] = y_scaler.fit_transform(y_source.reshape(-1, 1)).flatten().astype(np.float32)
    result["target_y"] = y_scaler.transform(y_target.reshape(-1, 1)).flatten().astype(np.float32)
    result["y_scaler"] = y_scaler

    if y_val is not None:
        result["validation_y"] = y_scaler.transform(y_val.reshape(-1, 1)).flatten().astype(np.float32)

    if y_holdout is not None:
        result["holdout_y"] = y_scaler.transform(y_holdout.reshape(-1, 1)).flatten().astype(np.float32)

    return result


if __name__ == "__main__":
    print("Testing data loader with sequence construction...")
    data = load_all_datasets(source_nrows=5000, target_nrows=2000,
                             val_nrows=1000, holdout_nrows=1000)

    X_src, y_src, lens_src = data["source"]
    print(f"\nSource X shape: {X_src.shape}")
    print(f"Source y shape: {y_src.shape}")
    print(f"Sequence lengths: min={lens_src.min()}, max={lens_src.max()}, mean={lens_src.mean():.1f}")
    print(f"Target (travel_time_sec): min={y_src.min():.1f}, max={y_src.max():.1f}, mean={y_src.mean():.1f}")

    # Test scaling
    X_tgt, y_tgt, lens_tgt = data["target"]
    X_val, y_val, lens_val = data["validation"]
    X_hold, y_hold, lens_hold = data["holdout"]

    scaled = scale_data(X_src, y_src, X_tgt, y_tgt, X_val, y_val, X_hold, y_hold)
    print(f"\nScaled source X range: [{scaled['source_X'].min():.4f}, {scaled['source_X'].max():.4f}]")
    print(f"Scaled source y range: [{scaled['source_y'].min():.4f}, {scaled['source_y'].max():.4f}]")
    print(f"Scaled target X range: [{scaled['target_X'].min():.4f}, {scaled['target_X'].max():.4f}]")
    print(f"Scaled target y range: [{scaled['target_y'].min():.4f}, {scaled['target_y'].max():.4f}]")
    print("\nData loader test complete!")
