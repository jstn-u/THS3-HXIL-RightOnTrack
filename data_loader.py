"""
data_loader.py
==============
GPS data loading, interpolation, and train/val/test splitting.

Public API:
    load_data_fixed(filepath, sample_fraction=1.0)
    load_train_test_val_data_fixed(data_folder='./data', sample_fraction=1.0)
    get_known_stops(df)
"""

import numpy as np
import pandas as pd
import os
import warnings
from sklearn.neighbors import BallTree

from config import Config, DEVICE, print_section, haversine_meters

warnings.filterwarnings('ignore')

# Module-level cache (was a global in the original monolithic files)
_known_stops_cache = {}

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def _interpolate_gps_small_gaps(df: pd.DataFrame,
                                 max_gap_seconds: float = 5.0) -> pd.DataFrame:
    """
    LOCAL LINEAR INTERPOLATION for missing GPS coordinates.

    For each trip, if a GPS record is NaN but the gap to the nearest valid
    neighbour is ≤ max_gap_seconds (default 5 s), fill via linear interpolation
    on the time axis.  Larger gaps are left as NaN so HDBSCAN can ignore them.

    This is applied PER TRIP so that interpolation never crosses trip
    boundaries (which would be nonsensical).

    Args:
        df              : DataFrame with 'trip_id', 'timestamp', 'latitude',
                          'longitude' columns.
        max_gap_seconds : Maximum allowed gap (seconds) to interpolate.
                          Gaps larger than this are left unfilled.

    Returns:
        DataFrame with small GPS gaps filled; 'is_gps_valid' column updated.
    """
    if 'timestamp' not in df.columns or 'latitude' not in df.columns:
        return df

    df = df.copy()

    # Ensure timestamps are datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    filled_count = 0

    for trip_id, grp in df.groupby('trip_id', sort=False):
        grp = grp.sort_values('timestamp').copy()
        idxs = grp.index

        # Identify rows with missing lat or lon
        missing_lat = grp['latitude'].isna() | (grp['latitude'] == 0)
        missing_lon = grp['longitude'].isna() | (grp['longitude'] == 0)
        missing_gps = missing_lat | missing_lon

        if not missing_gps.any():
            continue

        # Time in seconds from trip start (numeric, for interpolation)
        t_sec = (grp['timestamp'] - grp['timestamp'].min()).dt.total_seconds()

        for coord_col in ['latitude', 'longitude']:
            col_vals = grp[coord_col].copy()
            # Zero-valued coords treated as NaN for this pass
            col_vals = col_vals.where(col_vals != 0, np.nan)

            # Compute time-based forward/backward gap per missing point
            col_time_indexed = col_vals.copy()
            col_time_indexed.index = t_sec.values

            # Interpolate by time (linear, limit by gap window)
            # We use a numeric index (seconds) so the interpolation respects
            # actual time distances, not row distances.
            col_numeric = col_vals.copy()
            col_numeric.index = range(len(col_numeric))

            valid_mask = col_vals.notna()
            if valid_mask.sum() < 2:
                continue  # need at least 2 anchor points

            # For each missing position, check if both its preceding and
            # following valid points are within max_gap_seconds
            t_arr = t_sec.values
            v_arr = col_vals.values.copy()

            for pos, (is_miss, t_pos) in enumerate(zip(missing_gps.values, t_arr)):
                if not is_miss:
                    continue
                if not np.isnan(v_arr[pos]) and v_arr[pos] != 0:
                    continue  # already has value

                # Search backward for nearest valid
                prev_v, prev_t = np.nan, np.nan
                for back in range(pos - 1, -1, -1):
                    if not missing_gps.values[back] and not np.isnan(v_arr[back]):
                        prev_v, prev_t = v_arr[back], t_arr[back]
                        break

                # Search forward for nearest valid
                next_v, next_t = np.nan, np.nan
                for fwd in range(pos + 1, len(v_arr)):
                    if not missing_gps.values[fwd] and not np.isnan(v_arr[fwd]):
                        next_v, next_t = v_arr[fwd], t_arr[fwd]
                        break

                if np.isnan(prev_v) or np.isnan(next_v):
                    continue  # can't interpolate without both anchors

                gap = next_t - prev_t
                if gap <= 0 or gap > max_gap_seconds:
                    continue  # gap too large — leave NaN

                # Linear interpolation
                alpha = (t_pos - prev_t) / gap
                interpolated = prev_v + alpha * (next_v - prev_v)
                df.loc[idxs[pos], coord_col] = interpolated
                filled_count += 1

    if filled_count > 0:
        # Recompute GPS validity flag after interpolation
        df['is_gps_valid'] = (
            df['latitude'].notna() & df['longitude'].notna() &
            (df['latitude'] != 0) & (df['longitude'] != 0)
        ).astype(int)
        print(f"   ✓ GPS interpolation: filled {filled_count:,} coordinate values "
              f"(gaps ≤ {max_gap_seconds}s)")

    return df



def load_data_fixed(filepath, sample_fraction=1.0):
    """
    Load and preprocess data with:
      - Timestamp parsing
      - Trip-level sampling (chronological order preserved)
      - GPS validity flag
      - Local linear interpolation for small GPS gaps (≤5 s)
    """
    print_section(f"LOADING DATA: {os.path.basename(filepath)}")

    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df):,} records")

    # ── Parse timestamps ──────────────────────────────────────────────────────
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    if sample_fraction < 1.0 and 'trip_id' in df.columns:
        unique_trips = df['trip_id'].dropna().unique()
        n_sample = max(1, int(len(unique_trips) * sample_fraction))
        sample_trips = np.random.choice(unique_trips, size=n_sample, replace=False)
        df = df[df['trip_id'].isin(sample_trips)].copy()
        print(f"📊 Sampled {sample_fraction * 100:.0f}% of data: {n_sample} trips")

    for c in ['latitude', 'longitude']:
        if c not in df.columns:
            df[c] = np.nan

    df['is_gps_valid'] = (
        df['latitude'].notna() &
        df['longitude'].notna() &
        (df['latitude'] != 0) &
        (df['longitude'] != 0)
    ).astype(int)

    # ── Local linear interpolation for small GPS gaps (≤5 s) ─────────────────
    if 'trip_id' in df.columns and 'timestamp' in df.columns:
        n_invalid_before = int((df['is_gps_valid'] == 0).sum())
        df = _interpolate_gps_small_gaps(df, max_gap_seconds=5.0)
        n_invalid_after = int((df['is_gps_valid'] == 0).sum())
        net_filled = n_invalid_before - n_invalid_after
        if net_filled > 0:
            print(f"   GPS rows recovered via interpolation: {net_filled:,}")

    required_cols = {
        'travel_time_sec': np.nan,
        'distance_m': np.nan,
        'speed_mps': np.nan,
        'is_slow_speed': 0,
        'is_long_dwell': 0,
        'is_slowdown': 0,
        'slowdown_lat': np.nan,
        'slowdown_lon': np.nan,
        'congestionLevel': np.nan,
        'odometer': np.nan,
    }
    for col, default_val in required_cols.items():
        if col not in df.columns:
            df[col] = default_val

    return df



def _chronological_sample(df: pd.DataFrame, sample_fraction: float) -> pd.DataFrame:
    """
    Return the FIRST `sample_fraction` of trips by their earliest timestamp.

    Unlike random sampling this preserves temporal order:
      - Training data = earlier trips
      - Prevents any future data leaking into training splits
    """
    if sample_fraction >= 1.0 or 'trip_id' not in df.columns:
        return df

    # Earliest timestamp per trip
    trip_start = (
        df.dropna(subset=['timestamp'])
        .groupby('trip_id')['timestamp']
        .min()
        .sort_values()
    )
    n_keep = max(1, int(len(trip_start) * sample_fraction))
    keep_trips = trip_start.index[:n_keep]
    return df[df['trip_id'].isin(keep_trips)].copy()



def load_train_test_val_data_fixed(data_folder='./data', sample_fraction=1.0):
    """
    Load all three datasets.

    CHRONOLOGICAL GUARANTEE
    -----------------------
    Each split file is sorted by timestamp so that:
      - Earlier records are always "before" later records within a split.
    When sample_fraction < 1.0 we use _chronological_sample() to keep only
    the EARLIEST trips — this means training data never contains any record
    that would be "in the future" relative to the sampled slice, preventing
    inadvertent temporal data leakage.
    """
    print_section("LOADING DATASETS")

    train_path = os.path.join(data_folder, 'train_data.csv')
    test_path  = os.path.join(data_folder, 'test_data.csv')
    val_path   = os.path.join(data_folder, 'validation_data.csv')

    datasets = {}
    for name, path in [('train', train_path), ('test', test_path), ('val', val_path)]:
        if os.path.exists(path):
            raw = load_data_fixed(path, sample_fraction=1.0)   # load ALL rows first

            # ── Chronological sampling (keeps earliest trips) ──────────────
            if sample_fraction < 1.0:
                raw = _chronological_sample(raw, sample_fraction)
                print(f"   ✓ {name.upper()} chronological sample: "
                      f"{len(raw):,} records ({sample_fraction * 100:.0f}% of trips)")

            # ── Sort each split by timestamp to ensure temporal order ──────
            if 'timestamp' in raw.columns:
                raw = raw.sort_values('timestamp').reset_index(drop=True)
                print(f"   ✓ {name.upper()} sorted chronologically")

            datasets[name] = raw
        else:
            print(f"⚠️  {name.upper()} file not found: {path}")
            datasets[name] = pd.DataFrame()

    return datasets['train'], datasets['test'], datasets['val']


# =============================================================================
# CLUSTERING
# =============================================================================


def get_known_stops(df):
    """
    Extract real named train stations from the dataset.
    Returns a dict:  { physical_station_name: (lat, lon) }

    Each station appears in the data as "Station Name Platform 1" and
    "Station Name Platform 2" — both platforms share the same lat/lon,
    so we strip the "Platform N" suffix and deduplicate so that each
    physical station appears exactly once.

    Reads both origin and destination columns and merges them.
    Also populates _known_stops_cache so the social adjacency builder
    can match clusters to station coordinates without the raw dataframe.
    """
    global _known_stops_cache
    import re

    frames = []
    for name_col, lat_col, lon_col in [
        ('originStopName',      'originLat',      'originLon'),
        ('destinationStopName', 'destinationLat', 'destinationLon'),
    ]:
        if all(c in df.columns for c in [name_col, lat_col, lon_col]):
            sub = df[[name_col, lat_col, lon_col]].dropna()
            sub = sub.rename(columns={name_col: 'name', lat_col: 'lat', lon_col: 'lon'})
            frames.append(sub)

    if not frames:
        return {}

    combined = pd.concat(frames, ignore_index=True)

    # Strip "Platform 1" / "Platform 2" (or any platform number) to get
    # the physical station name
    combined['station'] = (combined['name']
                           .str.replace(r'\s+Platform\s+\d+$', '', regex=True)
                           .str.strip())

    # Average lat/lon per physical station (platforms share the same coords
    # so mean is identical — this is just a safe fallback)
    unique_stations = (combined
                       .groupby('station')
                       .agg(lat=('lat', 'mean'), lon=('lon', 'mean'))
                       .reset_index())

    known_stops = {
        row['station']: (float(row['lat']), float(row['lon']))
        for _, row in unique_stations.iterrows()
        if row['station'] and str(row['station']).lower() not in ('nan', '')
    }

    # Populate global cache so social adjacency builder can use coords
    _known_stops_cache.clear()
    _known_stops_cache.update(known_stops)

    print(f"   Known physical stations extracted: {len(known_stops)}")
    for name, (lat, lon) in sorted(known_stops.items()):
        print(f"     {name:<35} ({lat:.6f}, {lon:.6f})")

    return known_stops


# =============================================================================
# VISUALISATION  (clusters + segments + statistics)
# =============================================================================