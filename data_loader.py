"""
data_loader.py
==============
GPS data loading and train/val/test splitting.

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

def load_data_fixed(filepath, sample_fraction=1.0):
    """
    Load and preprocess data with:
      - Timestamp parsing
      - Trip-level sampling (chronological order preserved)
      - GPS validity flag
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