"""
Complete feature diagnostic: Check raw data AND segments.
"""

import pandas as pd
import numpy as np
from data_loader import load_train_test_val_data_fixed, get_known_stops
from cluster_dbscan import event_driven_clustering_fixed
from segments import build_segments_fixed

print("\n" + "=" * 80)
print("STEP 1: CHECKING RAW CSV FILES")
print("=" * 80)

# Load raw CSV to see what columns exist
train_raw = pd.read_csv('./data/train_data.csv', nrows=1000)  # Sample for speed

print(f"\n✅ COLUMNS IN train_data.csv ({len(train_raw.columns)} total):")
print("-" * 80)
for i, col in enumerate(train_raw.columns, 1):
    print(f"  {i:2d}. {col}")

print("\n🔍 CHECKING FOR OPERATIONAL FEATURES IN RAW DATA:")
operational_cols = ['arrivalDelay', 'departureDelay']
for col in operational_cols:
    if col in train_raw.columns:
        vals = train_raw[col].dropna()
        if len(vals) > 0:
            print(f"  ✅ {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
                  f"unique={len(vals.unique())}, nulls={train_raw[col].isna().sum()}")
        else:
            print(f"  ⚠️  {col}: EXISTS but all NaN")
    else:
        print(f"  ❌ {col}: NOT IN RAW DATA")

print("\n🔍 CHECKING FOR WEATHER FEATURES IN RAW DATA:")
weather_cols = ['temperature_2m', 'apparent_temperature', 'precipitation',
                'rain', 'snowfall', 'windspeed_10m', 'windgusts_10m',
                'winddirection_10m']
for col in weather_cols:
    if col in train_raw.columns:
        vals = train_raw[col].dropna()
        if len(vals) > 0:
            print(f"  ✅ {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
                  f"unique={len(vals.unique())}, nulls={train_raw[col].isna().sum()}")
        else:
            print(f"  ⚠️  {col}: EXISTS but all NaN")
    else:
        print(f"  ❌ {col}: NOT IN RAW DATA")

print("\n" + "=" * 80)
print("STEP 2: CHECKING SEGMENTS (AFTER PROCESSING)")
print("=" * 80)

# Now check segments
train_df, _, _ = load_train_test_val_data_fixed(data_folder='./data', sample_fraction=0.01)
known_stops = get_known_stops(train_df)
clusters, _ = event_driven_clustering_fixed(train_df, known_stops=known_stops)
train_segments = build_segments_fixed(train_df, clusters)

print(f"\n🔍 COLUMNS IN SEGMENTS ({len(train_segments.columns)} total):")
print("-" * 80)
for i, col in enumerate(train_segments.columns, 1):
    print(f"  {i:2d}. {col}")

print("\n🔍 OPERATIONAL FEATURES IN SEGMENTS:")
for col in operational_cols:
    if col in train_segments.columns:
        vals = train_segments[col].dropna()
        print(f"  ✅ {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, unique={len(vals.unique())}")
    else:
        print(f"  ❌ {col}: MISSING FROM SEGMENTS (lost during segment building)")

print("\n🔍 WEATHER FEATURES IN SEGMENTS:")
for col in weather_cols:
    if col in train_segments.columns:
        vals = train_segments[col].dropna()
        print(f"  ✅ {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, unique={len(vals.unique())}")
    else:
        print(f"  ❌ {col}: MISSING FROM SEGMENTS (lost during segment building)")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

# Find which features exist in raw but not in segments
raw_cols = set(train_raw.columns)
seg_cols = set(train_segments.columns)

missing_in_segments = raw_cols - seg_cols
extra_in_segments = seg_cols - raw_cols

if missing_in_segments:
    print(f"\n⚠️  FEATURES IN RAW DATA BUT LOST IN SEGMENTS ({len(missing_in_segments)}):")
    for col in sorted(missing_in_segments):
        if col in operational_cols + weather_cols:
            print(f"  ❌ {col} ← THIS NEEDS TO BE CARRIED OVER!")
        else:
            print(f"     {col}")

print("\n" + "=" * 80)
print("SOLUTION: Update segments.py to preserve these features")
print("=" * 80)