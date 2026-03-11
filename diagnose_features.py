"""
Lightweight feature diagnosis - NO clustering needed
Checks variance BEFORE the expensive clustering step
"""

import numpy as np
import pandas as pd

print("=" * 100)
print("LIGHTWEIGHT FEATURE VARIANCE CHECK")
print("=" * 100)

# ✅ Load ONLY a small sample
print("\n📂 Loading SAMPLE of data (10%)...")
train_df = pd.read_csv('./data/train_data.csv').sample(frac=0.1, random_state=42)
print(f"   Loaded {len(train_df):,} rows (10% sample)")

# Check what columns exist
print(f"\n📋 Available columns:")
print(train_df.columns.tolist())

# ============================================================================
# CHECK OPERATIONAL FEATURES
# ============================================================================
print("\n" + "=" * 100)
print("OPERATIONAL FEATURES DIAGNOSIS")
print("=" * 100)

operational_cols = ['arrivalDelay', 'departureDelay']

for col in operational_cols:
    if col in train_df.columns:
        values = train_df[col].dropna().values

        print(f"\n{col}:")
        print(f"  Count: {len(values):,}")
        print(f"  Min: {values.min():.4f}")
        print(f"  Max: {values.max():.4f}")
        print(f"  Mean: {values.mean():.4f}")
        print(f"  Median: {np.median(values):.4f}")
        print(f"  Std: {values.std():.4f}")
        print(f"  25th percentile: {np.percentile(values, 25):.4f}")
        print(f"  75th percentile: {np.percentile(values, 75):.4f}")
        print(f"  IQR: {np.percentile(values, 75) - np.percentile(values, 25):.4f}")
        print(f"  Unique values: {len(np.unique(values)):,}")

        # Distribution
        print(f"  Distribution:")
        print(f"    Zero: {(values == 0).sum():,} ({(values == 0).mean() * 100:.1f}%)")
        print(f"    Negative: {(values < 0).sum():,} ({(values < 0).mean() * 100:.1f}%)")
        print(f"    Positive: {(values > 0).sum():,} ({(values > 0).mean() * 100:.1f}%)")

        # ❌ Problem checks
        if values.std() < 1.0:
            print(f"  ❌ CRITICAL: Very low variance! Std = {values.std():.4f}")
        if (values == 0).mean() > 0.95:
            print(f"  ❌ CRITICAL: 95%+ of values are ZERO!")
        if len(np.unique(values)) < 10:
            print(f"  ⚠️  WARNING: Only {len(np.unique(values))} unique values")
    else:
        print(f"\n{col}: ❌ NOT FOUND IN DATA")

# ============================================================================
# CHECK TEMPORAL FEATURES (for binary flags)
# ============================================================================
print("\n" + "=" * 100)
print("TEMPORAL FEATURES (for is_weekend, is_peak_hour)")
print("=" * 100)

if 'timestamp' in train_df.columns:
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    train_df['hour'] = train_df['timestamp'].dt.hour
    train_df['day_of_week'] = train_df['timestamp'].dt.dayofweek

    # Create flags manually
    train_df['is_weekend'] = (train_df['day_of_week'] >= 5).astype(int)
    train_df['is_peak_hour'] = (
                                       ((train_df['hour'] >= 7) & (train_df['hour'] < 9)) |
                                       ((train_df['hour'] >= 16) & (train_df['hour'] < 19))
                               ).astype(int) * (train_df['day_of_week'] < 5).astype(int)

    print("\nis_weekend:")
    print(f"  Distribution: {train_df['is_weekend'].value_counts().to_dict()}")
    print(f"  Percentage weekend: {train_df['is_weekend'].mean() * 100:.1f}%")

    print("\nis_peak_hour:")
    print(f"  Distribution: {train_df['is_peak_hour'].value_counts().to_dict()}")
    print(f"  Percentage peak: {train_df['is_peak_hour'].mean() * 100:.1f}%")

    print("\nHour distribution:")
    print(train_df['hour'].value_counts().sort_index())

    print("\nDay of week distribution:")
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for dow in range(7):
        count = (train_df['day_of_week'] == dow).sum()
        print(f"  {dow_names[dow]}: {count:,} ({count / len(train_df) * 100:.1f}%)")

# ============================================================================
# CHECK WEATHER FEATURES
# ============================================================================
print("\n" + "=" * 100)
print("WEATHER FEATURES DIAGNOSIS")
print("=" * 100)

weather_cols = ['temperature_2m', 'apparent_temperature', 'precipitation',
                'rain', 'snowfall', 'windspeed_10m', 'windgusts_10m',
                'winddirection_10m']

weather_found = False
for col in weather_cols:
    if col in train_df.columns:
        weather_found = True
        values = train_df[col].dropna().values

        print(f"\n{col}:")
        print(f"  Count: {len(values):,}")
        print(f"  Min: {values.min():.4f}")
        print(f"  Max: {values.max():.4f}")
        print(f"  Mean: {values.mean():.4f}")
        print(f"  Std: {values.std():.4f}")
        print(f"  Unique values: {len(np.unique(values)):,}")

        # ❌ Problem checks
        if values.std() < 0.1:
            print(f"  ❌ CRITICAL: Very low variance! Std = {values.std():.4f}")
        if len(np.unique(values)) == 1:
            print(f"  ❌ CRITICAL: Constant value = {np.unique(values)[0]}")
        if len(np.unique(values)) < 5:
            print(f"  ⚠️  WARNING: Only {len(np.unique(values))} unique values")
            print(f"      Values: {np.unique(values)}")

if not weather_found:
    print("\n❌ NO WEATHER FEATURES FOUND IN DATA")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 100)
print("DIAGNOSIS SUMMARY")
print("=" * 100)

problems = []

# Check operational
for col in operational_cols:
    if col in train_df.columns:
        values = train_df[col].dropna().values
        if values.std() < 1.0:
            problems.append(f"❌ {col}: Low variance (std={values.std():.4f})")
        if (values == 0).mean() > 0.95:
            problems.append(f"❌ {col}: 95%+ are zeros")
    else:
        problems.append(f"❌ {col}: Missing from data")

# Check weather
weather_constant = 0
for col in weather_cols:
    if col in train_df.columns:
        values = train_df[col].dropna().values
        if values.std() < 0.1:
            problems.append(f"⚠️  {col}: Low variance (std={values.std():.4f})")
        if len(np.unique(values)) == 1:
            weather_constant += 1
            problems.append(f"❌ {col}: Constant ({np.unique(values)[0]})")

if weather_constant >= 5:
    problems.append(f"❌ CRITICAL: {weather_constant}/8 weather features are constant!")

if problems:
    print("\n🔴 PROBLEMS DETECTED:")
    for p in problems:
        print(f"   {p}")

    print("\n💡 LIKELY CAUSE OF IDENTICAL RMSE:")
    if any("Low variance" in p for p in problems):
        print("   → Features have no variance, so LSTM can't learn from them")
    if any("zeros" in p for p in problems):
        print("   → Delays are all zero, no operational signal to learn")
    if weather_constant >= 5:
        print("   → Weather is constant, no weather signal to learn")
else:
    print("\n✅ No obvious problems detected")
    print("   The issue may be in the model architecture or training process")

print("\n" + "=" * 100)