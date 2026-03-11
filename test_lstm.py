"""
Test if LSTM is producing corrections
Run this BEFORE training to verify the model works
"""

import torch
import numpy as np
from model import MAGTTE, EnhancedSegmentDataset, enhanced_collate_fn
from residual import MAGNN_LSTM_Residual
from segments import build_segments_fixed
from cluster_hdbscan import event_driven_clustering_fixed
from data_loader import load_train_test_val_data_fixed, get_known_stops
from torch.utils.data import DataLoader

print("=" * 100)
print("LSTM OUTPUT DIAGNOSTIC")
print("=" * 100)

# Load small sample
print("\n📂 Loading data (1% sample)...")
train_df, _, _ = load_train_test_val_data_fixed('./data', sample_fraction=0.01)
print(f"   Loaded {len(train_df):,} rows")

print("\n🔧 Clustering...")
known_stops = get_known_stops(train_df)
clusters, _ = event_driven_clustering_fixed(train_df, known_stops=known_stops)
print(f"   Created {len(clusters)} clusters")

print("\n🔧 Building segments...")
segments = build_segments_fixed(train_df, clusters)
print(f"   Created {len(segments):,} segments")

segment_types = segments['segment_id'].unique()
print(f"   Unique segment types: {len(segment_types)}")

print("\n🔧 Creating dataset...")
dataset = EnhancedSegmentDataset(segments, segment_types, fit_scalers=True)
loader = DataLoader(dataset, batch_size=32, collate_fn=enhanced_collate_fn, shuffle=False)
print(f"   Dataset size: {len(dataset):,}")

# Create models
print("\n🔧 Creating models...")
num_segments = len(segment_types)
magnn = MAGTTE(num_segments, 3, 32, 32, 64, 16, 0.3)

adj_geo = np.eye(num_segments)
adj_dist = np.eye(num_segments)
adj_soc = np.eye(num_segments)
magnn.set_adjacency_matrices(adj_geo, adj_dist, adj_soc)

residual_model = MAGNN_LSTM_Residual(magnn, 32, 4, 8, 5, 128, 1, 0.2, freeze_magnn=True)

# Test forward pass
print("\n" + "=" * 100)
print("FORWARD PASS TEST")
print("=" * 100)

batch = next(iter(loader))
seg_idx, temporal, operational, weather, target, lengths, mask = batch

print(f"\n✅ Batch shapes:")
print(f"   seg_indices: {seg_idx.shape}")
print(f"   temporal: {temporal.shape}")
print(f"   operational: {operational.shape}")
print(f"   weather: {weather.shape}")
print(f"   target: {target.shape}")

print(f"\n🔍 Operational features (first 5 samples):")
print(operational.squeeze(1)[:5])
print(f"\n   Stats per column:")
print(f"   [0] arrivalDelay:    mean={operational[:, :, 0].mean():.4f}, std={operational[:, :, 0].std():.4f}")
print(f"   [1] departureDelay:  mean={operational[:, :, 1].mean():.4f}, std={operational[:, :, 1].std():.4f}")
print(
    f"   [2] is_weekend:      unique={operational[:, :, 2].unique().tolist()}, mean={operational[:, :, 2].mean():.4f}")
print(
    f"   [3] is_peak_hour:    unique={operational[:, :, 3].unique().tolist()}, mean={operational[:, :, 3].mean():.4f}")

print(f"\n🔍 Weather features stats:")
for i in range(weather.shape[2]):
    print(f"   [{i}] mean={weather[:, :, i].mean():.4f}, std={weather[:, :, i].std():.4f}")

# Run model
print("\n" + "=" * 100)
print("MODEL OUTPUTS")
print("=" * 100)

baseline, correction, alpha, final = residual_model(
    seg_idx,
    temporal.squeeze(1),
    operational.squeeze(1),
    weather.squeeze(1),
    return_components=True
)

print(f"\n✅ Output shapes:")
print(f"   Baseline: {baseline.shape}")
print(f"   Correction: {correction.shape}")
print(f"   Alpha: {alpha.shape}")
print(f"   Final: {final.shape}")

print(f"\n📊 Output statistics:")
print(f"   Baseline (MAGNN):")
print(f"     mean={baseline.mean():.4f}, std={baseline.std():.4f}")
print(f"     min={baseline.min():.4f}, max={baseline.max():.4f}")

print(f"\n   Correction (LSTM):")
print(f"     mean={correction.mean():.4f}, std={correction.std():.4f}")
print(f"     min={correction.min():.4f}, max={correction.max():.4f}")
print(f"     mean(|correction|)={correction.abs().mean():.4f}")

print(f"\n   Alpha (gate):")
print(f"     mean={alpha.mean():.4f}, std={alpha.std():.4f}")
print(f"     min={alpha.min():.4f}, max={alpha.max():.4f}")

print(f"\n   Final prediction:")
print(f"     mean={final.mean():.4f}, std={final.std():.4f}")

# ✅ Diagnostics
print("\n" + "=" * 100)
print("DIAGNOSTIC CHECKS")
print("=" * 100)

issues = []

if correction.abs().mean() < 0.01:
    issues.append("❌ CRITICAL: LSTM corrections near zero!")
    print(f"\n❌ CRITICAL: LSTM corrections are near zero!")
    print(f"   mean(|correction|) = {correction.abs().mean():.6f}")
    print(f"   This means LSTM isn't learning - all models will have identical RMSE")
else:
    print(f"\n✅ LSTM is producing corrections")
    print(f"   mean(|correction|) = {correction.abs().mean():.4f}")

if alpha.std() < 0.01:
    issues.append("❌ CRITICAL: Adaptive gate stuck!")
    print(f"\n❌ CRITICAL: Adaptive gate is stuck")
    print(f"   std(alpha) = {alpha.std():.6f}")
else:
    print(f"\n✅ Adaptive gate has variance")
    print(f"   std(alpha) = {alpha.std():.4f}")

if operational.std() < 0.01:
    issues.append("⚠️  WARNING: Operational features have no variance")
    print(f"\n⚠️  WARNING: Operational features have very low variance")
    print(f"   std = {operational.std():.6f}")
else:
    print(f"\n✅ Operational features have variance")
    print(f"   std = {operational.std():.4f}")

if weather.std() < 0.01:
    issues.append("⚠️  WARNING: Weather features have no variance")
    print(f"\n⚠️  WARNING: Weather features have very low variance")
    print(f"   std = {weather.std():.6f}")
else:
    print(f"\n✅ Weather features have variance")
    print(f"   std = {weather.std():.4f}")

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

if not issues:
    print("\n✅ ALL CHECKS PASSED!")
    print("   Model appears to be working correctly")
    print("   LSTM is producing corrections")
    print("   Features have variance")
    print("\n   You can now run: python main.py --residual")
else:
    print(f"\n🔴 {len(issues)} ISSUE(S) DETECTED:")
    for issue in issues:
        print(f"   {issue}")

    print("\n💡 RECOMMENDATIONS:")
    if "LSTM corrections near zero" in str(issues):
        print("   1. Check LSTM initialization (should use Xavier)")
        print("   2. Increase learning rate (try 0.001 * 0.5)")
        print("   3. Check if features have variance")
    if "gate stuck" in str(issues):
        print("   1. Gate may need more training epochs")
        print("   2. Check spatial embeddings have variance")

print("\n" + "=" * 100)