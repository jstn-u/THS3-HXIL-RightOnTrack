"""
model.py - UPDATED WITH BUG FIXES
===================================
✅ Fixed weather scaling bug (was using raw values)
✅ Updated operational features: 4 features (arrivalDelay, departureDelay, is_weekend, is_peak_hour)
✅ Updated weather features: 8 features (all weather columns)
✅ All datasets updated to handle binary flags

All neural network components including Multi-Task Learning for paths
"""

import numpy as np
import pandas as pd
import os
import json
import warnings
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from config import Config, DEVICE, print_section, haversine_meters
from mtl import (MTLHead, MTLLoss,
                 DualTaskMTLHead, DualTaskMTLLoss,
                 MAGNN_LSTM_DualTaskMTL, TripSequenceEncoder)

warnings.filterwarnings('ignore')


# =============================================================================
# BASE DATASET (NO CHANGES)
# =============================================================================

class SegmentDataset(Dataset):
    def __init__(self, segments_df, segment_types,
                 fit_scalers: bool = True,
                 target_scaler: RobustScaler = None,
                 speed_scaler: RobustScaler = None):
        self.segments_df = segments_df.copy()
        self.segment_types = list(segment_types)
        self.seg_to_idx = {seg: i for i, seg in enumerate(self.segment_types)}

        self.segments_df['seg_type_idx'] = self.segments_df['segment_id'].map(self.seg_to_idx)
        n_before = len(self.segments_df)
        self.segments_df = self.segments_df.dropna(subset=['seg_type_idx']).copy()
        n_dropped = n_before - len(self.segments_df)
        if n_dropped > 0:
            print(f"   ⚠️  Dropped {n_dropped:,} rows with unseen segment types")
        self.segments_df['seg_type_idx'] = self.segments_df['seg_type_idx'].astype(int)

        self.segments_df['hour_sin'] = np.sin(2 * np.pi * self.segments_df['hour'] / 24)
        self.segments_df['hour_cos'] = np.cos(2 * np.pi * self.segments_df['hour'] / 24)
        self.segments_df['dow_sin'] = np.sin(2 * np.pi * self.segments_df['day_of_week'] / 7)
        self.segments_df['dow_cos'] = np.cos(2 * np.pi * self.segments_df['day_of_week'] / 7)

        if fit_scalers:
            self.target_scaler = RobustScaler()
            self.segments_df['duration_scaled'] = self.target_scaler.fit_transform(
                self.segments_df[['duration_sec']]
            )
        else:
            if target_scaler is None:
                raise ValueError("target_scaler must be provided when fit_scalers=False")
            self.target_scaler = target_scaler
            self.segments_df['duration_scaled'] = self.target_scaler.transform(
                self.segments_df[['duration_sec']]
            )

        if 'speed_mps' in self.segments_df.columns:
            speed_vals = self.segments_df[['speed_mps']].copy()
            speed_vals = speed_vals.replace([np.inf, -np.inf], np.nan)
            if fit_scalers:
                self.speed_scaler = RobustScaler()
                speed_scaled = self.speed_scaler.fit_transform(
                    speed_vals.fillna(speed_vals.median())
                )
            else:
                if speed_scaler is None:
                    raise ValueError("speed_scaler must be provided when fit_scalers=False")
                self.speed_scaler = speed_scaler
                speed_scaled = self.speed_scaler.transform(
                    speed_vals.fillna(speed_vals.median())
                )
            speed_scaled = np.nan_to_num(speed_scaled, nan=0.0)
            self.segments_df['speed_scaled'] = speed_scaled.flatten()
        else:
            self.speed_scaler = RobustScaler() if fit_scalers else speed_scaler
            self.segments_df['speed_scaled'] = 0.0

        self.segments_df['seq_len'] = 1

    def __len__(self):
        return len(self.segments_df)

    def __getitem__(self, idx):
        row = self.segments_df.iloc[idx]
        seg_type_idx = int(row['seg_type_idx'])
        seq_len = int(row['seq_len'])

        temporal = torch.FloatTensor([
            float(row['hour_sin']),
            float(row['hour_cos']),
            float(row['dow_sin']),
            float(row['dow_cos']),
            float(row['speed_scaled']),
        ])

        target = torch.FloatTensor([float(row['duration_scaled'])])
        return seg_type_idx, temporal, target, seq_len


# =============================================================================
# ENHANCED DATASET (✅ FIXED WEATHER SCALING + UPDATED FEATURES)
# =============================================================================

class EnhancedSegmentDataset(SegmentDataset):
    """✅ FIXED: Smart weather scaling + don't scale binary flags"""

    def __init__(self, segments_df, segment_types,
                 fit_scalers: bool = True,
                 target_scaler: RobustScaler = None,
                 speed_scaler: RobustScaler = None,
                 operational_scaler: RobustScaler = None,
                 weather_scaler: RobustScaler = None):
        super().__init__(segments_df, segment_types, fit_scalers, target_scaler, speed_scaler)

        # ✅ FIX: Separate continuous and binary operational features
        continuous_operational_cols = ['arrivalDelay', 'departureDelay']
        binary_flag_cols = ['is_weekend', 'is_peak_hour']

        for col in continuous_operational_cols + binary_flag_cols:
            if col not in self.segments_df.columns:
                self.segments_df[col] = 0.0

        # Scale ONLY continuous operational features
        continuous_data = self.segments_df[continuous_operational_cols].copy()
        continuous_data = continuous_data.replace([np.inf, -np.inf], np.nan)
        continuous_data = continuous_data.fillna(continuous_data.median())

        if fit_scalers:
            self.operational_scaler = RobustScaler()
            continuous_scaled = self.operational_scaler.fit_transform(continuous_data)

            print(f"\n✅ Operational features (mixed scaling):")
            print(f"   Continuous (RobustScaler): {continuous_operational_cols}")
            for i, col in enumerate(continuous_operational_cols):
                vals = continuous_scaled[:, i]
                print(f"     {col}: min={vals.min():.2f}, max={vals.max():.2f}, std={vals.std():.2f}")
        else:
            if operational_scaler is None:
                raise ValueError("operational_scaler must be provided")
            self.operational_scaler = operational_scaler
            continuous_scaled = self.operational_scaler.transform(continuous_data)

        # Store scaled continuous features
        for i, col in enumerate(continuous_operational_cols):
            self.segments_df[f'{col}_scaled'] = continuous_scaled[:, i]

        # ✅ FIX: Keep binary flags as-is (0 or 1, don't scale!)
        if fit_scalers:
            print(f"   Binary (no scaling): {binary_flag_cols}")

        for col in binary_flag_cols:
            self.segments_df[f'{col}_scaled'] = self.segments_df[col].values
            if fit_scalers:
                vals = self.segments_df[col].values
                unique = np.unique(vals)
                counts = np.bincount(vals.astype(int))
                print(f"     {col}: values={unique.tolist()}, dist={counts.tolist()}")

        self.operational_cols_scaled = [f'{col}_scaled' for col in continuous_operational_cols + binary_flag_cols]

        # ✅ FIX: Smart weather scaling (detect if already normalized)
        weather_cols = ['temperature_2m', 'apparent_temperature', 'precipitation',
                        'rain', 'snowfall', 'windspeed_10m', 'windgusts_10m',
                        'winddirection_10m']

        for col in weather_cols:
            if col not in self.segments_df.columns:
                self.segments_df[col] = 0.0

        weather_data = self.segments_df[weather_cols].copy()
        weather_data = weather_data.replace([np.inf, -np.inf], np.nan)
        weather_data = weather_data.fillna(0.0)

        if fit_scalers:
            # Check if weather is already normalized (z-scores)
            weather_mean = weather_data.mean().mean()
            weather_std = weather_data.std().mean()

            print(f"\n🔍 Weather feature check:")
            print(f"   Overall mean: {weather_mean:.4f}")
            print(f"   Overall std: {weather_std:.4f}")

            # If mean ≈ 0 and std ≈ 1, likely already normalized
            if abs(weather_mean) < 0.5 and 0.5 < weather_std < 1.5:
                print(f"   ✅ Weather appears already normalized (z-scores)")
                print(f"   Skipping RobustScaler to avoid double-scaling")
                self.weather_scaler = None
                weather_scaled = weather_data.values
            else:
                print(f"   🔧 Applying RobustScaler to weather")
                self.weather_scaler = RobustScaler()
                weather_scaled = self.weather_scaler.fit_transform(weather_data)
        else:
            # ✅ FIX: Set self.weather_scaler BEFORE using it
            self.weather_scaler = weather_scaler

            if self.weather_scaler is not None:
                weather_scaled = self.weather_scaler.transform(weather_data)
            else:
                # Weather was not scaled in training (already normalized)
                weather_scaled = weather_data.values

        for i, col in enumerate(weather_cols):
            self.segments_df[f'{col}_scaled'] = weather_scaled[:, i]

        self.weather_cols_scaled = [f'{col}_scaled' for col in weather_cols]

    # BEFORE (line 235-239)
    # AFTER
    def __getitem__(self, idx):
        from residual import split_features_for_segment  # import at top of file instead

        row = self.segments_df.iloc[idx]
        seg_type_idx = int(row['seg_type_idx'])
        seq_len = int(row['seq_len'])

        temporal = torch.FloatTensor([
            float(row['hour_sin']), float(row['hour_cos']),
            float(row['dow_sin']), float(row['dow_cos']),
            float(row['speed_scaled']),
        ])

        # Split into endpoint-aware tensors using GPS lookup if available,
        # otherwise falls back to the averaged values (both endpoints identical)
        feats = split_features_for_segment(
            row,
            gps_lookup=getattr(self, 'gps_lookup', None),
            mapper=getattr(self, 'station_mapper', None),
        )

        context_flags = torch.FloatTensor(feats['context_flags'])  # (2,)
        origin_operational = torch.FloatTensor(feats['origin_operational'])  # (2,)
        dest_operational = torch.FloatTensor(feats['dest_operational'])  # (2,)
        origin_weather = torch.FloatTensor(feats['origin_weather'])  # (8,)
        dest_weather = torch.FloatTensor(feats['dest_weather'])  # (8,)

        target = torch.FloatTensor([float(row['duration_scaled'])])

        return (seg_type_idx, temporal,
                context_flags, origin_operational, dest_operational,
                origin_weather, dest_weather,
                target, seq_len)

# =============================================================================
# PATH DATASET (✅ UPDATED WITH BINARY FLAGS)
# =============================================================================

class PathDataset(Dataset):
    """Dataset for multi-segment paths (seq_len > 1)."""

    def __init__(self, paths_df, segment_types, max_path_length=10,
                 fit_scalers: bool = True,
                 target_scaler: RobustScaler = None,
                 speed_scaler: RobustScaler = None,
                 operational_scaler: RobustScaler = None,
                 weather_scaler: RobustScaler = None,
                 use_operational: bool = True,
                 use_weather: bool = True):

        self.paths_df = paths_df.copy()
        self.segment_types = list(segment_types)
        self.seg_to_idx = {seg: i for i, seg in enumerate(self.segment_types)}
        self.max_path_length = max_path_length

        self.use_operational = use_operational
        self.use_weather = use_weather

        if fit_scalers:
            self.target_scaler = RobustScaler()
            self.speed_scaler = RobustScaler()
            self.operational_scaler = RobustScaler()
            self.weather_scaler = RobustScaler()

            self.target_scaler.fit(self.paths_df[['total_duration']])

            all_speeds = []
            all_operational = []
            all_weather = []

            for idx, row in self.paths_df.iterrows():
                all_speeds.extend(row['speeds'])
                for i in range(row['seq_len']):
                    # ✅ UPDATED: 4 operational features
                    all_operational.append([
                        row['arrival_delays'][i],
                        row['departure_delays'][i],
                        row.get('is_weekend_flags', [0] * row['seq_len'])[i],
                        row.get('is_peak_hour_flags', [0] * row['seq_len'])[i]
                    ])
                    # ✅ UPDATED: 8 weather features
                    all_weather.append([
                        row['temperatures'][i],
                        row['apparent_temps'][i],
                        row.get('precipitations', [0] * row['seq_len'])[i],
                        row.get('rains', [0] * row['seq_len'])[i],
                        row.get('snowfalls', [0] * row['seq_len'])[i],
                        row['windspeeds'][i],
                        row['windgusts'][i],
                        row['wind_directions'][i]
                    ])

            if all_speeds:
                self.speed_scaler.fit(np.array(all_speeds).reshape(-1, 1))
            if all_operational:
                self.operational_scaler.fit(np.array(all_operational))
            if all_weather:
                self.weather_scaler.fit(np.array(all_weather))
        else:
            self.target_scaler = target_scaler
            self.speed_scaler = speed_scaler
            self.operational_scaler = operational_scaler
            self.weather_scaler = weather_scaler

    def __len__(self):
        return len(self.paths_df)

    def __getitem__(self, idx):
        row = self.paths_df.iloc[idx]
        seq_len = int(row['seq_len'])

        seg_indices = []
        for seg_id in row['segment_ids']:
            seg_idx = self.seg_to_idx.get(seg_id, 0)
            seg_indices.append(seg_idx)

        temporal_features = []
        for i in range(seq_len):
            hour = row['hours'][i]
            dow = row['days_of_week'][i]
            speed = row['speeds'][i]

            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            dow_sin = np.sin(2 * np.pi * dow / 7)
            dow_cos = np.cos(2 * np.pi * dow / 7)

            speed_scaled = self.speed_scaler.transform([[speed]])[0, 0]

            temporal_features.append([
                hour_sin, hour_cos, dow_sin, dow_cos, speed_scaled
            ])

        # ✅ UPDATED: 4 operational features with binary flags
        operational_features = []
        for i in range(seq_len):
            if self.use_operational:
                operational_features.append([
                    row['arrival_delays'][i],
                    row['departure_delays'][i],
                    row.get('is_weekend_flags', [0] * seq_len)[i],
                    row.get('is_peak_hour_flags', [0] * seq_len)[i]
                ])
            else:
                operational_features.append([0.0, 0.0, 0.0, 0.0])

        if self.use_operational:
            operational_scaled = self.operational_scaler.transform(operational_features)
        else:
            operational_scaled = np.array(operational_features)

        # ✅ UPDATED: 8 weather features
        weather_features = []
        for i in range(seq_len):
            if self.use_weather:
                weather_features.append([
                    row['temperatures'][i],
                    row['apparent_temps'][i],
                    row.get('precipitations', [0] * seq_len)[i],
                    row.get('rains', [0] * seq_len)[i],
                    row.get('snowfalls', [0] * seq_len)[i],
                    row['windspeeds'][i],
                    row['windgusts'][i],
                    row['wind_directions'][i]
                ])
            else:
                weather_features.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        if self.use_weather:
            weather_scaled = self.weather_scaler.transform(weather_features)
        else:
            weather_scaled = np.array(weather_features)

        target_scaled = self.target_scaler.transform([[row['total_duration']]])[0, 0]

        # Padding
        while len(seg_indices) < self.max_path_length:
            seg_indices.append(0)
            temporal_features.append([0, 0, 0, 0, 0])
            operational_scaled = np.vstack([operational_scaled, [0, 0, 0, 0]])
            weather_scaled = np.vstack([weather_scaled, [0, 0, 0, 0, 0, 0, 0, 0]])

        seg_indices = torch.LongTensor(seg_indices[:self.max_path_length])
        temporal = torch.FloatTensor(temporal_features[:self.max_path_length])
        operational = torch.FloatTensor(operational_scaled[:self.max_path_length])
        weather = torch.FloatTensor(weather_scaled[:self.max_path_length])
        target = torch.FloatTensor([target_scaled])

        return seg_indices, temporal, operational, weather, target, seq_len


# =============================================================================
<<<<<<< Updated upstream
# MODELS (NO CHANGES TO CORE ARCHITECTURE)
=======
# TRIP DATASET  — groups segments by trip_id + calendar day
# =============================================================================

class TripDataset(Dataset):
    """
    Each sample is a chronologically ordered sequence of segments belonging
    to the same trip_id on the same calendar day.

    A 'trip' is defined as: all rows in segments_df that share the same
    trip_id AND whose departure date (derived from hour + day_of_week) falls
    on the same day.  In practice trip_id already encodes the vehicle run, so
    the same-day filter is a safety guard against trips that cross midnight.

    Each item returned:
        seg_indices        LongTensor  (T,)
        temporal           FloatTensor (T, 5)
        context_flags      FloatTensor (T, 2)   is_weekend, is_peak_hour
        origin_operational FloatTensor (T, 2)   arrivalDelay, departureDelay @ origin
        dest_operational   FloatTensor (T, 2)   same @ destination
        origin_weather     FloatTensor (T, 8)
        dest_weather       FloatTensor (T, 8)
        seg_targets        FloatTensor (T, 1)   per-segment duration (scaled)
        trip_target        FloatTensor (1,)     total trip duration (scaled)
        seq_len            int                  actual number of segments

    Pads all sequences to max_trip_length with zeros.
    """

    MAX_TRIP_LEN = 15   # cap very long runs; adjust via constructor

    def __init__(self,
                 segments_df,
                 segment_types,
                 max_trip_length: int = 15,
                 fit_scalers: bool = True,
                 target_scaler: RobustScaler = None,
                 speed_scaler: RobustScaler = None,
                 operational_scaler: RobustScaler = None,
                 weather_scaler: RobustScaler = None):

        self.max_trip_length  = max_trip_length
        self.segment_types    = list(segment_types)
        self.seg_to_idx       = {s: i for i, s in enumerate(self.segment_types)}

        df = segments_df.copy()

        # ── Cyclical temporal features ────────────────────────────────
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin']  = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos']  = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # ── Speed scaler ──────────────────────────────────────────────
        if 'speed_mps' not in df.columns:
            df['speed_mps'] = 0.0
        speed_vals = df[['speed_mps']].replace([np.inf, -np.inf], np.nan)
        if fit_scalers:
            self.speed_scaler = RobustScaler()
            df['speed_scaled'] = self.speed_scaler.fit_transform(
                speed_vals.fillna(speed_vals.median())).flatten()
        else:
            self.speed_scaler = speed_scaler
            df['speed_scaled'] = self.speed_scaler.transform(
                speed_vals.fillna(speed_vals.median())).flatten()
        df['speed_scaled'] = np.nan_to_num(df['speed_scaled'].values, nan=0.0)

        # ── Duration scaler ───────────────────────────────────────────
        if fit_scalers:
            self.target_scaler = RobustScaler()
            df['duration_scaled'] = self.target_scaler.fit_transform(
                df[['duration_sec']]).flatten()
        else:
            self.target_scaler = target_scaler
            df['duration_scaled'] = self.target_scaler.transform(
                df[['duration_sec']]).flatten()

        # ── Operational scaler (continuous only) ─────────────────────
        cont_op_cols = ['arrivalDelay', 'departureDelay']
        for c in cont_op_cols:
            if c not in df.columns:
                df[c] = 0.0
        op_data = df[cont_op_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if fit_scalers:
            self.operational_scaler = RobustScaler()
            op_scaled = self.operational_scaler.fit_transform(op_data)
        else:
            self.operational_scaler = operational_scaler
            op_scaled = self.operational_scaler.transform(op_data)
        df['arrivalDelay_scaled']   = op_scaled[:, 0]
        df['departureDelay_scaled'] = op_scaled[:, 1]

        # binary flags — no scaling
        for c in ['is_weekend', 'is_peak_hour']:
            if c not in df.columns:
                df[c] = 0.0

        # ── Weather scaler ────────────────────────────────────────────
        wx_cols = ['temperature_2m', 'apparent_temperature', 'precipitation',
                   'rain', 'snowfall', 'windspeed_10m', 'windgusts_10m',
                   'winddirection_10m']
        for c in wx_cols:
            if c not in df.columns:
                df[c] = 0.0
        wx_data = df[wx_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if fit_scalers:
            wx_mean = wx_data.mean().mean()
            wx_std  = wx_data.std().mean()
            if abs(wx_mean) < 0.5 and 0.5 < wx_std < 1.5:
                self.weather_scaler = None
                wx_scaled = wx_data.values
            else:
                self.weather_scaler = RobustScaler()
                wx_scaled = self.weather_scaler.fit_transform(wx_data)
        else:
            self.weather_scaler = weather_scaler
            wx_scaled = (self.weather_scaler.transform(wx_data)
                         if self.weather_scaler is not None
                         else wx_data.values)
        for i, c in enumerate(wx_cols):
            df[f'{c}_scaled'] = wx_scaled[:, i]

        # ── Drop rows with unknown segment types ──────────────────────
        df['seg_type_idx'] = df['segment_id'].map(self.seg_to_idx)
        n_before = len(df)
        df = df.dropna(subset=['seg_type_idx']).copy()
        df['seg_type_idx'] = df['seg_type_idx'].astype(int)
        if len(df) < n_before:
            print(f"   TripDataset: dropped {n_before - len(df):,} rows "
                  f"with unseen segment types")

        # ── Group into trips (trip_id + day_of_week as proxy for date) ─
        # segments_df contains 'hour' and 'day_of_week' from departure_time
        # We use trip_id as the primary key.  Within each trip_id we sort
        # chronologically by hour (good enough — trips rarely span >1 hour).
        trips = []
        for tid, grp in df.groupby('trip_id', sort=False):
            grp = grp.sort_values('hour').reset_index(drop=True)

            # Same-day guard: split at midnight crossings
            # (day_of_week changes mid-trip → separate trips)
            day_changes = (grp['day_of_week'].diff().fillna(0) != 0)
            split_ids   = day_changes.cumsum()
            for _, sub in grp.groupby(split_ids):
                sub = sub.reset_index(drop=True)
                if len(sub) < 2:          # at least 2 segments to form a trip
                    continue
                if len(sub) > max_trip_length:
                    sub = sub.head(max_trip_length)
                trips.append(sub)

        self.trips = trips
        print(f"   TripDataset: {len(trips):,} trips built from "
              f"{len(df):,} segments  "
              f"(max_trip_length={max_trip_length})")
        if trips:
            lengths = [len(t) for t in trips]
            print(f"     Trip length: min={min(lengths)}, "
                  f"max={max(lengths)}, "
                  f"mean={np.mean(lengths):.1f}")

        self._wx_cols_scaled = [f'{c}_scaled' for c in wx_cols]

    def __len__(self):
        return len(self.trips)

    def __getitem__(self, idx):
        from residual import split_features_for_segment

        trip = self.trips[idx]
        seq_len = len(trip)
        T = self.max_trip_length

        seg_indices        = torch.zeros(T, dtype=torch.long)
        temporal           = torch.zeros(T, 5)
        context_flags      = torch.zeros(T, 2)
        origin_operational = torch.zeros(T, 2)
        dest_operational   = torch.zeros(T, 2)
        origin_weather     = torch.zeros(T, 8)
        dest_weather       = torch.zeros(T, 8)
        seg_targets        = torch.zeros(T, 1)

        for t, (_, row) in enumerate(trip.iterrows()):
            seg_indices[t] = int(row['seg_type_idx'])
            temporal[t] = torch.tensor([
                float(row['hour_sin']), float(row['hour_cos']),
                float(row['dow_sin']),  float(row['dow_cos']),
                float(row['speed_scaled']),
            ])

            feats = split_features_for_segment(
                row,
                gps_lookup=getattr(self, 'gps_lookup', None),
                mapper=getattr(self, 'station_mapper', None),
            )
            context_flags[t]      = torch.FloatTensor(feats['context_flags'])
            origin_operational[t] = torch.FloatTensor(feats['origin_operational'])
            dest_operational[t]   = torch.FloatTensor(feats['dest_operational'])
            origin_weather[t]     = torch.FloatTensor(feats['origin_weather'])
            dest_weather[t]       = torch.FloatTensor(feats['dest_weather'])
            seg_targets[t, 0]     = float(row['duration_scaled'])

        # Global target = sum of actual (scaled) segment durations
        # We invert, sum, then re-scale so the global target is in the
        # same scaled space as individual targets.
        raw_durations = np.array([float(r['duration_sec'])
                                  for _, r in trip.iterrows()]).reshape(-1, 1)
        total_sec = raw_durations.sum()
        trip_target = torch.FloatTensor(
            self.target_scaler.transform([[total_sec]])[0])   # (1,)

        return (seg_indices, temporal,
                context_flags, origin_operational, dest_operational,
                origin_weather, dest_weather,
                seg_targets, trip_target, seq_len)


def trip_collate_fn(batch):
    """
    Collate function for TripDataset.

    Returns
    -------
    seg_indices        (B, T)
    temporal           (B, T, 5)
    context_flags      (B, T, 2)
    origin_operational (B, T, 2)
    dest_operational   (B, T, 2)
    origin_weather     (B, T, 8)
    dest_weather       (B, T, 8)
    seg_targets        (B, T, 1)   per-segment scaled durations
    trip_targets       (B, 1)      total trip scaled duration
    lengths            (B,)        actual trip lengths
    mask               (B, T)      True = valid segment
    """
    seg_indices        = torch.stack([b[0] for b in batch])   # (B, T)
    temporal           = torch.stack([b[1] for b in batch])   # (B, T, 5)
    context_flags      = torch.stack([b[2] for b in batch])
    origin_operational = torch.stack([b[3] for b in batch])
    dest_operational   = torch.stack([b[4] for b in batch])
    origin_weather     = torch.stack([b[5] for b in batch])
    dest_weather       = torch.stack([b[6] for b in batch])
    seg_targets        = torch.stack([b[7] for b in batch])   # (B, T, 1)
    trip_targets       = torch.stack([b[8] for b in batch])   # (B, 1)
    lengths            = torch.LongTensor([b[9] for b in batch])

    T    = seg_indices.size(1)
    mask = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T)

    return (seg_indices, temporal,
            context_flags, origin_operational, dest_operational,
            origin_weather, dest_weather,
            seg_targets, trip_targets, lengths, mask)


# =============================================================================
# MODELS (CORE ARCHITECTURE)
>>>>>>> Stashed changes
# =============================================================================

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3, alpha=0.2):
        super().__init__()
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = dropout

    def forward(self, h, adj):
        batch_size, num_nodes, _ = h.size()
        if not isinstance(adj, torch.Tensor):
            adj = torch.FloatTensor(adj)
        adj = adj.to(h.device)
        Wh = torch.matmul(h, self.W)
        Wh_i = Wh.unsqueeze(2).repeat(1, 1, num_nodes, 1)
        Wh_j = Wh.unsqueeze(1).repeat(1, num_nodes, 1, 1)
        a_input = torch.cat([Wh_i, Wh_j], dim=3)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.unsqueeze(0) > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        return h_prime


class MultiRelationalGAT(nn.Module):
    def __init__(self, n_heads, in_features, out_per_head, dropout=0.3):
        super().__init__()
        self.gat_heads = nn.ModuleList([
            GraphAttentionLayer(in_features, out_per_head, dropout)
            for _ in range(n_heads)
        ])
        self.out_proj = nn.Linear(out_per_head * n_heads, out_per_head)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj_list):
        head_outputs = [gat(h, adj) for gat, adj in zip(self.gat_heads, adj_list)]
        h_concat = torch.cat(head_outputs, dim=2)
        h_out = self.out_proj(h_concat)
        h_out = F.elu(h_out)
        return self.dropout(h_out)


class HistoricalEmbedding(nn.Module):
    def __init__(self, num_segments, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_segments, embed_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)

    def forward(self, segment_ids):
        return self.embedding(segment_ids)


class MAGTTE(nn.Module):
    def __init__(self, num_nodes, n_heads=3, node_embed_dim=32,
                 gat_hidden=32, lstm_hidden=64, historical_dim=16, dropout=0.3):
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, node_embed_dim)
        nn.init.normal_(self.node_embedding.weight, mean=0, std=0.1)
        self.multi_gat = MultiRelationalGAT(n_heads, node_embed_dim, gat_hidden, dropout)
        self.historical_embed = HistoricalEmbedding(num_nodes, historical_dim)

        fusion_in = gat_hidden + historical_dim + 5
        fusion_out = max(fusion_in // 2, 16)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_out),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=fusion_out,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )

        self.regression_head = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        self.register_buffer('adj_geo', None)
        self.register_buffer('adj_dist', None)
        self.register_buffer('adj_soc', None)

    def set_adjacency_matrices(self, adj_geo, adj_dist, adj_soc):
        self.register_buffer('adj_geo', torch.FloatTensor(adj_geo))
        self.register_buffer('adj_dist', torch.FloatTensor(adj_dist))
        self.register_buffer('adj_soc', torch.FloatTensor(adj_soc))

    def forward(self, seg_indices, temporal_features):
        all_nodes = self.node_embedding.weight.unsqueeze(0)
        spatial_all = self.multi_gat(all_nodes, [self.adj_geo, self.adj_dist, self.adj_soc])
        spatial_all = spatial_all.squeeze(0)
        segment_spatial = spatial_all[seg_indices]
        segment_historical = self.historical_embed(seg_indices)
        combined = torch.cat([segment_spatial, segment_historical, temporal_features], dim=1)
        fused = self.fusion(combined)
        lstm_out, _ = self.lstm(fused.unsqueeze(1))
        lstm_out = lstm_out.squeeze(1)
        return self.regression_head(lstm_out)


class GlobalTemporalAttention(nn.Module):
    def __init__(self, feature_dim, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.W_Q = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_K = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_V = nn.Linear(feature_dim, feature_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.scale = np.sqrt(feature_dim)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, V)
        out = self.layer_norm(out + x)
        return out, attn_weights


class LSTMWithGlobalTemporalAttention(nn.Module):
    def __init__(self, spatial_dim, operational_dim, weather_dim,
                 temporal_dim=5, hidden_dim=128, n_layers=1, dropout=0.1, out_dim=1):
        super().__init__()
        lstm_input_dim = spatial_dim + operational_dim + weather_dim + temporal_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.0
        )
        self.global_attention = GlobalTemporalAttention(feature_dim=hidden_dim, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, out_dim)
        )
        for layer in self.fusion:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, seq_x):
        batch_size = seq_x.size(0)
        seq_x = seq_x.reshape(batch_size, 1, -1)
        lstm_out, _ = self.lstm(seq_x)
        attn_out, _ = self.global_attention(lstm_out)
        attn_last = attn_out[:, -1, :]
        out = self.fusion(attn_last)
        return out


class MAGNN_LSTM(nn.Module):
    """✅ UPDATED: Now expects 4 operational features and 8 weather features"""

    def __init__(self, magnn_model, spatial_dim, operational_dim, weather_dim,
                 temporal_dim=5, lstm_hidden=32, lstm_layers=1, dropout=0.4, freeze_magnn=True):
        super().__init__()
        self.magnn = magnn_model
        self.freeze_magnn = freeze_magnn
        if freeze_magnn:
            for param in self.magnn.parameters():
                param.requires_grad = False
        self.lstm_model = LSTMWithGlobalTemporalAttention(
            spatial_dim=spatial_dim,
            operational_dim=operational_dim,  # Should be 4
            weather_dim=weather_dim,  # Should be 8
            temporal_dim=temporal_dim,
            hidden_dim=lstm_hidden,
            n_layers=lstm_layers,
            dropout=dropout,
            out_dim=1
        )

    def get_magnn_embeddings(self, seg_indices):
        all_nodes = self.magnn.node_embedding.weight.unsqueeze(0)
        spatial_all = self.magnn.multi_gat(
            all_nodes,
            [self.magnn.adj_geo, self.magnn.adj_dist, self.magnn.adj_soc]
        )
        spatial_all = spatial_all.squeeze(0)
        spatial_embeddings = spatial_all[seg_indices]
        return spatial_embeddings

    def forward(self, seg_indices, temporal_features, operational_features, weather_features):
        with torch.no_grad():
            magnn_baseline = self.magnn(seg_indices, temporal_features)
        if self.freeze_magnn:
            with torch.no_grad():
                spatial_embeddings = self.get_magnn_embeddings(seg_indices)
        else:
            spatial_embeddings = self.get_magnn_embeddings(seg_indices)
        combined_features = torch.cat([
            spatial_embeddings,
            operational_features,
            weather_features,
            temporal_features
        ], dim=1)
        seq_features = combined_features.unsqueeze(1)
        lstm_correction = self.lstm_model(seq_features)
        final_prediction = magnn_baseline + 0.5 * lstm_correction
        return final_prediction


class MAGNN_LSTM_MTL(nn.Module):
    """✅ UPDATED: Multi-Task Learning with updated feature dimensions"""

    def __init__(self, magnn_model, spatial_dim, operational_dim, weather_dim,
                 temporal_dim=5, lstm_hidden=64, lstm_layers=1, dropout=0.2,
                 mtl_segment_hidden=64, mtl_path_hidden=128,
                 freeze_magnn=True):
        super().__init__()

        self.magnn = magnn_model
        self.freeze_magnn = freeze_magnn
        self.lstm_input_dim = spatial_dim + operational_dim + weather_dim + temporal_dim

        if freeze_magnn:
            for param in self.magnn.parameters():
                param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.0
        )

        self.global_attention = GlobalTemporalAttention(
            feature_dim=lstm_hidden,
            dropout=dropout
        )

        self.mtl_head = MTLHead(
            feature_dim=lstm_hidden,
            segment_hidden=mtl_segment_hidden,
            path_hidden=mtl_path_hidden,
            dropout=dropout
        )

    def get_magnn_embeddings(self, seg_indices):
        all_nodes = self.magnn.node_embedding.weight.unsqueeze(0)
        spatial_all = self.magnn.multi_gat(
            all_nodes,
            [self.magnn.adj_geo, self.magnn.adj_dist, self.magnn.adj_soc]
        )
        spatial_all = spatial_all.squeeze(0)
        spatial_embeddings = spatial_all[seg_indices]
        return spatial_embeddings

    def forward(self, seg_indices, temporal_features, operational_features, weather_features,
                mask=None, return_components=False):
        batch_size, seq_len, _ = temporal_features.size()

        if self.freeze_magnn:
            with torch.no_grad():
                spatial_embeddings = self.get_magnn_embeddings(seg_indices)
        else:
            spatial_embeddings = self.get_magnn_embeddings(seg_indices)

        combined_sequence = []
        for t in range(seq_len):
            combined = torch.cat([
                spatial_embeddings[:, t, :],
                operational_features[:, t, :],
                weather_features[:, t, :],
                temporal_features[:, t, :]
            ], dim=1)
            combined_sequence.append(combined)

        combined_sequence = torch.stack(combined_sequence, dim=1)
        lstm_out, _ = self.lstm(combined_sequence)
        attn_out, attn_weights = self.global_attention(lstm_out)

        if return_components:
            individual_preds, collective_pred, attention = self.mtl_head(
                attn_out, mask=mask, return_components=True
            )
            return individual_preds, collective_pred, attention
        else:
            collective_pred = self.mtl_head(attn_out, mask=mask, return_components=False)
            return collective_pred


<<<<<<< Updated upstream
=======
# MAGNN_LSTM_DualTaskMTL is imported from mtl.py
# It wraps a frozen MAGNN_LSTM_Residual as the per-segment encoder
# and adds a TripSequenceEncoder + DualTaskMTLHead on top.


>>>>>>> Stashed changes
class SimpleMLP(nn.Module):
    def __init__(self, num_segments, embed_dim=32, dropout=0.3):
        super().__init__()
        self.seg_embed = nn.Embedding(num_segments, embed_dim)
        nn.init.normal_(self.seg_embed.weight, 0, 0.1)
        in_dim = embed_dim + 5
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, seg_indices, temporal):
        emb = self.seg_embed(seg_indices)
        x = torch.cat([emb, temporal], dim=1)
        return self.net(x)


# =============================================================================
# COLLATE FUNCTIONS
# =============================================================================

def masked_collate_fn(batch):
    seg_indices = torch.LongTensor([item[0] for item in batch])
    targets = torch.stack([item[2] for item in batch])
    lengths = torch.LongTensor([item[3] for item in batch])
    temporal_seqs = [item[1].unsqueeze(0) for item in batch]
    max_len = int(lengths.max().item())
    temporal_dim = temporal_seqs[0].size(-1)
    temporal_pad = torch.zeros(len(batch), max_len, temporal_dim)
    for i, (seq, slen) in enumerate(zip(temporal_seqs, lengths)):
        slen = int(slen.item())
        temporal_pad[i, :slen, :] = seq[:slen]
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return seg_indices, temporal_pad, targets, lengths, mask


# AFTER
def enhanced_collate_fn(batch):
    seg_indices = torch.LongTensor([item[0] for item in batch])
    targets     = torch.stack([item[7] for item in batch])
    lengths     = torch.LongTensor([item[8] for item in batch])

    def _pad(tensors, lengths):
        max_len = int(max(lengths))
        dim = tensors[0].size(-1)
        out = torch.zeros(len(tensors), max_len, dim)
        for i, (t, l) in enumerate(zip(tensors, lengths)):
            out[i, :int(l), :] = t.unsqueeze(0)[:, :int(l), :]
        return out

    temporal_pad        = _pad([item[1].unsqueeze(0) for item in batch], lengths)
    context_pad         = _pad([item[2].unsqueeze(0) for item in batch], lengths)
    origin_op_pad       = _pad([item[3].unsqueeze(0) for item in batch], lengths)
    dest_op_pad         = _pad([item[4].unsqueeze(0) for item in batch], lengths)
    origin_weather_pad  = _pad([item[5].unsqueeze(0) for item in batch], lengths)
    dest_weather_pad    = _pad([item[6].unsqueeze(0) for item in batch], lengths)

    mask = torch.arange(temporal_pad.size(1)).unsqueeze(0) < lengths.unsqueeze(1)

    return (seg_indices,
            temporal_pad, context_pad,
            origin_op_pad, dest_op_pad,
            origin_weather_pad, dest_weather_pad,
            targets, lengths, mask)


def path_collate_fn(batch):
    seg_indices = torch.stack([item[0] for item in batch])
    temporal = torch.stack([item[1] for item in batch])
    operational = torch.stack([item[2] for item in batch])
    weather = torch.stack([item[3] for item in batch])
    targets = torch.stack([item[4] for item in batch])
    lengths = torch.LongTensor([item[5] for item in batch])

    max_len = seg_indices.size(1)
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    return seg_indices, temporal, operational, weather, targets, lengths, mask


# =============================================================================
# TRAINING UTILITIES (NO CHANGES)
# =============================================================================

def _safe_batch(seg_idx, temporal, target):
    t_flat = temporal.reshape(temporal.size(0), -1)
    valid = ~torch.isnan(t_flat).any(dim=1)
    valid &= ~torch.isinf(t_flat).any(dim=1)
    valid &= ~torch.isnan(target).any(dim=1)
    if valid.sum() == 0:
        return None, None, None
    return seg_idx[valid], temporal[valid], target[valid]


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        seg_idx, temporal_pad, target, lengths, mask = batch
        temporal = temporal_pad.squeeze(1)
        seg_idx, temporal, target = _safe_batch(
            seg_idx.to(device), temporal.to(device), target.to(device)
        )
        if seg_idx is None:
            continue
        predictions = model(seg_idx, temporal)
        loss = criterion(predictions, target)
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate(model, dataloader, criterion, device, scaler):
    model.eval()
    predictions_list = []
    targets_list = []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            seg_idx, temporal_pad, target, lengths, mask = batch
            temporal = temporal_pad.squeeze(1)
            seg_idx, temporal, target = _safe_batch(
                seg_idx.to(device), temporal.to(device), target.to(device)
            )
            if seg_idx is None:
                continue
            predictions = model(seg_idx, temporal)
            loss = criterion(predictions, target)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            total_loss += loss.item()
            n_batches += 1
            predictions_list.append(predictions.cpu().numpy())
            targets_list.append(target.cpu().numpy())
    if not predictions_list:
        return {'loss': float('nan'), 'r2': float('nan'),
                'rmse': float('nan'), 'mae': float('nan'),
                'mape': float('nan'), 'preds': [], 'actual': []}
    preds = np.concatenate(predictions_list)
    targets = np.concatenate(targets_list)
    preds_orig = scaler.inverse_transform(preds)
    targets_orig = scaler.inverse_transform(targets)
    r2 = r2_score(targets_orig, preds_orig)
    rmse = np.sqrt(mean_squared_error(targets_orig, preds_orig))
    mae = mean_absolute_error(targets_orig, preds_orig)
    mask = targets_orig.flatten() > 0
    mape = (np.mean(np.abs((targets_orig.flatten()[mask] - preds_orig.flatten()[mask]) /
                           targets_orig.flatten()[mask])) * 100 if mask.any() else float('nan'))
    return {
        'loss': total_loss / max(n_batches, 1),
        'r2': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'preds': preds_orig.flatten().tolist(),
        'actual': targets_orig.flatten().tolist(),
    }


# =============================================================================
# TRAINING FUNCTIONS (NO CHANGES TO EXISTING)
# =============================================================================

def train_magtte(train_loader, val_loader, test_loader,
                 adj_geo, adj_dist, adj_soc,
                 segment_types, scaler,
                 output_folder, device, cfg):
    print_section("MAGTTE + GAT TRAINING")
    num_segments = len(segment_types)
    print(f"   Segments: {num_segments}, Epochs: {cfg.n_epochs}, Device: {device}")

    model = MAGTTE(num_segments, cfg.n_heads, cfg.node_embed_dim, cfg.gat_hidden,
                   cfg.lstm_hidden, cfg.historical_dim, cfg.dropout).to(device)
    model.set_adjacency_matrices(adj_geo, adj_dist, adj_soc)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.lr_scheduler_factor,
                                                     patience=cfg.lr_scheduler_patience)
    criterion = nn.SmoothL1Loss()
    best_val_loss = float('inf')
    patience_counter = 0
    best_ckpt = os.path.join(output_folder, 'magtte_best.pth')

    print()
    for epoch in range(1, cfg.n_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device, scaler)
        val_loss = val_metrics['loss']
        scheduler.step(val_loss if not np.isnan(val_loss) else best_val_loss)

        if epoch % max(1, cfg.n_epochs // 5) == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{cfg.n_epochs}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_R²={val_metrics['r2']:.4f}")

        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f"\n  ⏹️  Early stopping at epoch {epoch}")
                break

    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    print_section("MAGTTE — FINAL RESULTS")

    def _eval(loader, name):
        m = evaluate(model, loader, criterion, device, scaler)
        print(f"   {name:<6}  R²={m['r2']:.4f}  RMSE={m['rmse']:.2f}s  "
              f"MAE={m['mae']:.2f}s  MAPE={m['mape']:.2f}%")
        return m

    results = {'Train': _eval(train_loader, 'Train'), 'Val': _eval(val_loader, 'Val'),
               'Test': _eval(test_loader, 'Test')}

    test_res = results.get('Test', {})
    if test_res.get('preds'):
        print(f"\n{'Idx':>4}  {'Actual(s)':>10}  {'Pred(s)':>10}  {'Error(s)':>9}  {'Error%':>7}")
        print("  " + "-" * 48)
        for i in range(min(20, len(test_res['actual']))):
            a, p = test_res['actual'][i], test_res['preds'][i]
            err = p - a
            pct = (err / a * 100) if a > 0 else 0.0
            print(f"  {i:>3}  {a:>10.2f}  {p:>10.2f}  {err:>9.2f}  {pct:>6.2f}%")

    return results, model


def train_magnn_lstm(train_loader, val_loader, test_loader,
                     adj_geo, adj_dist, adj_soc,
                     segment_types, scaler,
                     output_folder, device, cfg,
                     pretrained_magnn_path=None,
                     freeze_magnn=True):
    print_section("MAGNN-LSTM TRAINING")
    num_segments = len(segment_types)

    magnn_base = MAGTTE(num_segments, cfg.n_heads, cfg.node_embed_dim, cfg.gat_hidden,
                        cfg.lstm_hidden, cfg.historical_dim, cfg.dropout).to(device)
    magnn_base.set_adjacency_matrices(adj_geo, adj_dist, adj_soc)

    if pretrained_magnn_path and os.path.exists(pretrained_magnn_path):
        magnn_base.load_state_dict(torch.load(pretrained_magnn_path, map_location=device))
        print(f"   ✓ Loaded pre-trained MAGNN")

    # ✅ UPDATED: operational_dim=4, weather_dim=8
    model = MAGNN_LSTM(magnn_base, cfg.gat_hidden, 4, 8, 5, 32, 1, 0.1, True).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params: {n_params:,}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg.learning_rate * 0.1, weight_decay=cfg.weight_decay * 5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.SmoothL1Loss()
    best_val_loss = float('inf')
    patience_counter = 0
    best_ckpt = os.path.join(output_folder, 'magnn_lstm_best.pth')

    print()
    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            seg_idx, temporal, operational, weather, target, lengths, mask = batch
            seg_idx = seg_idx.to(device)
            temporal = temporal.squeeze(1).to(device)
            operational = operational.squeeze(1).to(device)
            weather = weather.squeeze(1).to(device)
            target = target.to(device)
            predictions = model(seg_idx, temporal, operational, weather)
            loss = criterion(predictions, target)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                seg_idx, temporal, operational, weather, target, lengths, mask = batch
                seg_idx = seg_idx.to(device)
                temporal = temporal.squeeze(1).to(device)
                operational = operational.squeeze(1).to(device)
                weather = weather.squeeze(1).to(device)
                target = target.to(device)
                predictions = model(seg_idx, temporal, operational, weather)
                loss = criterion(predictions, target)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
                    n_val += 1
                    val_preds.append(predictions.cpu().numpy())
                    val_targets.append(target.cpu().numpy())
        val_loss /= max(n_val, 1)
        scheduler.step(val_loss)

        if val_preds:
            vp = scaler.inverse_transform(np.concatenate(val_preds))
            vt = scaler.inverse_transform(np.concatenate(val_targets))
            val_r2 = r2_score(vt, vp)
        else:
            val_r2 = float('nan')

        if epoch % max(1, cfg.n_epochs // 5) == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{cfg.n_epochs}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_R²={val_r2:.4f}")

        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f"\n  ⏹️  Early stopping at epoch {epoch}")
                break

    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    print_section("MAGNN-LSTM — FINAL RESULTS")

    def _eval(loader, name):
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in loader:
                seg_idx, temporal, operational, weather, target, lengths, mask = batch
                seg_idx = seg_idx.to(device)
                temporal = temporal.squeeze(1).to(device)
                operational = operational.squeeze(1).to(device)
                weather = weather.squeeze(1).to(device)
                pred = model(seg_idx, temporal, operational, weather)
                preds.append(pred.cpu().numpy())
                targets.append(target.cpu().numpy())
        if not preds:
            return {}
        p = scaler.inverse_transform(np.concatenate(preds))
        t = scaler.inverse_transform(np.concatenate(targets))
        r2 = r2_score(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))
        mae = mean_absolute_error(t, p)
        mask = t.flatten() > 0
        mape = np.mean(np.abs((t.flatten()[mask] - p.flatten()[mask]) /
                              t.flatten()[mask])) * 100 if mask.any() else float('nan')
        print(f"   {name:<6}  R²={r2:.4f}  RMSE={rmse:.2f}s  MAE={mae:.2f}s  MAPE={mape:.2f}%")
        return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape,
                'preds': p.flatten().tolist(), 'actual': t.flatten().tolist()}

    results = {'Train': _eval(train_loader, 'Train'), 'Val': _eval(val_loader, 'Val'),
               'Test': _eval(test_loader, 'Test')}

    test_res = results.get('Test', {})
    if test_res.get('preds'):
        print(f"\n{'Idx':>4}  {'Actual(s)':>10}  {'Pred(s)':>10}  {'Error(s)':>9}  {'Error%':>7}")
        print("  " + "-" * 48)
        for i in range(min(20, len(test_res['actual']))):
            a, p = test_res['actual'][i], test_res['preds'][i]
            err = p - a
            epct = (err / a * 100) if a > 0 else 0.0
            print(f"  {i:>3}  {a:>10.2f}  {p:>10.2f}  {err:>9.2f}  {epct:>6.2f}%")

    return results, model


def train_magnn_lstm_mtl(train_loader, val_loader, test_loader,
                         adj_geo, adj_dist, adj_soc,
                         segment_types, scaler,
                         output_folder, device, cfg,
                         pretrained_magnn_path=None,
                         freeze_magnn=True):
    """Train MAGNN-LSTM-MTL with REAL multi-segment paths."""
    print_section("MAGNN-LSTM-MTL TRAINING (Multi-Segment Paths)")
    num_segments = len(segment_types)

    magnn_base = MAGTTE(num_segments, cfg.n_heads, cfg.node_embed_dim, cfg.gat_hidden,
                        cfg.lstm_hidden, cfg.historical_dim, cfg.dropout).to(device)
    magnn_base.set_adjacency_matrices(adj_geo, adj_dist, adj_soc)

    if pretrained_magnn_path and os.path.exists(pretrained_magnn_path):
        magnn_base.load_state_dict(torch.load(pretrained_magnn_path, map_location=device))
        print(f"   ✓ Loaded pre-trained MAGNN from {pretrained_magnn_path}")

    # ✅ UPDATED: operational_dim=4, weather_dim=8
    model = MAGNN_LSTM_MTL(
        magnn_base,
        cfg.gat_hidden,
        4,  # operational_dim
        8,  # weather_dim
        5,  # temporal_dim
        64,  # lstm_hidden
        1,  # lstm_layers
        0.2,  # dropout
        64,  # mtl_segment_hidden
        128,  # mtl_path_hidden
        freeze_magnn
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params: {n_params:,}")
    print(f"   MTL lambda: {cfg.mtl_lambda} (individual vs collective balance)")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate * 0.1,
        weight_decay=cfg.weight_decay * 5
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    mtl_criterion = MTLLoss(lambda_weight=cfg.mtl_lambda, criterion=nn.SmoothL1Loss())

    best_val_loss = float('inf')
    patience_counter = 0
    best_ckpt = os.path.join(output_folder, 'magnn_lstm_mtl_best.pth')

    print()
    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        train_loss = 0.0
        train_ind_loss = 0.0
        train_col_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            seg_idx, temporal, operational, weather, target, lengths, mask = batch
            seg_idx = seg_idx.to(device)
            temporal = temporal.to(device)
            operational = operational.to(device)
            weather = weather.to(device)
            target = target.to(device)
            mask = mask.to(device)

            individual_preds, collective_pred, attention = model(
                seg_idx, temporal, operational, weather, mask, return_components=True
            )

            loss, loss_dict = mtl_criterion(individual_preds, collective_pred, target, mask)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss_dict['total']
            train_ind_loss += loss_dict['individual']
            train_col_loss += loss_dict['collective']
            n_batches += 1

        train_loss /= max(n_batches, 1)
        train_ind_loss /= max(n_batches, 1)
        train_col_loss /= max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                seg_idx, temporal, operational, weather, target, lengths, mask = batch
                seg_idx = seg_idx.to(device)
                temporal = temporal.to(device)
                operational = operational.to(device)
                weather = weather.to(device)
                target = target.to(device)
                mask = mask.to(device)

                individual_preds, collective_pred, attention = model(
                    seg_idx, temporal, operational, weather, mask, return_components=True
                )

                loss, loss_dict = mtl_criterion(individual_preds, collective_pred, target, mask)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss_dict['total']
                    n_val += 1
                    val_preds.append(collective_pred.cpu().numpy())
                    val_targets.append(target.cpu().numpy())

        val_loss /= max(n_val, 1)
        scheduler.step(val_loss)

        if val_preds:
            vp = scaler.inverse_transform(np.concatenate(val_preds))
            vt = scaler.inverse_transform(np.concatenate(val_targets))
            val_r2 = r2_score(vt, vp)
        else:
            val_r2 = float('nan')

        if epoch % max(1, cfg.n_epochs // 5) == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{cfg.n_epochs}  "
                  f"train_loss={train_loss:.4f} (ind:{train_ind_loss:.4f} col:{train_col_loss:.4f})  "
                  f"val_loss={val_loss:.4f}  val_R²={val_r2:.4f}")

        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f"\n  ⏹️  Early stopping at epoch {epoch}")
                break

    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    print_section("MAGNN-LSTM-MTL — FINAL RESULTS")

    def _eval(loader, name):
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in loader:
                seg_idx, temporal, operational, weather, target, lengths, mask = batch
                seg_idx = seg_idx.to(device)
                temporal = temporal.to(device)
                operational = operational.to(device)
                weather = weather.to(device)
                mask = mask.to(device)

                pred = model(seg_idx, temporal, operational, weather, mask, return_components=False)
                preds.append(pred.cpu().numpy())
                targets.append(target.cpu().numpy())

        if not preds:
            return {}

        p = scaler.inverse_transform(np.concatenate(preds))
        t = scaler.inverse_transform(np.concatenate(targets))
        r2 = r2_score(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))
        mae = mean_absolute_error(t, p)
        mask_valid = t.flatten() > 0
        mape = np.mean(np.abs((t.flatten()[mask_valid] - p.flatten()[mask_valid]) /
                              t.flatten()[mask_valid])) * 100 if mask_valid.any() else float('nan')

        print(f"   {name:<6}  R²={r2:.4f}  RMSE={rmse:.2f}s  MAE={mae:.2f}s  MAPE={mape:.2f}%")
        return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape,
                'preds': p.flatten().tolist(), 'actual': t.flatten().tolist()}

    results = {'Train': _eval(train_loader, 'Train'),
               'Val': _eval(val_loader, 'Val'),
               'Test': _eval(test_loader, 'Test')}

    test_res = results.get('Test', {})
    if test_res.get('preds'):
        print(f"\n{'Idx':>4}  {'Actual(s)':>10}  {'Pred(s)':>10}  {'Error(s)':>9}  {'Error%':>7}")
        print("  " + "-" * 48)
        for i in range(min(20, len(test_res['actual']))):
            a, p = test_res['actual'][i], test_res['preds'][i]
            err = p - a
            epct = (err / a * 100) if a > 0 else 0.0
            print(f"  {i:>3}  {a:>10.2f}  {p:>10.2f}  {err:>9.2f}  {epct:>6.2f}%")

    return results, model


<<<<<<< Updated upstream
=======
# =============================================================================
# DUAL-TASK MTL TRAINING FUNCTION  (trip-aware, residual-encoder)
# =============================================================================

def train_magnn_lstm_dualtask_mtl(train_loader, val_loader, test_loader,
                                   adj_geo, adj_dist, adj_soc,
                                   segment_types, scaler,
                                   output_folder, device, cfg,
                                   pretrained_residual_model=None,
                                   pretrained_magnn_path=None,
                                   freeze_magnn=True):
    """
    Train MAGNN-LSTM-DualTaskMTL.

    LOCAL  task: predict each segment's duration within a trip.
    GLOBAL task: predict the total trip travel time
                 (sum of all segment durations, same trip_id + same day).

    The model wraps a frozen MAGNN_LSTM_Residual as its per-segment encoder.
    Only the TripSequenceEncoder and DualTaskMTLHead are trained here.

    Parameters
    ----------
    pretrained_residual_model : MAGNN_LSTM_Residual | None
        If provided and already on `device`, used directly.
        Otherwise a fresh residual model is built from pretrained_magnn_path.
    pretrained_magnn_path : str | None
        Path to a saved MAGTTE checkpoint; used when
        pretrained_residual_model is None.
    """
    from residual import MAGNN_LSTM_Residual

    print_section("MAGNN-LSTM-DUALTASK-MTL TRAINING  (Trip-Aware Residual Encoder)")
    num_segments = len(segment_types)

    # ── Build / reuse the residual encoder ─────────────────────────────
    if pretrained_residual_model is not None:
        residual_enc = pretrained_residual_model
        print("   ✓ Reusing pre-trained MAGNN_LSTM_Residual encoder")
    else:
        magnn_base = MAGTTE(
            num_segments, cfg.n_heads, cfg.node_embed_dim,
            cfg.gat_hidden, cfg.lstm_hidden, cfg.historical_dim,
            cfg.dropout).to(device)
        magnn_base.set_adjacency_matrices(adj_geo, adj_dist, adj_soc)

        if pretrained_magnn_path and os.path.exists(pretrained_magnn_path):
            magnn_base.load_state_dict(
                torch.load(pretrained_magnn_path, map_location=device))
            print(f"   ✓ Loaded pre-trained MAGNN from {pretrained_magnn_path}")

        residual_enc = MAGNN_LSTM_Residual(
            magnn_base,
            spatial_dim=cfg.gat_hidden,
            freeze_magnn=freeze_magnn,
            lstm_hidden=128,
            lstm_layers=1,
            dropout=0.2,
            temporal_dim=5,
        ).to(device)
        print("   ✓ Built fresh MAGNN_LSTM_Residual encoder")

    # ── Build the DualTaskMTL wrapper ───────────────────────────────────
    model = MAGNN_LSTM_DualTaskMTL(
        residual_model=residual_enc,
        spatial_dim=cfg.gat_hidden,
        enc_hidden=128,
        local_hidden=64,
        global_hidden=128,
        dropout=0.2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params: {n_params:,}  "
          f"(only TripEncoder + MTLHead)")

    lr = getattr(cfg, 'residual_learning_rate', 5e-4)
    wd = getattr(cfg, 'lstm_weight_decay', 1e-6)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    criterion = DualTaskMTLLoss()

    best_val_loss    = float('inf')
    patience_counter = 0
    best_ckpt = os.path.join(output_folder,
                             'magnn_lstm_dualtask_mtl_best.pth')

    def _unpack(batch, dev):
        (seg_idx, temporal, ctx, o_op, d_op,
         o_wx, d_wx, seg_tgt, trip_tgt, lengths, mask) = batch
        return (seg_idx.to(dev), temporal.to(dev),
                ctx.to(dev), o_op.to(dev), d_op.to(dev),
                o_wx.to(dev), d_wx.to(dev),
                seg_tgt.to(dev), trip_tgt.to(dev),
                lengths.to(dev), mask.to(dev))

    print()
    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        tr_total = tr_local = tr_global = 0.0
        n_batches = 0

        for batch in train_loader:
            (seg_idx, temporal, ctx, o_op, d_op,
             o_wx, d_wx, seg_tgt, trip_tgt,
             lengths, mask) = _unpack(batch, device)

            local_preds, global_pred = model(
                seg_idx, temporal, ctx, o_op, d_op,
                o_wx, d_wx, mask, lengths, return_local=True)

            loss, ld = criterion(
                local_preds, global_pred,
                seg_tgt, trip_tgt, mask,
                model.mtl_head.log_var_local,
                model.mtl_head.log_var_global)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_total  += ld['total']
            tr_local  += ld['local']
            tr_global += ld['global']
            n_batches += 1

        tr_total  /= max(n_batches, 1)
        tr_local  /= max(n_batches, 1)
        tr_global /= max(n_batches, 1)

        # ── Validation ───────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                (seg_idx, temporal, ctx, o_op, d_op,
                 o_wx, d_wx, seg_tgt, trip_tgt,
                 lengths, mask) = _unpack(batch, device)

                local_preds, global_pred = model(
                    seg_idx, temporal, ctx, o_op, d_op,
                    o_wx, d_wx, mask, lengths, return_local=True)

                loss, ld = criterion(
                    local_preds, global_pred,
                    seg_tgt, trip_tgt, mask,
                    model.mtl_head.log_var_local,
                    model.mtl_head.log_var_global)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += ld['total']
                    n_val    += 1
                    val_preds.append(global_pred.cpu().numpy())
                    val_targets.append(trip_tgt.cpu().numpy())

        val_loss /= max(n_val, 1)
        scheduler.step(val_loss)

        if val_preds:
            vp = scaler.inverse_transform(np.concatenate(val_preds))
            vt = scaler.inverse_transform(np.concatenate(val_targets))
            val_r2 = r2_score(vt, vp)
        else:
            val_r2 = float('nan')

        if epoch % max(1, cfg.n_epochs // 5) == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{cfg.n_epochs}  "
                  f"loss={tr_total:.4f} "
                  f"(local:{tr_local:.4f} global:{tr_global:.4f})  "
                  f"val_loss={val_loss:.4f}  val_R²={val_r2:.4f}")

        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f"\n  ⏹️  Early stopping at epoch {epoch}")
                break

    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # ── Final evaluation ─────────────────────────────────────────────
    print_section("MAGNN-LSTM-DUALTASK-MTL — FINAL RESULTS")

    def _eval(loader, name):
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in loader:
                (seg_idx, temporal, ctx, o_op, d_op,
                 o_wx, d_wx, seg_tgt, trip_tgt,
                 lengths, mask) = _unpack(batch, device)

                global_pred = model(
                    seg_idx, temporal, ctx, o_op, d_op,
                    o_wx, d_wx, mask, lengths, return_local=False)

                preds.append(global_pred.cpu().numpy())
                targets.append(trip_tgt.cpu().numpy())

        if not preds:
            return {}

        p = scaler.inverse_transform(np.concatenate(preds))
        t = scaler.inverse_transform(np.concatenate(targets))
        r2   = r2_score(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))
        mae  = mean_absolute_error(t, p)
        mv   = t.flatten() > 0
        mape = (np.mean(np.abs((t.flatten()[mv] - p.flatten()[mv]) /
                               t.flatten()[mv])) * 100
                if mv.any() else float('nan'))

        print(f"   {name:<6}  R²={r2:.4f}  RMSE={rmse:.2f}s  "
              f"MAE={mae:.2f}s  MAPE={mape:.2f}%")
        return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape,
                'preds': p.flatten().tolist(),
                'actual': t.flatten().tolist()}

    results = {
        'Train': _eval(train_loader, 'Train'),
        'Val':   _eval(val_loader,   'Val'),
        'Test':  _eval(test_loader,  'Test'),
    }

    test_res = results.get('Test', {})
    if test_res.get('preds'):
        print(f"\n{'Idx':>4}  {'Actual(s)':>10}  {'Pred(s)':>10}  "
              f"{'Error(s)':>9}  {'Error%':>7}")
        print("  " + "-" * 48)
        for i in range(min(20, len(test_res['actual']))):
            a, p_v = test_res['actual'][i], test_res['preds'][i]
            err  = p_v - a
            epct = (err / a * 100) if a > 0 else 0.0
            print(f"  {i:>3}  {a:>10.2f}  {p_v:>10.2f}  "
                  f"{err:>9.2f}  {epct:>6.2f}%")

    return results, model


>>>>>>> Stashed changes
def train_simple(train_loader, val_loader, test_loader, segment_types, scaler,
                 output_folder, device, n_epochs=50, lr=0.001, dropout=0.3):
    print_section("SIMPLE MLP TRAINING")
    num_segments = len(segment_types)
    model = SimpleMLP(num_segments, embed_dim=32, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.SmoothL1Loss()
    best_val_loss = float('inf')

    print(f"   Segments: {num_segments}, Epochs: {n_epochs}, Device: {device}\n")

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            seg_idx, temporal_pad, target, lengths, mask = batch
            temporal = temporal_pad.squeeze(1)
            pred = model(seg_idx.to(device), temporal.to(device))
            loss = criterion(pred, target.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                seg_idx, temporal_pad, target, lengths, mask = batch
                temporal = temporal_pad.squeeze(1)
                pred = model(seg_idx.to(device), temporal.to(device))
                val_loss += criterion(pred, target.to(device)).item()
        val_loss /= len(val_loader)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:>3}/{n_epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_folder, 'simple_best.pth'))

    return {}, model