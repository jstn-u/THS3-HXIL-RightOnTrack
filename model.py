"""
model.py — UPDATED v3  (Dual-MTL with prep.py integration)
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
                 MAGNN_LSTM_DualTaskMTL, TripSequenceEncoder,
                 HierarchicalTripPredictor)

_CONTEXT_DIM = 4

warnings.filterwarnings('ignore')


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


class EnhancedSegmentDataset(SegmentDataset):
    """Smart weather scaling + don't scale binary flags."""

    def __init__(self, segments_df, segment_types,
                 fit_scalers: bool = True,
                 target_scaler: RobustScaler = None,
                 speed_scaler: RobustScaler = None,
                 operational_scaler: RobustScaler = None,
                 weather_scaler: RobustScaler = None):
        super().__init__(segments_df, segment_types, fit_scalers, target_scaler, speed_scaler)

        continuous_operational_cols = ['arrivalDelay', 'departureDelay']
        binary_flag_cols = ['is_weekend', 'is_peak_hour']

        for col in continuous_operational_cols + binary_flag_cols:
            if col not in self.segments_df.columns:
                self.segments_df[col] = 0.0

        continuous_data = self.segments_df[continuous_operational_cols].copy()
        continuous_data = continuous_data.replace([np.inf, -np.inf], np.nan)
        continuous_data = continuous_data.fillna(continuous_data.median())

        if fit_scalers:
            self.operational_scaler = RobustScaler()
            continuous_scaled = self.operational_scaler.fit_transform(continuous_data)
        else:
            if operational_scaler is None:
                raise ValueError("operational_scaler must be provided")
            self.operational_scaler = operational_scaler
            continuous_scaled = self.operational_scaler.transform(continuous_data)

        for i, col in enumerate(continuous_operational_cols):
            self.segments_df[f'{col}_scaled'] = continuous_scaled[:, i]

        for col in binary_flag_cols:
            self.segments_df[f'{col}_scaled'] = self.segments_df[col].values

        self.operational_cols_scaled = [f'{col}_scaled' for col in continuous_operational_cols + binary_flag_cols]

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
            weather_mean = weather_data.mean().mean()
            weather_std = weather_data.std().mean()
            if abs(weather_mean) < 0.5 and 0.5 < weather_std < 1.5:
                self.weather_scaler = None
                weather_scaled = weather_data.values
            else:
                self.weather_scaler = RobustScaler()
                weather_scaled = self.weather_scaler.fit_transform(weather_data)
        else:
            self.weather_scaler = weather_scaler
            if self.weather_scaler is not None:
                weather_scaled = self.weather_scaler.transform(weather_data)
            else:
                weather_scaled = weather_data.values

        for i, col in enumerate(weather_cols):
            self.segments_df[f'{col}_scaled'] = weather_scaled[:, i]

        self.weather_cols_scaled = [f'{col}_scaled' for col in weather_cols]

    def __getitem__(self, idx):
        row = self.segments_df.iloc[idx]
        seg_type_idx = int(row['seg_type_idx'])
        seq_len = int(row['seq_len'])

        temporal = torch.FloatTensor([
            float(row['hour_sin']), float(row['hour_cos']),
            float(row['dow_sin']), float(row['dow_cos']),
            float(row['speed_scaled']),
        ])

        op_raw = np.array([
            float(row.get('arrivalDelay', 0) or 0),
            float(row.get('departureDelay', row.get('arrivalDelay', 0)) or 0),
            float(row.get('dwellTime_sec', 0) or 0),
        ], dtype=np.float32)
        try:
            op_sc = self.operational_scaler.transform(op_raw[:2].reshape(1,-1))[0]
            op_vec = np.append(op_sc, op_raw[2]).astype(np.float32)
        except Exception:
            op_vec = op_raw

        wx_cols = ['Temperature_2m','Apparent_temperature','Precipitation',
                   'Rain','Snowfall','Windspeed_10m','Windgusts_10m','Winddirection_10m']
        wx_raw = np.array([float(row.get(c, row.get(c.lower(), 0)) or 0)
                           for c in wx_cols], dtype=np.float32)
        try:
            wx_sc = (self.weather_scaler.transform(wx_raw.reshape(1,-1))[0].astype(np.float32)
                     if self.weather_scaler is not None else wx_raw)
        except Exception:
            wx_sc = wx_raw

        arr_raw = float(row.get('arrivalDelay', 0) or 0)
        dep_raw = float(row.get('departureDelay', row.get('arrivalDelay', 0)) or 0)
        context_flags = torch.FloatTensor([
            float(row.get('Is_weekend',   row.get('is_weekend',   0)) or 0),
            float(row.get('is_peak_hour', 0) or 0),
            float(row.get('has_prev_stop', 0) or 0),
            float(abs(arr_raw) > 20 or abs(dep_raw) > 20),
        ])

        operational = torch.FloatTensor(op_vec)
        weather     = torch.FloatTensor(wx_sc)

        target    = torch.FloatTensor([float(row['duration_scaled'])])
        is_travel = int(row.get('is_travel', 1))

        return (seg_type_idx, temporal,
                context_flags, operational, operational,
                weather, weather,
                target, seq_len, is_travel)


class PathDataset(Dataset):
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
            all_speeds, all_operational, all_weather = [], [], []
            for idx, row in self.paths_df.iterrows():
                all_speeds.extend(row['speeds'])
                for i in range(row['seq_len']):
                    all_operational.append([
                        row['arrival_delays'][i],
                        row['departure_delays'][i],
                        row.get('is_weekend_flags', [0] * row['seq_len'])[i],
                        row.get('is_peak_hour_flags', [0] * row['seq_len'])[i]
                    ])
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
        seg_indices = [self.seg_to_idx.get(sid, 0) for sid in row['segment_ids']]
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
            temporal_features.append([hour_sin, hour_cos, dow_sin, dow_cos, speed_scaled])

        operational_data = []
        weather_data_list = []
        for i in range(seq_len):
            op = [row['arrival_delays'][i], row['departure_delays'][i],
                  row.get('is_weekend_flags', [0]*seq_len)[i],
                  row.get('is_peak_hour_flags', [0]*seq_len)[i]]
            wx = [row['temperatures'][i], row['apparent_temps'][i],
                  row.get('precipitations', [0]*seq_len)[i],
                  row.get('rains', [0]*seq_len)[i],
                  row.get('snowfalls', [0]*seq_len)[i],
                  row['windspeeds'][i], row['windgusts'][i], row['wind_directions'][i]]
            operational_data.append(op)
            weather_data_list.append(wx)

        operational_scaled = self.operational_scaler.transform(np.array(operational_data))
        weather_scaled = self.weather_scaler.transform(np.array(weather_data_list))
        target_scaled = float(self.target_scaler.transform([[row['total_duration']]])[0, 0])

        while len(seg_indices) < self.max_path_length:
            seg_indices.append(0)
            temporal_features.append([0, 0, 0, 0, 0])
            operational_scaled = np.vstack([operational_scaled, [0, 0, 0, 0]])
            weather_scaled = np.vstack([weather_scaled, [0, 0, 0, 0, 0, 0, 0, 0]])

        return (torch.LongTensor(seg_indices[:self.max_path_length]),
                torch.FloatTensor(temporal_features[:self.max_path_length]),
                torch.FloatTensor(operational_scaled[:self.max_path_length]),
                torch.FloatTensor(weather_scaled[:self.max_path_length]),
                torch.FloatTensor([target_scaled]),
                seq_len)


class TripDataset(Dataset):
    """
    Groups segments by trip_id+day into chronologically-ordered sequences.

    v3 change: segments are now sorted by departure_time (ISO string column)
    if it exists, otherwise by hour.  This is critical for the LSTM's
    cumulative correction architecture.
    """

    MAX_TRIP_LEN = 15

    def __init__(self,
                 segments_df,
                 segment_types,
                 max_trip_length: int = 15,
                 fit_scalers: bool = True,
                 target_scaler: RobustScaler = None,
                 speed_scaler: RobustScaler = None,
                 operational_scaler: RobustScaler = None,
                 weather_scaler: RobustScaler = None,
                 trip_target_scaler: RobustScaler = None,
                 day_target_scaler: RobustScaler = None):

        self.max_trip_length  = max_trip_length
        self.segment_types    = list(segment_types)
        self.seg_to_idx       = {s: i for i, s in enumerate(self.segment_types)}

        df = segments_df.copy()

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin']  = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos']  = np.cos(2 * np.pi * df['day_of_week'] / 7)

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

        if fit_scalers:
            self.target_scaler = RobustScaler()
            df['duration_scaled'] = self.target_scaler.fit_transform(
                df[['duration_sec']]).flatten()
        else:
            self.target_scaler = target_scaler
            df['duration_scaled'] = self.target_scaler.transform(
                df[['duration_sec']]).flatten()

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

        for c in ['is_weekend', 'is_peak_hour']:
            if c not in df.columns:
                df[c] = 0.0

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

        df['seg_type_idx'] = df['segment_id'].map(self.seg_to_idx)
        n_before = len(df)
        df = df.dropna(subset=['seg_type_idx']).copy()
        df['seg_type_idx'] = df['seg_type_idx'].astype(int)
        if len(df) < n_before:
            print(f"   TripDataset: dropped {n_before - len(df):,} rows "
                  f"with unseen segment types")

        if 'departure_time' in df.columns:
            df['_dep_dt'] = pd.to_datetime(df['departure_time'], errors='coerce')
            df['_date']   = df['_dep_dt'].dt.normalize()
        else:
            df['_dep_dt'] = pd.NaT
            df['_date']   = pd.NaT

        def _sort_group(grp):
            sort_keys = []
            asc_flags = []
            if '_dep_dt' in grp.columns and grp['_dep_dt'].notna().any():
                sort_keys.append('_dep_dt')
                asc_flags.append(True)
            if 'trip_stop_sequence' in grp.columns and grp['trip_stop_sequence'].notna().any():
                sort_keys.append('trip_stop_sequence')
                asc_flags.append(True)
            if not sort_keys:
                sort_keys.append('hour')
                asc_flags.append(True)
            return grp.sort_values(sort_keys, ascending=asc_flags)

        group_cols = ['trip_id', '_date']
        trips = []
        for _, grp in df.groupby(group_cols, sort=True, dropna=False):
            grp = _sort_group(grp).reset_index(drop=True)
            if len(grp) < 2:
                continue
            if len(grp) > max_trip_length:
                grp = grp.head(max_trip_length)
            trips.append(grp)

        self.trips = trips
        sort_desc = 'departure_time (primary) + trip_stop_sequence (tiebreaker)' \
                    if 'departure_time' in df.columns else 'hour'
        print(f"   TripDataset: {len(trips):,} trips built from "
              f"{len(df):,} segments  (sorted by {sort_desc})")
        if trips:
            lengths = [len(t) for t in trips]
            print(f"     Trip length: min={min(lengths)}, "
                  f"max={max(lengths)}, "
                  f"mean={np.mean(lengths):.1f}")


        def _trip_first_dep(trip):
            if '_dep_dt' in trip.columns and trip['_dep_dt'].notna().any():
                return trip['_dep_dt'].dropna().min()
            return pd.Timestamp.max

        def _trip_dur(trip):
            if 'total_travel_time_sec' in trip.columns:
                vals = trip['total_travel_time_sec'].dropna()
                if len(vals) > 0:
                    return float(vals.sum())
            return float(trip['duration_sec'].sum())

        date_to_trip_indices = {}
        for i, trip in enumerate(trips):
            if '_dep_dt' in trip.columns and trip['_dep_dt'].notna().any():
                d = trip['_dep_dt'].dropna().min().normalize()
            else:
                d = pd.Timestamp('1970-01-01')
            date_to_trip_indices.setdefault(d, []).append(i)

        self.day_trip_rank        = [0] * len(trips)
        self.cum_prior_actual_sec = [0.0] * len(trips)
        self.day_total_actual_sec = [0.0] * len(trips)

        for d, indices in date_to_trip_indices.items():
            indices_sorted = sorted(
                indices,
                key=lambda i: _trip_first_dep(trips[i])
            )
            day_durs = [_trip_dur(trips[i]) for i in indices_sorted]
            day_total = sum(day_durs)
            cum = 0.0
            for rank, (idx, dur) in enumerate(zip(indices_sorted, day_durs)):
                self.day_trip_rank[idx]        = rank
                self.cum_prior_actual_sec[idx] = cum
                self.day_total_actual_sec[idx] = day_total
                cum += dur

        print(f"   Super-global: {len(date_to_trip_indices)} service days  "
              f"day_total range: "
              f"{min(self.day_total_actual_sec):.0f}–"
              f"{max(self.day_total_actual_sec):.0f}s")

        def _trip_total(trip):
            if 'total_travel_time_sec' in trip.columns:
                vals = trip['total_travel_time_sec'].dropna()
                if len(vals) > 0:
                    return float(vals.sum())
            return float(trip['duration_sec'].sum())

        if fit_scalers:
            all_trip_totals = [[_trip_total(t)] for t in trips]
            self.trip_target_scaler = RobustScaler()
            self.trip_target_scaler.fit(all_trip_totals)

            all_day_totals = [[v] for v in self.day_total_actual_sec]
            self.day_target_scaler = RobustScaler()
            self.day_target_scaler.fit(all_day_totals)

            self.cum_prior_scaler = self.trip_target_scaler

            print(f"   trip_target_scaler : center={self.trip_target_scaler.center_[0]:.1f}s  "
                  f"scale={self.trip_target_scaler.scale_[0]:.1f}s")
            print(f"   day_target_scaler  : center={self.day_target_scaler.center_[0]:.1f}s  "
                  f"scale={self.day_target_scaler.scale_[0]:.1f}s")
        else:
            self.trip_target_scaler = (trip_target_scaler
                                       if trip_target_scaler is not None
                                       else self.target_scaler)
            self.day_target_scaler  = (day_target_scaler
                                       if day_target_scaler is not None
                                       else self.trip_target_scaler)
            self.cum_prior_scaler   = self.trip_target_scaler

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
        context_flags      = torch.zeros(T, _CONTEXT_DIM)
        origin_operational = torch.zeros(T, 3)
        dest_operational   = torch.zeros(T, 3)
        origin_weather     = torch.zeros(T, 8)
        dest_weather       = torch.zeros(T, 8)
        seg_targets        = torch.zeros(T, 1)

        wx_cols = ['Temperature_2m','Apparent_temperature','Precipitation',
                   'Rain','Snowfall','Windspeed_10m','Windgusts_10m','Winddirection_10m']

        for t, (_, row) in enumerate(trip.iterrows()):
            seg_indices[t] = int(row['seg_type_idx'])
            temporal[t] = torch.tensor([
                float(row['hour_sin']), float(row['hour_cos']),
                float(row['dow_sin']),  float(row['dow_cos']),
                float(row['speed_scaled']),
            ])

            arr_raw = float(row.get('arrivalDelay', 0) or 0)
            dep_raw = float(row.get('departureDelay', arr_raw) or 0)
            dwell   = float(row.get('dwellTime_sec', 0) or 0)
            op_raw  = np.array([arr_raw, dep_raw, dwell], dtype=np.float32)
            try:
                op_sc = self.operational_scaler.transform(op_raw[:2].reshape(1,-1))[0]
                op_vec = np.append(op_sc, dwell).astype(np.float32)
            except Exception:
                op_vec = op_raw

            wx_raw = np.array([float(row.get(c, row.get(c.lower(), 0)) or 0)
                               for c in wx_cols], dtype=np.float32)
            try:
                wx_sc = (self.weather_scaler.transform(wx_raw.reshape(1,-1))[0].astype(np.float32)
                         if self.weather_scaler is not None else wx_raw)
            except Exception:
                wx_sc = wx_raw

            context_flags[t] = torch.FloatTensor([
                float(row.get('Is_weekend',   row.get('is_weekend',   0)) or 0),
                float(row.get('is_peak_hour', 0) or 0),
                float(t > 0),
                float(abs(arr_raw) > 20 or abs(dep_raw) > 20),
            ])

            origin_operational[t] = torch.FloatTensor(op_vec)
            dest_operational[t]   = torch.FloatTensor(op_vec)
            origin_weather[t]     = torch.FloatTensor(wx_sc)
            dest_weather[t]       = torch.FloatTensor(wx_sc)

            seg_targets[t, 0] = float(row['duration_scaled'])

        if 'total_travel_time_sec' in trip.columns:
            vals = trip['total_travel_time_sec'].dropna()
            total_sec = float(vals.sum()) if len(vals) > 0 else float(trip['duration_sec'].sum())
        else:
            total_sec = float(trip['duration_sec'].sum())
        trip_target = torch.FloatTensor(
            self.trip_target_scaler.transform([[total_sec]])[0])

        cum_prior_sec  = self.cum_prior_actual_sec[idx]
        day_total_sec  = self.day_total_actual_sec[idx]
        day_trip_rank  = self.day_trip_rank[idx]

        cum_prior_scaled = torch.FloatTensor(
            self.cum_prior_scaler.transform([[cum_prior_sec]])[0])
        day_target = torch.FloatTensor(
            self.day_target_scaler.transform([[day_total_sec]])[0])
        day_rank_t = torch.FloatTensor([float(day_trip_rank)])

        return (seg_indices, temporal,
                context_flags, origin_operational, dest_operational,
                origin_weather, dest_weather,
                seg_targets, trip_target,
                cum_prior_scaled, day_target, day_rank_t,
                seq_len)


PrepTripDataset = TripDataset


def prep_trip_collate_fn(batch):
    """Alias of trip_collate_fn — PrepTripDataset is now TripDataset."""
    return trip_collate_fn(batch)


def trip_collate_fn(batch):
    seg_indices        = torch.stack([b[0] for b in batch])
    temporal           = torch.stack([b[1] for b in batch])
    context_flags      = torch.stack([b[2] for b in batch])
    origin_operational = torch.stack([b[3] for b in batch])
    dest_operational   = torch.stack([b[4] for b in batch])
    origin_weather     = torch.stack([b[5] for b in batch])
    dest_weather       = torch.stack([b[6] for b in batch])
    seg_targets        = torch.stack([b[7] for b in batch])
    trip_targets       = torch.stack([b[8] for b in batch])
    cum_prior_scaled   = torch.stack([b[9]  for b in batch])
    day_targets        = torch.stack([b[10] for b in batch])
    day_ranks          = torch.stack([b[11] for b in batch])
    lengths            = torch.LongTensor([b[12] for b in batch])

    T    = seg_indices.size(1)
    mask = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)

    return (seg_indices, temporal,
            context_flags, origin_operational, dest_operational,
            origin_weather, dest_weather,
            seg_targets, trip_targets,
            cum_prior_scaled, day_targets, day_ranks,
            lengths, mask)


def path_collate_fn(batch):
    """
    Collate function for PathDataset.
    Each item: (seg_indices, temporal, operational, weather, target, seq_len)
    """
    seg_indices = torch.stack([b[0] for b in batch])
    temporal    = torch.stack([b[1] for b in batch])
    operational = torch.stack([b[2] for b in batch])
    weather     = torch.stack([b[3] for b in batch])
    targets     = torch.stack([b[4] for b in batch])
    lengths     = torch.LongTensor([b[5] for b in batch])

    T    = seg_indices.size(1)
    mask = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)

    return seg_indices, temporal, operational, weather, targets, lengths, mask


def masked_collate_fn(batch):
    seg_indices   = torch.LongTensor([b[0] for b in batch])
    temporals     = torch.stack([b[1].unsqueeze(0) for b in batch])
    targets       = torch.stack([b[2] for b in batch])
    seq_lens      = torch.LongTensor([b[3] for b in batch])
    max_len       = seq_lens.max().item()
    mask          = torch.arange(max_len).unsqueeze(0) < seq_lens.unsqueeze(1)
    return seg_indices, temporals, targets, seq_lens, mask


def enhanced_collate_fn(batch):
    seg_indices        = torch.LongTensor([b[0]          for b in batch])
    temporal           = torch.stack([b[1].unsqueeze(0)  for b in batch])
    context_flags      = torch.stack([b[2].unsqueeze(0)  for b in batch])
    origin_operational = torch.stack([b[3].unsqueeze(0)  for b in batch])
    dest_operational   = torch.stack([b[4].unsqueeze(0)  for b in batch])
    origin_weather     = torch.stack([b[5].unsqueeze(0)  for b in batch])
    dest_weather       = torch.stack([b[6].unsqueeze(0)  for b in batch])
    targets            = torch.stack([b[7]               for b in batch])
    seq_lens           = torch.LongTensor([b[8]          for b in batch])
    is_travel_flags    = torch.LongTensor([b[9]          for b in batch])
    max_len            = seq_lens.max().item()
    mask               = torch.arange(max_len).unsqueeze(0) < seq_lens.unsqueeze(1)
    return (seg_indices, temporal, context_flags,
            origin_operational, dest_operational,
            origin_weather, dest_weather,
            targets, seq_lens, mask, is_travel_flags)


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

        self.register_buffer('adj_geo',  None)
        self.register_buffer('adj_dist', None)
        self.register_buffer('adj_soc',  None)

    def set_adjacency_matrices(self, adj_geo, adj_dist, adj_soc):
        self.register_buffer('adj_geo',  torch.FloatTensor(adj_geo))
        self.register_buffer('adj_dist', torch.FloatTensor(adj_dist))
        self.register_buffer('adj_soc',  torch.FloatTensor(adj_soc))

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

    def _get_spatial_embeddings(self, seg_indices):
        """Return per-segment spatial embeddings (for MTL repr extraction)."""
        all_nodes = self.node_embedding.weight.unsqueeze(0)
        spatial_all = self.multi_gat(all_nodes, [self.adj_geo, self.adj_dist, self.adj_soc])
        return spatial_all.squeeze(0)[seg_indices]


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
        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        out = self.layer_norm(torch.matmul(attn_weights, V) + x)
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
        self.global_attention = GlobalTemporalAttention(hidden_dim, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout),
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
        return self.fusion(attn_out[:, -1, :])


class MAGNN_LSTM(nn.Module):
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
            operational_dim=operational_dim,
            weather_dim=weather_dim,
            temporal_dim=temporal_dim,
            hidden_dim=lstm_hidden,
            n_layers=lstm_layers,
            dropout=dropout,
        )

    def _get_spatial_embeddings(self, seg_indices):
        return self.magnn._get_spatial_embeddings(seg_indices)

    def forward(self, seg_indices, temporal_features,
                operational_features=None, weather_features=None):
        spatial_emb = self._get_spatial_embeddings(seg_indices)
        if operational_features is None:
            operational_features = torch.zeros(
                seg_indices.shape[0], 3, device=seg_indices.device)
        if weather_features is None:
            weather_features = torch.zeros(
                seg_indices.shape[0], 8, device=seg_indices.device)
        lstm_input = torch.cat(
            [spatial_emb, operational_features, weather_features, temporal_features], dim=1)
        return self.lstm_model(lstm_input)


class SimpleMLP(nn.Module):
    def __init__(self, num_segments, embed_dim=32, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(num_segments, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + 5, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, seg_idx, temporal):
        emb = self.embedding(seg_idx)
        x = torch.cat([emb, temporal], dim=1)
        return self.mlp(x)


class DwellPredictor(nn.Module):
    def __init__(self, num_segments, embed_dim=32, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(num_segments, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim + _CONTEXT_DIM + 5, 64),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, seg_idx, temporal, context_flags):
        emb = self.embedding(seg_idx)
        x = torch.cat([emb, temporal, context_flags], dim=1)
        return self.head(x)

def evaluate(model, dataloader, criterion, device, scaler):
    """Evaluate model with detailed inference timing metrics."""
    import time
    model.eval()
    predictions_list = []
    targets_list = []
    total_loss = 0.0
    n_batches = 0
    model_forward_times = []
    total_inference_start = time.time()
    with torch.no_grad():
        for batch in dataloader:
            seg_idx, temporal_pad, target, lengths, mask = batch
            temporal = temporal_pad[:, 0, :]
            seg_idx = seg_idx.to(device)
            temporal = temporal.to(device)
            target = target.to(device)
            if seg_idx is None:
                continue
            forward_start = time.time()
            predictions = model(seg_idx, temporal)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            model_forward_times.append(time.time() - forward_start)
            loss = criterion(predictions, target)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            total_loss += loss.item()
            n_batches += 1
            predictions_list.append(predictions.cpu().numpy())
            targets_list.append(target.cpu().numpy())
    total_inference_time = time.time() - total_inference_start
    if not predictions_list:
        return {'loss': float('nan'), 'r2': float('nan'),
                'rmse': float('nan'), 'mae': float('nan'),
                'mape': float('nan'), 'preds': [], 'actual': [],
                'inference_timing': {}}
    preds = np.concatenate(predictions_list)
    targets = np.concatenate(targets_list)
    preds_orig = scaler.inverse_transform(preds)
    targets_orig = scaler.inverse_transform(targets)
    r2   = r2_score(targets_orig, preds_orig)
    rmse = float(np.sqrt(mean_squared_error(targets_orig, preds_orig)))
    mae  = float(mean_absolute_error(targets_orig, preds_orig))
    mask = targets_orig.flatten() > 0
    mape = (np.mean(np.abs((targets_orig.flatten()[mask] - preds_orig.flatten()[mask]) /
                           targets_orig.flatten()[mask])) * 100 if mask.any() else float('nan'))
    num_samples = len(preds)
    return {
        'loss': total_loss / max(n_batches, 1),
        'r2': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'preds': preds_orig.flatten().tolist(),
        'actual': targets_orig.flatten().tolist(),
        'inference_timing': {
            'avg_model_forward_ms': np.mean(model_forward_times) * 1000 if model_forward_times else 0,
            'total_inference_time_s': total_inference_time,
            'throughput_samples_per_sec': num_samples / total_inference_time if total_inference_time > 0 else 0,
            'avg_latency_per_sample_ms': (total_inference_time / num_samples) * 1000 if num_samples > 0 else 0,
        }
    }


def train_magtte(train_loader, val_loader, test_loader,
                 adj_geo, adj_dist, adj_soc,
                 segment_types, scaler,
                 output_folder, device, config):
    """Train the baseline MAGTTE model (unchanged)."""
    print_section("MAGTTE TRAINING")
    num_segments = len(segment_types)
    model = MAGTTE(
        num_nodes=num_segments,
        n_heads=config.n_heads,
        node_embed_dim=config.node_embed_dim,
        gat_hidden=config.gat_hidden,
        lstm_hidden=config.lstm_hidden,
        historical_dim=config.historical_dim,
        dropout=config.dropout,
    ).to(device)
    model.set_adjacency_matrices(adj_geo, adj_dist, adj_soc)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    criterion = nn.SmoothL1Loss()
    best_val_loss = float('inf')
    best_ckpt = os.path.join(output_folder, 'magtte_best.pth')

    for epoch in range(1, config.n_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            seg_idx, temporal_pad, target, lengths, mask = batch
            temporal = temporal_pad[:, 0, :]
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
                temporal = temporal_pad[:, 0, :]
                pred = model(seg_idx.to(device), temporal.to(device))
                val_loss += criterion(pred, target.to(device)).item()
        val_loss /= len(val_loader)

        if epoch % max(1, config.n_epochs // 10) == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{config.n_epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt)

    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    def _eval(loader, name, with_timing=False):
        if with_timing:
            res = evaluate(model, loader, criterion, device, scaler)
        else:
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for batch in loader:
                    seg_idx, temporal_pad, target, lengths, mask = batch
                    temporal = temporal_pad[:, 0, :]
                    pred = model(seg_idx.to(device), temporal.to(device))
                    preds.append(pred.cpu().numpy())
                    targets.append(target.numpy())
            if not preds:
                return {}
            p = scaler.inverse_transform(np.concatenate(preds))
            t = scaler.inverse_transform(np.concatenate(targets))
            r2   = r2_score(t, p)
            rmse = float(np.sqrt(mean_squared_error(t, p)))
            mae  = float(mean_absolute_error(t, p))
            mv   = t.flatten() > 0
            mape = (float(np.mean(np.abs((t.flatten()[mv] - p.flatten()[mv]) /
                                          t.flatten()[mv])) * 100)
                    if mv.any() else float('nan'))
            res = {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape,
                   'preds': p.flatten().tolist(), 'actual': t.flatten().tolist()}
        print(f"   {name:<6}  R²={res.get('r2', float('nan')):.4f}  "
              f"RMSE={res.get('rmse', float('nan')):.2f}s  "
              f"MAE={res.get('mae', float('nan')):.2f}s  "
              f"MAPE={res.get('mape', float('nan')):.2f}%")
        return res

    print_section("MAGTTE — FINAL RESULTS")
    results = {
        'Train': _eval(train_loader, 'Train'),
        'Val':   _eval(val_loader,   'Val'),
        'Test':  _eval(test_loader,  'Test', with_timing=True),
    }

    test_res = results.get('Test', {})
    if test_res.get('preds'):
        print(f"\n  {'Idx':>4}  {'Actual(s)':>10}  {'Pred(s)':>10}  "
              f"{'Error(s)':>9}  {'Error%':>7}")
        print("  " + "-" * 48)
        for i in range(min(20, len(test_res['actual']))):
            a, pv = test_res['actual'][i], test_res['preds'][i]
            err  = pv - a
            epct = (err / a * 100) if a != 0 else 0.0
            print(f"  {i:>3}  {a:>10.2f}  {pv:>10.2f}  "
                  f"{err:>9.2f}  {epct:>6.2f}%")

    return results, model


def train_simple(train_loader, val_loader, test_loader, segment_types, scaler,
                 output_folder, device, n_epochs=50, lr=0.001, dropout=0.3):
    print_section("SIMPLE MLP TRAINING")
    num_segments = len(segment_types)
    model = SimpleMLP(num_segments, embed_dim=32, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.SmoothL1Loss()
    best_val_loss = float('inf')

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            seg_idx, temporal_pad, target, lengths, mask = batch
            temporal = temporal_pad.squeeze(1)
            pred = model(seg_idx.to(device), temporal.to(device))
            loss = criterion(pred, target.to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
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
            print(f"  Epoch {epoch:>3}/{n_epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_folder, 'simple_best.pth'))

    return {}, model


def train_magnn_lstm_dualtask_mtl(
    train_loader, val_loader, test_loader,
    segment_types, scaler, output_folder, device, cfg,
    pretrained_residual_model=None,
    trip_scaler=None,
    use_prep_dataset: bool = False,
):
    """
    Train MAGNN_LSTM_DualTaskMTL.

    LOCAL  task : predict each segment's duration (fine-grained signal).
    GLOBAL task : predict total trip duration = Σ segments.
    CONSISTENCY : enforce Σ(local_preds) ≈ global_pred.

    use_prep_dataset=True: loaders come from PrepTripDataset / prep_trip_collate_fn
      → batches have 12 elements including magnn_baselines tensor.
    use_prep_dataset=False: loaders come from TripDataset / trip_collate_fn (legacy).
    """
    print_section("MAGNN-LSTM-DUALTASK-MTL TRAINING (v3)")

    if pretrained_residual_model is None:
        raise ValueError("pretrained_residual_model is required")

    _lstm_hidden = getattr(
        getattr(pretrained_residual_model, 'residual_lstm', None), 'hidden_dim', 128)

    model = MAGNN_LSTM_DualTaskMTL(
        residual_model=pretrained_residual_model,
        spatial_dim=cfg.gat_hidden,
        lstm_hidden=_lstm_hidden,
        enc_hidden=getattr(cfg, 'enc_hidden', 64),
        local_hidden=getattr(cfg, 'enc_hidden', 64),
        global_hidden=getattr(cfg, 'enc_hidden', 64),
        dropout=getattr(cfg, 'mtl_dropout', 0.3),
        lambda_cons=getattr(cfg, 'lambda_cons', 0.1),
    ).to(device)

    criterion = model.criterion

    trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
    lr = getattr(cfg, 'residual_learning_rate', 5e-4)
    wd = getattr(cfg, 'lstm_weight_decay', 1e-6)
    optimizer = optim.Adam(trainable, lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8)

    n_epochs   = getattr(cfg, 'n_epochs', 100)
    early_stop = getattr(cfg, 'early_stopping_patience', 15)
    best_ckpt  = os.path.join(output_folder, 'dualtask_mtl_best.pth')

    best_val_loss    = float('inf')
    patience_counter = 0

    def _unpack(batch, dev):
        """Unpack 11- or 12-element batch."""
        (seg_idx, temporal, ctx, o_op, d_op,
         o_wx, d_wx, seg_tgt, trip_tgt, lengths, mask) = batch[:11]
        return (seg_idx.to(dev), temporal.to(dev),
                ctx.to(dev), o_op.to(dev), d_op.to(dev),
                o_wx.to(dev), d_wx.to(dev),
                seg_tgt.to(dev), trip_tgt.to(dev),
                lengths.to(dev), mask.to(dev))

    _ts = trip_scaler or scaler

    print(f"   Epochs={n_epochs}  patience={early_stop}  lr={lr}  "
          f"PrepDataset={'yes' if use_prep_dataset else 'no'}")
    print()

    for epoch in range(1, n_epochs + 1):
        model.train()
        tr_total = tr_local = tr_global = tr_cons = 0.0
        n_batches = 0

        for batch in train_loader:
            (seg_idx, temporal, ctx, o_op, d_op,
             o_wx, d_wx, seg_tgt, trip_tgt,
             lengths, mask) = _unpack(batch, device)

            local_preds, global_pred = model(
                seg_idx, temporal, ctx, o_op, d_op,
                o_wx, d_wx, mask, lengths, return_local=True,
                seg_targets=seg_tgt)

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
            tr_cons   += ld.get('consistency', 0.0)
            n_batches += 1

        tr_total  /= max(n_batches, 1)
        tr_local  /= max(n_batches, 1)
        tr_global /= max(n_batches, 1)
        tr_cons   /= max(n_batches, 1)

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
                    o_wx, d_wx, mask, lengths, return_local=True,
                    seg_targets=None)

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

        val_r2 = float('nan')
        if val_preds:
            vp = _ts.inverse_transform(np.concatenate(val_preds))
            vt = _ts.inverse_transform(np.concatenate(val_targets))
            try:
                val_r2 = r2_score(vt, vp)
            except Exception:
                pass

        if epoch % max(1, n_epochs // 5) == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{n_epochs}  "
                  f"loss={tr_total:.4f} "
                  f"(L:{tr_local:.4f} G:{tr_global:.4f} C:{tr_cons:.4f})  "
                  f"val={val_loss:.4f}  val_R²={val_r2:.4f}")

        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= early_stop:
                print(f"\n  ⏹️  Early stopping at epoch {epoch}")
                break

    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    print_section("MAGNN-LSTM-DUALTASK-MTL — FINAL RESULTS")

    def _eval(loader, name):
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in loader:
                (seg_idx, temporal, ctx, o_op, d_op,
                 o_wx, d_wx, seg_tgt, trip_tgt,
                 lengths, mask) = _unpack(batch, device)

                local_preds, global_pred = model(
                    seg_idx, temporal, ctx, o_op, d_op,
                    o_wx, d_wx, mask, lengths, return_local=False,
                    seg_targets=None)

                preds.append(global_pred.cpu().numpy())
                targets.append(trip_tgt.cpu().numpy())

        if not preds:
            return {}

        p = _ts.inverse_transform(np.concatenate(preds))
        t = _ts.inverse_transform(np.concatenate(targets))
        r2   = r2_score(t, p)
        rmse = float(np.sqrt(mean_squared_error(t, p)))
        mae  = float(mean_absolute_error(t, p))
        mv   = t.flatten() > 0
        mape = (float(np.mean(np.abs((t.flatten()[mv] - p.flatten()[mv]) /
                                      t.flatten()[mv])) * 100)
                if mv.any() else float('nan'))
        print(f"   {name:<6}  R²={r2:.4f}  RMSE={rmse:.2f}s  "
              f"MAE={mae:.2f}s  MAPE={mape:.2f}%")
        return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape,
                'preds': p.flatten().tolist(), 'actual': t.flatten().tolist()}

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
            a, pv = test_res['actual'][i], test_res['preds'][i]
            err  = pv - a
            epct = (err / a * 100) if a > 0 else 0.0
            print(f"  {i:>3}  {a:>10.2f}  {pv:>10.2f}  "
                  f"{err:>9.2f}  {epct:>6.2f}%")

    return results, model


def train_trip_level_predictor(train_loader, val_loader, test_loader,
                                segment_types, trip_scaler,
                                output_folder, device, cfg,
                                pretrained_residual_model=None,
                                seg_scaler=None,
                                day_scaler=None,
                                n_epochs_override=None):
    print_section("HIERARCHICAL TRIP PREDICTOR TRAINING")

    if pretrained_residual_model is None:
        raise ValueError("pretrained_residual_model is required")

    _lstm_hidden = getattr(
        getattr(pretrained_residual_model, 'residual_lstm', None), 'hidden_dim', 128)

    model = HierarchicalTripPredictor(
        residual_model=pretrained_residual_model,
        spatial_dim=cfg.gat_hidden,
        lstm_hidden=_lstm_hidden,
        enc_hidden=128,
        dropout=0.2,
        lambda_local=0.3,
    ).to(device)

    lr = getattr(cfg, 'residual_learning_rate', 5e-4)
    wd = getattr(cfg, 'lstm_weight_decay', 1e-6)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8)

    criterion_global = nn.SmoothL1Loss()
    criterion_local  = nn.SmoothL1Loss(reduction='none')
    criterion_day    = nn.SmoothL1Loss()

    n_epochs   = n_epochs_override or max(getattr(cfg, 'n_epochs', 50), 100)
    early_stop = max(getattr(cfg, 'early_stopping_patience', 10), 15)
    best_ckpt  = os.path.join(output_folder, 'hierarchical_trip_best.pth')

    best_val_loss    = float('inf')
    patience_counter = 0

    def _unpack(batch, dev):
        (seg_idx, temporal, ctx, o_op, d_op,
         o_wx, d_wx, seg_tgt, trip_tgt,
         cum_prior, day_tgt, day_rank,
         lengths, mask) = batch[:14]
        return (seg_idx.to(dev), temporal.to(dev),
                ctx.to(dev), o_op.to(dev), d_op.to(dev),
                o_wx.to(dev), d_wx.to(dev),
                seg_tgt.to(dev), trip_tgt.to(dev),
                cum_prior.to(dev), day_tgt.to(dev), day_rank.to(dev),
                lengths.to(dev), mask.to(dev))

    for epoch in range(1, n_epochs + 1):
        model.train()
        tr_global = tr_local = tr_day = tr_bias_mag = 0.0
        n_batches = 0

        for batch in train_loader:
            (seg_idx, temporal, ctx, o_op, d_op,
             o_wx, d_wx, seg_tgt, trip_tgt,
             cum_prior, day_tgt, day_rank,
             lengths, mask) = _unpack(batch, device)

            global_pred, local_preds, day_pred = model(
                seg_idx, temporal, ctx, o_op, d_op,
                o_wx, d_wx, mask, lengths,
                return_local=True,
                seg_targets=seg_tgt,
                cum_prior_actual=cum_prior)

            m      = mask.unsqueeze(-1).float()
            loss_l = (criterion_local(local_preds, seg_tgt) * m).sum() / m.sum().clamp(min=1)
            loss_g = criterion_global(global_pred, trip_tgt)
            loss_d = criterion_day(day_pred, day_tgt)

            bias_reg = (model._last_trip_bias.abs().mean()
                        if hasattr(model, '_last_trip_bias')
                        else torch.tensor(0., device=device))

            loss = (loss_g
                    + model.lambda_local * loss_l
                    + model.lambda_day   * loss_d
                    + model.lambda_bias  * bias_reg)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_global    += loss_g.item()
            tr_local     += loss_l.item()
            tr_day       += loss_d.item()
            tr_bias_mag  += bias_reg.item() if hasattr(bias_reg, 'item') else float(bias_reg)
            n_batches    += 1

        tr_global   /= max(n_batches, 1)
        tr_local    /= max(n_batches, 1)
        tr_day      /= max(n_batches, 1)
        tr_bias_mag /= max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                (seg_idx, temporal, ctx, o_op, d_op,
                 o_wx, d_wx, seg_tgt, trip_tgt,
                 cum_prior, day_tgt, day_rank,
                 lengths, mask) = _unpack(batch, device)

                global_pred, _, day_pred = model(
                    seg_idx, temporal, ctx, o_op, d_op,
                    o_wx, d_wx, mask, lengths,
                    return_local=False,
                    seg_targets=None,
                    cum_prior_actual=cum_prior)

                loss = criterion_global(global_pred, trip_tgt) + \
                       model.lambda_day * criterion_day(day_pred, day_tgt)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
                    n_val    += 1
                val_preds.append(global_pred.cpu().numpy())
                val_targets.append(trip_tgt.cpu().numpy())

        val_loss /= max(n_val, 1)
        scheduler.step(val_loss)

        val_r2 = float('nan')
        if val_preds and trip_scaler is not None:
            try:
                vp = trip_scaler.inverse_transform(np.concatenate(val_preds))
                vt = trip_scaler.inverse_transform(np.concatenate(val_targets))
                val_r2 = r2_score(vt, vp)
            except Exception:
                pass

        if epoch % max(1, n_epochs // 10) == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{n_epochs}  "
                  f"global={tr_global:.4f}  local={tr_local:.4f}  "
                  f"day={tr_day:.4f}  bias={tr_bias_mag:.4f}  "
                  f"val={val_loss:.4f}  val_R²={val_r2:.4f}")

        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= early_stop:
                print(f"\n  ⏹️  Early stopping at epoch {epoch}")
                break

    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    print_section("HIERARCHICAL TRIP PREDICTOR — FINAL RESULTS")

    def _eval(loader, name):
        model.eval()
        preds, targets, naive_sums, bias_mags = [], [], [], []
        day_preds_l, day_targets_l = [], []

        with torch.no_grad():
            for batch in loader:
                (seg_idx, temporal, ctx, o_op, d_op,
                 o_wx, d_wx, seg_tgt, trip_tgt,
                 cum_prior, day_tgt, day_rank,
                 lengths, mask) = _unpack(batch, device)

                global_pred, local_preds, day_pred = model(
                    seg_idx, temporal, ctx, o_op, d_op,
                    o_wx, d_wx, mask, lengths,
                    return_local=True,
                    seg_targets=None,
                    cum_prior_actual=cum_prior)

                naive_sums.append(model._last_naive_sum.cpu().numpy())
                bias_mags.append(model._last_trip_bias.abs().cpu().numpy())
                preds.append(global_pred.cpu().numpy())
                targets.append(trip_tgt.cpu().numpy())
                day_preds_l.append(day_pred.cpu().numpy())
                day_targets_l.append(day_tgt.cpu().numpy())

        if not preds:
            return {}

        _ttrip = trip_scaler if trip_scaler is not None else seg_scaler
        _tseg  = seg_scaler  if seg_scaler  is not None else trip_scaler
        _tday  = day_scaler  if day_scaler  is not None else trip_scaler

        p  = _ttrip.inverse_transform(np.concatenate(preds))
        t  = _ttrip.inverse_transform(np.concatenate(targets))
        ns = _tseg.inverse_transform(np.concatenate(naive_sums))
        dp = _tday.inverse_transform(np.concatenate(day_preds_l))
        dt = _tday.inverse_transform(np.concatenate(day_targets_l))

        avg_bias = float(np.concatenate(bias_mags).mean())

        def _metrics(pred_s, true_s, label):
            r2   = r2_score(true_s, pred_s)
            rmse = float(np.sqrt(mean_squared_error(true_s, pred_s)))
            mae  = float(mean_absolute_error(true_s, pred_s))
            mv   = true_s.flatten() > 0
            mape = (float(np.mean(np.abs(
                (true_s.flatten()[mv] - pred_s.flatten()[mv]) /
                true_s.flatten()[mv])) * 100) if mv.any() else float('nan'))
            print(f"   {label:<44}  R²={r2:.4f}  RMSE={rmse:.2f}s  "
                  f"MAE={mae:.2f}s  MAPE={mape:.2f}%")
            return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape}

        print(f"\n   {name} set  (avg |trip_bias| = {avg_bias:.4f} scaled):")
        mtl_m   = _metrics(p,  t,  "LOCAL+GLOBAL  Σ(local)+bias  [per-trip]")
        naive_m = _metrics(ns, t,  "Naive  Σ(residual preds)     [per-trip]")
        day_m   = _metrics(dp, dt, "SUPER-GLOBAL  day head        [per-day] ")
        delta   = naive_m['rmse'] - mtl_m['rmse']
        print(f"   {'─'*70}")
        if delta > 0:
            print(f"   ✓ bias correction improves naive sum by {delta:.2f}s RMSE")
        else:
            print(f"   bias correction hurts by {-delta:.2f}s RMSE — bias not learned yet")

        return {'r2': mtl_m['r2'], 'rmse': mtl_m['rmse'],
                'mae': mtl_m['mae'], 'mape': mtl_m['mape'],
                'naive_rmse': naive_m['rmse'], 'naive_r2': naive_m['r2'],
                'day_r2': day_m['r2'], 'day_rmse': day_m['rmse'],
                'preds': p.flatten().tolist(), 'actual': t.flatten().tolist()}

    results = {
        'Train': _eval(train_loader, 'Train'),
        'Val':   _eval(val_loader,   'Val'),
        'Test':  _eval(test_loader,  'Test'),
    }
    return results, model


train_magnn_lstm_mtl = train_magnn_lstm_dualtask_mtl
