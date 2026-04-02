"""
segments.py - UPDATED
NOW WITH:
- is_weekend flag (1 if Saturday/Sunday, 0 otherwise)
- is_peak_hour flag (1 if 7-9 AM or 4-7 PM on weekdays, 0 otherwise)
- Operational delay + weather feature preservation
"""

import numpy as np
import pandas as pd
import os
import json
import warnings
from math import radians, cos, sin, asin, sqrt
from sklearn.neighbors import BallTree

from config import Config, DEVICE, print_section, haversine_meters

warnings.filterwarnings('ignore')

_known_stops_cache = {}

def is_weekend(day_of_week):
    return 1 if day_of_week >= 5 else 0


def is_peak_hour(hour, day_of_week):
    if day_of_week >= 5:  # Weekend
        return 0

    morning_peak = 7 <= hour < 9
    evening_peak = 16 <= hour < 19

    return 1 if (morning_peak or evening_peak) else 0

def build_segments_fixed(df, clusters):
    print_section("BUILDING SEGMENTS (WITH BINARY FLAGS)")

    if len(clusters) == 0:
        print("No clusters available")
        return pd.DataFrame()

    cluster_tree = BallTree(np.radians(clusters), metric='haversine')

    CLUSTER_ASSIGN_RADIUS_M = 75
    CLUSTER_ASSIGN_RADIUS_RAD = CLUSTER_ASSIGN_RADIUS_M / 6371000

    valid_gps = df['is_gps_valid'] == 1
    coords = df.loc[valid_gps, ['latitude', 'longitude']].values

    if len(coords) == 0:
        print("No valid GPS coordinates")
        return pd.DataFrame()

    coords_rad = np.radians(coords)
    distances, indices = cluster_tree.query(coords_rad, k=1)

    distances = distances.flatten()
    indices = indices.flatten()
    within_radius = distances <= CLUSTER_ASSIGN_RADIUS_RAD

    assigned_ids = np.where(within_radius, indices, -1)
    df.loc[valid_gps, 'cluster_id'] = assigned_ids
    df['cluster_id'] = df['cluster_id'].fillna(-1).astype(int)

    n_assigned = within_radius.sum()
    n_noise = (~within_radius).sum()
    print(f"   Assigned {n_assigned:,} points to {len(clusters)} clusters "
          f"(within {CLUSTER_ASSIGN_RADIUS_M}m)")
    print(f"   Between-station noise points: {n_noise:,} "
          f"({n_noise / len(coords) * 100:.1f}% — treated as in-transit)")

    segment_features = []
    operational_features = ['arrivalDelay', 'departureDelay']
    weather_features = ['temperature_2m', 'apparent_temperature', 'precipitation',
                        'rain', 'snowfall', 'windspeed_10m', 'windgusts_10m',
                        'winddirection_10m']

    available_operational = [f for f in operational_features if f in df.columns]
    available_weather = [f for f in weather_features if f in df.columns]

    print(f"   Preserving features from raw data:")
    print(f"     Operational: {available_operational}")
    print(f"     Weather: {available_weather}")
    print(f"     Binary flags: is_weekend, is_peak_hour")

    for trip_id, trip_df in df.groupby('trip_id'):
        trip_df = trip_df.sort_values('timestamp').reset_index(drop=True)

        i = 0
        while i < len(trip_df):
            if trip_df.loc[i, 'cluster_id'] == -1:
                i += 1
                continue

            origin_cluster = trip_df.loc[i, 'cluster_id']
            origin_start_idx = i

            last_origin_idx = i
            j = i + 1
            while j < len(trip_df):
                if trip_df.loc[j, 'cluster_id'] == origin_cluster:
                    last_origin_idx = j
                j += 1

            dest_found = False
            noise_count = 0
            j = last_origin_idx + 1

            while j < len(trip_df):
                current_cluster = trip_df.loc[j, 'cluster_id']

                if current_cluster == -1:
                    noise_count += 1
                    j += 1
                    continue

                if current_cluster != origin_cluster:
                    dest_cluster = current_cluster
                    dest_arrival_idx = j
                    dest_found = True
                    break

                j += 1

            if not dest_found:
                break

            departure_time = pd.to_datetime(trip_df.loc[last_origin_idx, 'timestamp'])
            arrival_time = pd.to_datetime(trip_df.loc[dest_arrival_idx, 'timestamp'])

            duration_sec = (arrival_time - departure_time).total_seconds()

            if duration_sec <= 5 or duration_sec > 3600:
                i = dest_arrival_idx
                continue

            distance_m = 0
            seg_df = trip_df.loc[last_origin_idx:dest_arrival_idx]

            if 'odometer' in seg_df.columns:
                odo_vals = seg_df['odometer'].dropna()
                if len(odo_vals) >= 2:
                    distance_m = odo_vals.max() - odo_vals.min()
                    if distance_m < 0 or distance_m > 50000:
                        distance_m = 0

            if distance_m == 0:
                distance_m = haversine_meters(
                    clusters[origin_cluster][0], clusters[origin_cluster][1],
                    clusters[dest_cluster][0], clusters[dest_cluster][1]
                )

            if distance_m < 50:
                i = dest_arrival_idx
                continue

            speed_mps = distance_m / duration_sec

            if speed_mps > 50:
                i = dest_arrival_idx
                continue
            hour = departure_time.hour
            day_of_week = departure_time.dayofweek
            weekend_flag = is_weekend(day_of_week)
            peak_flag = is_peak_hour(hour, day_of_week)
            avg_congestion = seg_df['congestionLevel'].mean() if 'congestionLevel' in seg_df.columns else 0
            operational_data = {}
            for feat in available_operational:
                feat_vals = seg_df[feat].dropna()
                if len(feat_vals) > 0:
                    operational_data[feat] = float(feat_vals.mean())
                else:
                    operational_data[feat] = 0.0

            weather_data = {}
            for feat in available_weather:
                feat_vals = seg_df[feat].dropna()
                if len(feat_vals) > 0:
                    weather_data[feat] = float(feat_vals.mean())
                else:
                    defaults = {
                        'temperature_2m': 15.0,
                        'apparent_temperature': 15.0,
                        'precipitation': 0.0,
                        'rain': 0.0,
                        'snowfall': 0.0,
                        'windspeed_10m': 5.0,
                        'windgusts_10m': 10.0,
                        'winddirection_10m': 180.0
                    }
                    weather_data[feat] = defaults.get(feat, 0.0)

            segment_dict = {
                'segment_id': f"{origin_cluster}_{dest_cluster}",
                'origin_cluster': origin_cluster,
                'dest_cluster': dest_cluster,
                'duration_sec': duration_sec,
                'distance_m': distance_m,
                'speed_mps': speed_mps,
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': weekend_flag,
                'is_peak_hour': peak_flag,
                'congestion': avg_congestion,
                'n_points': len(seg_df),
                'n_noise_points': noise_count,
                'trip_id': trip_id
            }

            segment_dict.update(operational_data)
            segment_dict.update(weather_data)

            segment_features.append(segment_dict)

            i = dest_arrival_idx

    segments_df = pd.DataFrame(segment_features)

    print(f"   Valid segments: {len(segments_df):,}")

    if len(segments_df) > 0:
        print(f"\n✓ Duration Statistics:")
        print(f"   Mean: {segments_df['duration_sec'].mean():.2f}s")
        print(f"   Median: {segments_df['duration_sec'].median():.2f}s")

        print(f"\n✓ Binary Flag Statistics:")
        print(
            f"   Weekend segments: {segments_df['is_weekend'].sum():,} ({segments_df['is_weekend'].mean() * 100:.1f}%)")
        print(
            f"   Peak hour segments: {segments_df['is_peak_hour'].sum():,} ({segments_df['is_peak_hour'].mean() * 100:.1f}%)")

        # Show delay statistics by flag
        if 'arrivalDelay' in segments_df.columns:
            print(f"\n✓ Delay Analysis by Context:")
            weekend_segs = segments_df[segments_df['is_weekend'] == 1]
            weekday_segs = segments_df[segments_df['is_weekend'] == 0]
            peak_segs = segments_df[segments_df['is_peak_hour'] == 1]
            offpeak_segs = segments_df[segments_df['is_peak_hour'] == 0]

            print(f"   Weekend avg delay: {weekend_segs['arrivalDelay'].mean():.2f}s")
            print(f"   Weekday avg delay: {weekday_segs['arrivalDelay'].mean():.2f}s")
            print(f"   Peak hour avg delay: {peak_segs['arrivalDelay'].mean():.2f}s")
            print(f"   Off-peak avg delay: {offpeak_segs['arrivalDelay'].mean():.2f}s")

        if available_operational:
            print(f"\n🔍 DEBUG: Operational features (RAW):")
            for feat in available_operational:
                if feat in segments_df.columns:
                    vals = segments_df[feat].values
                    print(f"   {feat}: min={vals.min():.4f}, max={vals.max():.4f}, "
                          f"mean={vals.mean():.4f}, std={vals.std():.4f}")

    return segments_df

def aggregate_segments_into_paths(segments_df, max_path_length=10):
    print_section("AGGREGATING SEGMENTS INTO PATHS")

    if 'trip_id' not in segments_df.columns:
        print("   ❌ No 'trip_id' column found")
        return segments_df

    paths = []

    for trip_id, trip_group in segments_df.groupby('trip_id'):
        trip_group = trip_group.head(max_path_length)
        seq_len = len(trip_group)

        if seq_len == 0:
            continue

        path = {
            'trip_id': trip_id,
            'seq_len': seq_len,
            'total_duration': trip_group['duration_sec'].sum(),

            'segment_ids': trip_group['segment_id'].tolist(),
            'segment_durations': trip_group['duration_sec'].tolist(),

            'hours': trip_group['hour'].tolist() if 'hour' in trip_group.columns else [0] * seq_len,
            'days_of_week': trip_group[
                'day_of_week'].tolist() if 'day_of_week' in trip_group.columns else [0] * seq_len,

            'is_weekend_flags': trip_group[
                'is_weekend'].tolist() if 'is_weekend' in trip_group.columns else [0] * seq_len,
            'is_peak_hour_flags': trip_group[
                'is_peak_hour'].tolist() if 'is_peak_hour' in trip_group.columns else [0] * seq_len,

            'arrival_delays': trip_group[
                'arrivalDelay'].tolist() if 'arrivalDelay' in trip_group.columns else [0] * seq_len,
            'departure_delays': trip_group[
                'departureDelay'].tolist() if 'departureDelay' in trip_group.columns else [0] * seq_len,

            'temperatures': trip_group[
                'temperature_2m'].tolist() if 'temperature_2m' in trip_group.columns else [0] * seq_len,
            'apparent_temps': trip_group[
                'apparent_temperature'].tolist() if 'apparent_temperature' in trip_group.columns else [0] * seq_len,
            'precipitations': trip_group[
                'precipitation'].tolist() if 'precipitation' in trip_group.columns else [0] * seq_len,
            'rains': trip_group['rain'].tolist() if 'rain' in trip_group.columns else [0] * seq_len,
            'snowfalls': trip_group['snowfall'].tolist() if 'snowfall' in trip_group.columns else [0] * seq_len,
            'windspeeds': trip_group[
                'windspeed_10m'].tolist() if 'windspeed_10m' in trip_group.columns else [0] * seq_len,
            'windgusts': trip_group[
                'windgusts_10m'].tolist() if 'windgusts_10m' in trip_group.columns else [0] * seq_len,
            'wind_directions': trip_group[
                'winddirection_10m'].tolist() if 'winddirection_10m' in trip_group.columns else [0] * seq_len,

            'speeds': trip_group['speed_mps'].tolist() if 'speed_mps' in trip_group.columns else [0] * seq_len,
        }

        paths.append(path)

    paths_df = pd.DataFrame(paths)

    print(f"   ✓ Aggregated {len(segments_df)} segments into {len(paths_df)} paths")
    print(f"   ✓ Path lengths: min={paths_df['seq_len'].min()}, "
          f"max={paths_df['seq_len'].max()}, "
          f"mean={paths_df['seq_len'].mean():.1f}")

    return paths_df

def _fuzzy_match_station(name, candidates):
    stop_words = {'street', 'avenue', 'place', 'drive', 'road', 'st', 'ave',
                  'platform', '1', '2', 'interchange', 'north', 'crescent'}
    name_core = set(name.lower().replace('&', 'and').split()) - stop_words
    best, best_score = None, 0
    for cand in candidates:
        cand_core = set(cand.lower().replace('&', 'and').split()) - stop_words
        if not name_core or not cand_core:
            continue
        overlap = len(name_core & cand_core) / max(len(name_core | cand_core), 1)
        if overlap > best_score:
            best, best_score = cand, overlap
    return best if best_score > 0.3 else None


def _assign_social_vectors_to_clusters(clusters, soc_df_with_coords):
    cluster_social = {}

    if not soc_df_with_coords:
        print("No social-function stations with GPS coords available")
        return cluster_social

    soc_lats = np.array([s[1] for s in soc_df_with_coords])
    soc_lons = np.array([s[2] for s in soc_df_with_coords])
    soc_names = [s[0] for s in soc_df_with_coords]
    soc_vecs = [s[3] for s in soc_df_with_coords]

    print(f"Assigning social vectors via GPS nearest-neighbour "
          f"({len(soc_df_with_coords)} stations available):")

    for idx, (c_lat, c_lon) in enumerate(clusters):
        dists = np.array([
            haversine_meters(c_lat, c_lon, s_lat, s_lon)
            for s_lat, s_lon in zip(soc_lats, soc_lons)
        ])
        nearest = int(np.argmin(dists))
        cluster_social[idx] = soc_vecs[nearest]

    return cluster_social


def build_adjacency_matrices_fixed(segments_df, clusters,
                                   known_stops=None,
                                   social_path='./data/social_function.csv'):
    print_section("BUILDING ADJACENCY MATRICES")

    if len(segments_df) == 0:
        return None, None, None, []

    segment_types = segments_df['segment_id'].unique()
    n_segments = len(segment_types)
    seg_to_idx = {seg: i for i, seg in enumerate(segment_types)}

    print(f"   Building matrices for {n_segments} segment types")

    adj_geo = np.zeros((n_segments, n_segments))
    adj_dist = np.zeros((n_segments, n_segments))
    adj_soc = np.zeros((n_segments, n_segments))

    SIGMA_M = 500.0
    seg_lengths = {
        sid: segments_df.loc[segments_df['segment_id'] == sid, 'distance_m'].mean()
        for sid in segment_types
    }
    for i, si in enumerate(segment_types):
        for j, sj in enumerate(segment_types):
            if i == j:
                adj_geo[i, j] = 1.0
            else:
                diff = seg_lengths[si] - seg_lengths[sj]
                adj_geo[i, j] = np.exp(-(diff ** 2) / (2 * SIGMA_M ** 2))

    print(f"   ✓ adj_geo  built")

    for sid in segment_types:
        o, d = map(int, sid.split('_'))
        idx_i = seg_to_idx[sid]
        for osid in segment_types:
            oo, _ = map(int, osid.split('_'))
            if d == oo:
                adj_dist[idx_i, seg_to_idx[osid]] = 1.0

    row_sums = adj_dist.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    adj_dist /= row_sums

    print(f"   ✓ adj_dist built")

    _cache = known_stops or {}

    if os.path.exists(social_path):
        soc_df = pd.read_csv(social_path).dropna(subset=['station'])
        print(f"   Social function data: {len(soc_df)} stations loaded")

        soc_df_with_coords = []

        if _cache:
            known_names = list(_cache.keys())
            known_coords = np.array([_cache[n] for n in known_names])

            for _, srow in soc_df.iterrows():
                soc_name = str(srow['station']).strip()
                if not soc_name:
                    continue
                soc_vec = np.array([float(srow['Level 1']),
                                    float(srow['Level 2']),
                                    float(srow['Level 3'])], dtype=float)

                soc_tokens = set(soc_name.lower().replace('&', 'and').split())
                stop_words = {'street', 'avenue', 'place', 'drive', 'road',
                              'st', 'ave', 'platform', '1', '2',
                              'interchange', 'north', 'crescent'}
                soc_core = soc_tokens - stop_words

                best_name, best_score = None, -1.0
                for kn in known_names:
                    kn_core = set(kn.lower().replace('&', 'and').split()) - stop_words
                    if not soc_core or not kn_core:
                        continue
                    score = len(soc_core & kn_core) / max(len(soc_core | kn_core), 1)
                    if score > best_score:
                        best_score, best_name = score, kn

                if best_name and best_score >= 0.25:
                    lat, lon = _cache[best_name]
                    soc_df_with_coords.append((soc_name, lat, lon, soc_vec))
                else:
                    centroid_lat = known_coords[:, 0].mean()
                    centroid_lon = known_coords[:, 1].mean()
                    soc_df_with_coords.append(
                        (soc_name, centroid_lat, centroid_lon, soc_vec)
                    )

        cluster_social = _assign_social_vectors_to_clusters(
            clusters, soc_df_with_coords
        )

        seg_social = np.zeros((n_segments, 3))
        for i, sid in enumerate(segment_types):
            origin, dest = map(int, sid.split('_'))
            v_o = cluster_social.get(origin, np.zeros(3))
            v_d = cluster_social.get(dest, np.zeros(3))
            seg_social[i] = (v_o + v_d) / 2.0

        norms = np.linalg.norm(seg_social, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1e-8
        seg_normed = seg_social / norms
        adj_soc = np.clip(seg_normed @ seg_normed.T, 0.0, 1.0)

        print(f"   ✓ adj_soc  built")

    else:
        print(f"{social_path} not found — adj_soc defaults to identity")
        np.fill_diagonal(adj_soc, 1.0)

    return adj_geo, adj_dist, adj_soc, segment_types