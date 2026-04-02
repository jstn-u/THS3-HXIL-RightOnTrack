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

    if day_of_week >= 5:
        return 0
    return 1 if (7 <= hour < 9 or 16 <= hour < 19) else 0


def _first(trip_df, idx, col):

    if col not in trip_df.columns:
        return None
    val = trip_df.loc[idx, col]
    return None if pd.isna(val) else val


def build_segments_fixed(df, clusters):

    print_section("BUILDING SEGMENTS (WITH BINARY FLAGS v3)")

    if len(clusters) == 0:
        print("❌ No clusters available")
        return pd.DataFrame()

    cluster_tree = BallTree(np.radians(clusters), metric='haversine')

    CLUSTER_ASSIGN_RADIUS_M   = 300
    CLUSTER_ASSIGN_RADIUS_RAD = CLUSTER_ASSIGN_RADIUS_M / 6371000

    valid_gps = df['is_gps_valid'] == 1
    coords    = df.loc[valid_gps, ['latitude', 'longitude']].values

    if len(coords) == 0:
        print("❌ No valid GPS coordinates")
        return pd.DataFrame()

    coords_rad         = np.radians(coords)
    distances, indices = cluster_tree.query(coords_rad, k=1)
    distances          = distances.flatten()
    indices            = indices.flatten()
    within_radius      = distances <= CLUSTER_ASSIGN_RADIUS_RAD

    assigned_ids = np.where(within_radius, indices, -1)
    df.loc[valid_gps, 'cluster_id'] = assigned_ids
    df['cluster_id'] = df['cluster_id'].fillna(-1).astype(int)

    n_assigned = within_radius.sum()
    n_noise    = (~within_radius).sum()
    print(f"   Assigned {n_assigned:,} points to {len(clusters)} clusters "
          f"(within {CLUSTER_ASSIGN_RADIUS_M}m)")
    print(f"   Between-station noise points: {n_noise:,} "
          f"({n_noise / len(coords) * 100:.1f}% — treated as in-transit)")

    operational_features = ['arrivalDelay', 'departureDelay', 'dwellTime_sec']
    weather_features     = ['temperature_2m', 'apparent_temperature', 'precipitation',
                             'rain', 'snowfall', 'windspeed_10m', 'windgusts_10m',
                             'winddirection_10m']
    binary_event_cols    = ['is_slowdown', 'is_congested', 'is_slow_speed']

    available_operational = [f for f in operational_features if f in df.columns]
    available_weather     = [f for f in weather_features     if f in df.columns]
    available_events      = [f for f in binary_event_cols    if f in df.columns]

    print(f"   Preserving features from raw data:")
    print(f"     Operational  : {available_operational}")
    print(f"     Weather      : {available_weather}")
    print(f"     Binary flags : is_weekend, is_peak_hour + {available_events}")

    has_stop_names = ('originStopName' in df.columns and
                      'destinationStopName' in df.columns and
                      'trip_stop_sequence' in df.columns)

    if 'service_date' in df.columns:
        df['_seg_date'] = pd.to_datetime(
            df['service_date'], errors='coerce').dt.normalize()
    else:
        df['_seg_date'] = pd.to_datetime(
            df['timestamp'], errors='coerce').dt.normalize()

    segment_features = []

    if has_stop_names:
        print("   Grouping strategy: originStopName/destinationStopName "
              "(authoritative stop-name path)")
        group_key = ['trip_id', '_seg_date', 'trip_stop_sequence']

        for group_vals, seg_df in df.groupby(group_key, dropna=False, sort=True):
            trip_id, seg_date, stop_seq = group_vals

            if len(seg_df) == 0:
                continue

            seg_df = seg_df.sort_values('timestamp').reset_index(drop=True)
            first_row = seg_df.iloc[0]
            last_row  = seg_df.iloc[-1]

            if ('travel_time_sec' in seg_df.columns and
                    seg_df['travel_time_sec'].notna().any()):
                duration_sec = float(seg_df['travel_time_sec'].sum())
            else:
                t_start = pd.to_datetime(first_row['timestamp'], errors='coerce')
                t_end   = pd.to_datetime(last_row['timestamp'],  errors='coerce')
                if pd.isna(t_start) or pd.isna(t_end):
                    continue
                duration_sec = (t_end - t_start).total_seconds()

            if duration_sec <= 5 or duration_sec > 3600:
                continue

            departure_time = pd.to_datetime(first_row['timestamp'], errors='coerce')
            if pd.isna(departure_time):
                continue
            hour        = departure_time.hour
            day_of_week = departure_time.dayofweek

            distance_m = 0
            if 'odometer' in seg_df.columns:
                odo_vals = seg_df['odometer'].dropna()
                if len(odo_vals) >= 2:
                    distance_m = odo_vals.max() - odo_vals.min()
                    if distance_m < 0 or distance_m > 50000:
                        distance_m = 0

            origin_pings = seg_df[seg_df['cluster_id'] != -1]
            dest_pings   = seg_df[seg_df['cluster_id'] != -1]

            mid = max(1, len(seg_df) // 2)
            origin_candidates = seg_df.iloc[:mid]['cluster_id']
            origin_candidates = origin_candidates[origin_candidates != -1]
            origin_cluster = (int(origin_candidates.mode().iloc[0])
                              if len(origin_candidates) > 0 else -1)

            dest_candidates = seg_df.iloc[mid:]['cluster_id']
            dest_candidates = dest_candidates[dest_candidates != -1]
            dest_cluster = (int(dest_candidates.mode().iloc[0])
                            if len(dest_candidates) > 0 else -1)

            if distance_m == 0 and origin_cluster != -1 and dest_cluster != -1:
                distance_m = haversine_meters(
                    clusters[origin_cluster][0], clusters[origin_cluster][1],
                    clusters[dest_cluster][0],   clusters[dest_cluster][1]
                )

            if distance_m < 50:
                continue

            speed_mps = distance_m / duration_sec
            if speed_mps > 50:
                continue

            operational_data = {}
            for feat in available_operational:
                val = last_row.get(feat)
                operational_data[feat] = float(val) if pd.notna(val) else 0.0

            weather_defaults = {
                'temperature_2m': 15.0, 'apparent_temperature': 15.0,
                'precipitation': 0.0, 'rain': 0.0, 'snowfall': 0.0,
                'windspeed_10m': 5.0, 'windgusts_10m': 10.0,
                'winddirection_10m': 180.0
            }
            weather_data = {}
            for feat in available_weather:
                feat_vals = seg_df[feat].dropna()
                weather_data[feat] = (float(feat_vals.mean())
                                      if len(feat_vals) > 0
                                      else weather_defaults.get(feat, 0.0))

            weekend_flag   = is_weekend(day_of_week)
            peak_flag      = is_peak_hour(hour, day_of_week)
            slowdown_flag  = int(seg_df['is_slowdown'].max())  if 'is_slowdown'  in seg_df.columns else 0
            congested_flag = int(seg_df['is_congested'].max()) if 'is_congested' in seg_df.columns else 0
            slow_spd_flag  = int(seg_df['is_slow_speed'].max()) if 'is_slow_speed' in seg_df.columns else 0
            avg_congestion = (seg_df['congestionLevel'].mean()
                              if 'congestionLevel' in seg_df.columns else 0)

            TRAVEL_DIST_M = 200
            segment_type  = 'TRAVEL' if distance_m >= TRAVEL_DIST_M else 'DWELL'
            is_travel     = int(segment_type == 'TRAVEL')

            if origin_cluster != -1 and dest_cluster != -1:
                segment_id = f"{origin_cluster}_{dest_cluster}"
            else:
                o_key = str(first_row.get('originStopName', 'unk')).replace(' ', '_')
                d_key = str(last_row.get('destinationStopName', 'unk')).replace(' ', '_')
                segment_id = f"{o_key}__{d_key}"

            segment_dict = {
                'segment_id':            segment_id,
                'origin_cluster':        origin_cluster,
                'dest_cluster':          dest_cluster,
                'duration_sec':          duration_sec,
                'distance_m':            distance_m,
                'speed_mps':             speed_mps,
                'hour':                  hour,
                'day_of_week':           day_of_week,
                'departure_time':        departure_time.isoformat(),
                'segment_type':          segment_type,
                'is_travel':             is_travel,
                'is_weekend':            weekend_flag,
                'is_peak_hour':          peak_flag,
                'is_slowdown':           slowdown_flag,
                'is_congested':          congested_flag,
                'is_slow_speed':         slow_spd_flag,
                'congestion':            avg_congestion,
                'n_points':              len(seg_df),
                'n_noise_points':        0,
                'trip_id':               trip_id,
                'trip_stop_sequence':    stop_seq,
                'vehicle_id':            first_row.get('vehicleID'),
                'vehicleLabel':          first_row.get('vehicleLabel'),
                'vehicleLicencePlate':   first_row.get('vehicleLicenceplate'),
                'vehicle_stop_sequence': first_row.get('vehicle_stop_sequence'),
                'originStopID':          first_row.get('originStopID'),
                'originStopName':        first_row.get('originStopName'),
                'originLat':             first_row.get('originLat'),
                'originLon':             first_row.get('originLon'),
                'destinationStopID':     last_row.get('destinationStopID'),
                'destinationStopName':   last_row.get('destinationStopName'),
                'destinationLat':        last_row.get('destinationLat'),
                'destinationLon':        last_row.get('destinationLon'),
                'total_travel_time_sec': (
                    float(last_row['total_travel_time_sec'])
                    if 'total_travel_time_sec' in seg_df.columns
                    and pd.notna(last_row.get('total_travel_time_sec'))
                    else None
                ),
            }
            segment_dict.update(operational_data)
            segment_dict.update(weather_data)
            segment_features.append(segment_dict)

    else:
        print("   Grouping strategy: GPS cluster transitions (fallback — "
              "originStopName/trip_stop_sequence not found in data)")

        df['_trip_date'] = pd.to_datetime(
            df['timestamp'], errors='coerce').dt.normalize()

        available_events_local = [f for f in binary_event_cols if f in df.columns]

        for (trip_id, _trip_date), trip_df in df.groupby(
                ['trip_id', '_trip_date'], dropna=False):
            if 'trip_stop_sequence' in trip_df.columns:
                trip_df = trip_df.sort_values(
                    ['trip_stop_sequence', 'timestamp']).reset_index(drop=True)
            else:
                trip_df = trip_df.sort_values('timestamp').reset_index(drop=True)

            i = 0
            while i < len(trip_df):
                if trip_df.loc[i, 'cluster_id'] == -1:
                    i += 1
                    continue

                origin_cluster  = trip_df.loc[i, 'cluster_id']
                last_origin_idx = i
                j = i + 1
                while j < len(trip_df):
                    if trip_df.loc[j, 'cluster_id'] == origin_cluster:
                        last_origin_idx = j
                    j += 1

                dest_found  = False
                noise_count = 0
                j = last_origin_idx + 1
                while j < len(trip_df):
                    current_cluster = trip_df.loc[j, 'cluster_id']
                    if current_cluster == -1:
                        noise_count += 1
                        j += 1
                        continue
                    if current_cluster != origin_cluster:
                        dest_cluster     = current_cluster
                        dest_arrival_idx = j
                        dest_found       = True
                        break
                    j += 1

                if not dest_found:
                    break

                departure_time = pd.to_datetime(
                    trip_df.loc[last_origin_idx, 'timestamp'])
                arrival_time   = pd.to_datetime(
                    trip_df.loc[dest_arrival_idx, 'timestamp'])

                if ('travel_time_sec' in trip_df.columns
                        and trip_df.loc[last_origin_idx:dest_arrival_idx,
                                        'travel_time_sec'].notna().any()):
                    duration_sec = float(
                        trip_df.loc[last_origin_idx:dest_arrival_idx,
                                    'travel_time_sec'].sum())
                else:
                    duration_sec = (arrival_time - departure_time).total_seconds()

                if duration_sec <= 5 or duration_sec > 3600:
                    i = dest_arrival_idx
                    continue

                seg_df     = trip_df.loc[last_origin_idx:dest_arrival_idx]
                distance_m = 0

                if 'odometer' in seg_df.columns:
                    odo_vals = seg_df['odometer'].dropna()
                    if len(odo_vals) >= 2:
                        distance_m = odo_vals.max() - odo_vals.min()
                        if distance_m < 0 or distance_m > 50000:
                            distance_m = 0

                if distance_m == 0:
                    distance_m = haversine_meters(
                        clusters[origin_cluster][0], clusters[origin_cluster][1],
                        clusters[dest_cluster][0],   clusters[dest_cluster][1]
                    )

                if distance_m < 50:
                    i = dest_arrival_idx
                    continue

                speed_mps = distance_m / duration_sec
                if speed_mps > 50:
                    i = dest_arrival_idx
                    continue

                hour        = departure_time.hour
                day_of_week = departure_time.dayofweek

                weekend_flag   = is_weekend(day_of_week)
                peak_flag      = is_peak_hour(hour, day_of_week)
                slowdown_flag  = int(seg_df['is_slowdown'].max())  if 'is_slowdown'  in seg_df.columns else 0
                congested_flag = int(seg_df['is_congested'].max()) if 'is_congested' in seg_df.columns else 0
                slow_spd_flag  = int(seg_df['is_slow_speed'].max()) if 'is_slow_speed' in seg_df.columns else 0
                avg_congestion = (seg_df['congestionLevel'].mean()
                                  if 'congestionLevel' in seg_df.columns else 0)

                operational_data = {}
                for feat in available_operational:
                    feat_vals = seg_df[feat].dropna()
                    operational_data[feat] = (float(feat_vals.mean())
                                              if len(feat_vals) > 0 else 0.0)

                weather_defaults = {
                    'temperature_2m': 15.0, 'apparent_temperature': 15.0,
                    'precipitation': 0.0, 'rain': 0.0, 'snowfall': 0.0,
                    'windspeed_10m': 5.0, 'windgusts_10m': 10.0,
                    'winddirection_10m': 180.0
                }
                weather_data = {}
                for feat in available_weather:
                    feat_vals = seg_df[feat].dropna()
                    weather_data[feat] = (float(feat_vals.mean())
                                          if len(feat_vals) > 0
                                          else weather_defaults.get(feat, 0.0))

                TRAVEL_DIST_M = 200
                segment_type  = 'TRAVEL' if distance_m >= TRAVEL_DIST_M else 'DWELL'
                is_travel     = int(segment_type == 'TRAVEL')

                segment_dict = {
                    'segment_id':            f"{origin_cluster}_{dest_cluster}",
                    'origin_cluster':        origin_cluster,
                    'dest_cluster':          dest_cluster,
                    'duration_sec':          duration_sec,
                    'distance_m':            distance_m,
                    'speed_mps':             speed_mps,
                    'hour':                  hour,
                    'day_of_week':           day_of_week,
                    'departure_time':        departure_time.isoformat(),
                    'segment_type':          segment_type,
                    'is_travel':             is_travel,
                    'is_weekend':            weekend_flag,
                    'is_peak_hour':          peak_flag,
                    'is_slowdown':           slowdown_flag,
                    'is_congested':          congested_flag,
                    'is_slow_speed':         slow_spd_flag,
                    'congestion':            avg_congestion,
                    'n_points':              len(seg_df),
                    'n_noise_points':        noise_count,
                    'trip_id':               trip_id,
                    'vehicle_id':            _first(trip_df, last_origin_idx, 'vehicle_id'),
                    'vehicleLabel':          _first(trip_df, last_origin_idx, 'vehicleLabel'),
                    'vehicleLicencePlate':   _first(trip_df, last_origin_idx, 'vehicleLicencePlate'),
                    'trip_stop_sequence':    _first(trip_df, last_origin_idx, 'trip_stop_sequence'),
                    'vehicle_stop_sequence': _first(trip_df, last_origin_idx, 'vehicle_stop_sequence'),
                    'originStopID':          _first(trip_df, last_origin_idx, 'originStopID'),
                    'originStopName':        _first(trip_df, last_origin_idx, 'originStopName'),
                    'originLat':             _first(trip_df, last_origin_idx, 'originLat'),
                    'originLon':             _first(trip_df, last_origin_idx, 'originLon'),
                    'destinationStopID':     _first(trip_df, dest_arrival_idx, 'destinationStopID'),
                    'destinationStopName':   _first(trip_df, dest_arrival_idx, 'destinationStopName'),
                    'destinationLat':        _first(trip_df, dest_arrival_idx, 'destinationLat'),
                    'destinationLon':        _first(trip_df, dest_arrival_idx, 'destinationLon'),
                    'total_travel_time_sec': (
                        float(trip_df.loc[dest_arrival_idx, 'total_travel_time_sec'])
                        if 'total_travel_time_sec' in trip_df.columns
                        and pd.notna(trip_df.loc[dest_arrival_idx, 'total_travel_time_sec'])
                        else None
                    ),
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

        if 'segment_type' in segments_df.columns:
            n_travel = (segments_df['segment_type'] == 'TRAVEL').sum()
            n_dwell  = (segments_df['segment_type'] == 'DWELL').sum()
            pct_t    = 100 * n_travel / max(1, len(segments_df))
            print(f"\n✓ Segment Type Split (distance threshold = 200m):")
            print(f"   TRAVEL (distance ≥ 200m): {n_travel:,} ({pct_t:.1f}%)  "
                  f"← continuous regression target, LSTM is meaningful here")
            print(f"   DWELL  (distance <  200m): {n_dwell:,} ({100-pct_t:.1f}%)  "
                  f"← discrete 15/30s target, treated separately")

        print(f"\n✓ Binary Flag Statistics:")
        for col in ['is_weekend', 'is_peak_hour', 'is_slowdown',
                    'is_congested', 'is_slow_speed']:
            if col in segments_df.columns:
                cnt = segments_df[col].sum()
                pct = segments_df[col].mean() * 100
                print(f"   {col:<18}: {cnt:,} ({pct:.1f}%)")

        if 'arrivalDelay' in segments_df.columns:
            print(f"\n✓ Delay Analysis by Context:")
            for flag, label in [('is_weekend', 'Weekend'),
                                 ('is_peak_hour', 'Peak hour'),
                                 ('is_slowdown', 'Slowdown'),
                                 ('is_congested', 'Congested')]:
                if flag in segments_df.columns:
                    in_grp  = segments_df[segments_df[flag] == 1]['arrivalDelay'].mean()
                    out_grp = segments_df[segments_df[flag] == 0]['arrivalDelay'].mean()
                    print(f"   {label:<12} avg delay: {in_grp:.2f}s  "
                          f"(non-{label.lower()}: {out_grp:.2f}s)")

        if available_operational:
            print(f"\n🔍 Operational features (RAW):")
            for feat in available_operational:
                if feat in segments_df.columns:
                    vals = segments_df[feat].values
                    print(f"   {feat}: min={vals.min():.4f}, max={vals.max():.4f}, "
                          f"mean={vals.mean():.4f}, std={vals.std():.4f}")

    return segments_df


def aggregate_segments_into_paths(segments_df, max_path_length=10):
    """
    Aggregate individual segments into multi-segment paths based on trip_id.
    v2: Preserves is_slowdown and is_congested alongside existing flags.
    """
    print_section("AGGREGATING SEGMENTS INTO PATHS")

    if 'trip_id' not in segments_df.columns:
        print("   ❌ No 'trip_id' column found")
        return segments_df

    paths = []

    if '_date' not in segments_df.columns and 'departure_time' in segments_df.columns:
        segments_df = segments_df.copy()
        segments_df['_date'] = pd.to_datetime(
            segments_df['departure_time'], errors='coerce').dt.normalize()
    group_cols = ['trip_id', '_date'] if '_date' in segments_df.columns else ['trip_id']

    for group_key, trip_group in segments_df.groupby(group_cols):
        trip_id = group_key[0] if isinstance(group_key, tuple) else group_key
        if 'trip_stop_sequence' in trip_group.columns:
            trip_group = trip_group.sort_values('trip_stop_sequence')
        elif 'departure_time' in trip_group.columns:
            trip_group = (trip_group
                          .assign(_dep_dt=pd.to_datetime(
                              trip_group['departure_time'], errors='coerce'))
                          .sort_values('_dep_dt')
                          .drop(columns=['_dep_dt']))
        elif 'hour' in trip_group.columns:
            trip_group = trip_group.sort_values('hour')
        trip_group = trip_group.head(max_path_length)
        seq_len    = len(trip_group)

        if seq_len == 0:
            continue

        def _col(name, default=0):
            return (trip_group[name].tolist()
                    if name in trip_group.columns
                    else [default] * seq_len)

        if ('total_travel_time_sec' in trip_group.columns):
            vals = trip_group['total_travel_time_sec'].dropna()
            total_duration = float(vals.sum()) if len(vals) > 0 else float(trip_group['duration_sec'].sum())
        else:
            total_duration = float(trip_group['duration_sec'].sum())

        path = {
            'trip_id':           trip_id,
            'seq_len':           seq_len,
            'total_duration':    total_duration,

            'segment_ids':       trip_group['segment_id'].tolist(),
            'segment_durations': trip_group['duration_sec'].tolist(),

            'hours':             _col('hour'),
            'days_of_week':      _col('day_of_week'),

            'is_weekend_flags':    _col('is_weekend'),
            'is_peak_hour_flags':  _col('is_peak_hour'),
            'is_slowdown_flags':   _col('is_slowdown'),
            'is_congested_flags':  _col('is_congested'),
            'is_slow_speed_flags': _col('is_slow_speed'),

            'arrival_delays':    _col('arrivalDelay'),
            'departure_delays':  _col('departureDelay'),

            'temperatures':      _col('temperature_2m'),
            'apparent_temps':    _col('apparent_temperature'),
            'precipitations':    _col('precipitation'),
            'rains':             _col('rain'),
            'snowfalls':         _col('snowfall'),
            'windspeeds':        _col('windspeed_10m'),
            'windgusts':         _col('windgusts_10m'),
            'wind_directions':   _col('winddirection_10m'),

            'speeds':            _col('speed_mps'),
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
    name_core  = set(name.lower().replace('&', 'and').split()) - stop_words
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
        print("     ⚠️  No social-function stations with GPS coords available")
        return cluster_social

    soc_lats  = np.array([s[1] for s in soc_df_with_coords])
    soc_lons  = np.array([s[2] for s in soc_df_with_coords])
    soc_vecs  = [s[3] for s in soc_df_with_coords]

    print(f"     Assigning social vectors via GPS nearest-neighbour "
          f"({len(soc_df_with_coords)} stations available):")

    for idx, (c_lat, c_lon) in enumerate(clusters):
        dists   = np.array([
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

    def _is_int_seg_id(sid):
        try:
            parts = str(sid).split('_')
            return len(parts) == 2 and all(p.lstrip('-').isdigit() for p in parts)
        except Exception:
            return False

    all_types     = segments_df['segment_id'].unique()
    segment_types = np.array([s for s in all_types if _is_int_seg_id(s)])
    n_skipped     = len(all_types) - len(segment_types)
    if n_skipped:
        print(f"   ⚠  Skipped {n_skipped} non-integer segment IDs "
              f"(fallback stop-name keys — no valid cluster assignment)")
    n_segments    = len(segment_types)
    seg_to_idx    = {seg: i for i, seg in enumerate(segment_types)}

    print(f"   Building matrices for {n_segments} segment types")

    adj_geo  = np.zeros((n_segments, n_segments))
    adj_dist = np.zeros((n_segments, n_segments))
    adj_soc  = np.zeros((n_segments, n_segments))

    SIGMA_M    = 500.0
    seg_lengths = {
        sid: segments_df.loc[segments_df['segment_id'] == sid,
                             'distance_m'].mean()
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
        try:
            o, d = map(int, str(sid).split('_'))
        except (ValueError, AttributeError):
            continue
        idx_i = seg_to_idx[sid]
        for osid in segment_types:
            try:
                oo, _ = map(int, str(osid).split('_'))
            except (ValueError, AttributeError):
                continue
            if d == oo:
                adj_dist[idx_i, seg_to_idx[osid]] = 1.0

    row_sums             = adj_dist.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    adj_dist            /= row_sums
    print(f"   ✓ adj_dist built")

    _cache = known_stops or {}

    if os.path.exists(social_path):
        soc_df = pd.read_csv(social_path).dropna(subset=['station'])
        print(f"   Social function data: {len(soc_df)} stations loaded")

        soc_df_with_coords = []

        if _cache:
            known_names  = list(_cache.keys())
            known_coords = np.array([_cache[n] for n in known_names])

            for _, srow in soc_df.iterrows():
                soc_name = str(srow['station']).strip()
                if not soc_name:
                    continue
                soc_vec = np.array([float(srow['Level 1']),
                                    float(srow['Level 2']),
                                    float(srow['Level 3'])], dtype=float)

                stop_words = {'street', 'avenue', 'place', 'drive', 'road',
                              'st', 'ave', 'platform', '1', '2',
                              'interchange', 'north', 'crescent'}
                soc_core   = (set(soc_name.lower().replace('&', 'and').split())
                               - stop_words)

                best_name, best_score = None, -1.0
                for kn in known_names:
                    kn_core = (set(kn.lower().replace('&', 'and').split())
                                - stop_words)
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
                        (soc_name, centroid_lat, centroid_lon, soc_vec))

        cluster_social = _assign_social_vectors_to_clusters(
            clusters, soc_df_with_coords)

        seg_social = np.zeros((n_segments, 3))
        for i, sid in enumerate(segment_types):
            try:
                origin, dest = map(int, str(sid).split('_'))
            except (ValueError, AttributeError):
                continue
            v_o          = cluster_social.get(origin, np.zeros(3))
            v_d          = cluster_social.get(dest,   np.zeros(3))
            seg_social[i] = (v_o + v_d) / 2.0

        norms               = np.linalg.norm(seg_social, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1e-8
        seg_normed          = seg_social / norms
        adj_soc             = np.clip(seg_normed @ seg_normed.T, 0.0, 1.0)
        print(f"   ✓ adj_soc  built")

    else:
        print(f"   ⚠️  {social_path} not found — adj_soc defaults to identity")
        np.fill_diagonal(adj_soc, 1.0)

    return adj_geo, adj_dist, adj_soc, segment_types
