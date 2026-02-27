"""
cluster_hdbscan.py
==================
HDBSCAN-based event-driven clustering.

Public API (identical across all cluster_*.py modules):
    event_driven_clustering_fixed(df, known_stops=None)
        -> (cluster_centers: np.ndarray shape (N,2),
            station_cluster_ids: set of int)

To switch clustering method in main.py, change only:
    from cluster_hdbscan import event_driven_clustering_fixed
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import warnings

warnings.filterwarnings('ignore')

try:
    import hdbscan
    HAS_HDBSCAN = True
    print("✓ HDBSCAN available")
except ImportError:
    HAS_HDBSCAN = False
    print("⚠️  WARNING: hdbscan not available")

from config import print_section, haversine_meters

# =============================================================================
# HDBSCAN EVENT-DRIVEN CLUSTERING
# =============================================================================

def event_driven_clustering_fixed(df, known_stops=None):
    """
    Event-driven clustering on dwell/slowdown points via HDBSCAN.

    Guaranteed station injection
    ─────────────────────────────
    Every entry in `known_stops` (dict: station_name → (lat, lon)) is
    ALWAYS included as a cluster centre, regardless of whether HDBSCAN
    finds enough points nearby.  This ensures all 13 real stations are
    present in the cluster list even when the sample fraction is small.

    After HDBSCAN:
      1. Any detected cluster whose centre is within MERGE_RADIUS_M of a
         known station centre is dropped (the station centre takes over).
      2. Station clusters are prepended to the array so their indices are
         0 … (n_stations-1).  Delay clusters follow at higher indices.

    Returns
    ───────
    cluster_centers  : np.ndarray  shape (N, 2)  — [lat, lon] per cluster
    station_cluster_ids : set of int — indices that are real stations
    """
    print_section("EVENT-DRIVEN CLUSTERING")

    if not HAS_HDBSCAN:
        print("❌ HDBSCAN not available")
        return np.array([]), set()

    # ------------------------------------------------------------------
    # 0. Build station array from known_stops
    # ------------------------------------------------------------------
    station_centers = []   # [(lat, lon, name), ...]
    if known_stops:
        for name, (lat, lon) in known_stops.items():
            if not (np.isnan(lat) or np.isnan(lon)):
                station_centers.append((lat, lon, name))

    print(f"   Station centres to inject: {len(station_centers)}")

    # ------------------------------------------------------------------
    # 1. HDBSCAN on event points
    # ------------------------------------------------------------------
    event_mask = (
        (df['is_long_dwell'] == 1) |
        (df['is_slow_speed'] == 1) |
        (df['is_slowdown'] == 1)
    ) & (df['is_gps_valid'] == 1)

    event_points = df[event_mask].copy()
    print(f"   Event points: {len(event_points):,} "
          f"({len(event_points) / max(len(df), 1) * 100:.1f}%)")

    delay_centers = []   # cluster centres from HDBSCAN

    if len(event_points) >= 50:
        use_slowdown = (event_points['slowdown_lat'].notna() &
                        event_points['slowdown_lon'].notna()).sum()

        if use_slowdown > len(event_points) * 0.5:
            coords = event_points[['slowdown_lat', 'slowdown_lon']].fillna(
                event_points[['latitude', 'longitude']]
            ).values
        else:
            coords = event_points[['latitude', 'longitude']].values

        coords_rad = np.radians(coords)

        clusterer = hdbscan.HDBSCAN(
            metric='haversine',
            min_cluster_size=30,
            min_samples=5
        )
        labels = clusterer.fit_predict(coords_rad)

        n_raw = len(set(labels) - {-1})
        print(f"   HDBSCAN clusters found: {n_raw}")

        for label in range(n_raw):
            mask = labels == label
            if mask.sum() > 0:
                delay_centers.append([coords[mask, 0].mean(),
                                       coords[mask, 1].mean()])
    else:
        print("   ⚠️  Too few event points for HDBSCAN — only station centres used")

    # ------------------------------------------------------------------
    # 2. Merge nearby delay clusters (within 100 m of each other)
    # ------------------------------------------------------------------
    if delay_centers:
        delay_arr = np.array(delay_centers)
        merged = []
        used = set()
        for i in range(len(delay_arr)):
            if i in used:
                continue
            group = [i]
            for j in range(i + 1, len(delay_arr)):
                if j in used:
                    continue
                if haversine_meters(delay_arr[i, 0], delay_arr[i, 1],
                                    delay_arr[j, 0], delay_arr[j, 1]) < 100:
                    group.append(j)
                    used.add(j)
            used.add(i)
            merged.append([np.mean([delay_arr[k, 0] for k in group]),
                            np.mean([delay_arr[k, 1] for k in group])])
        delay_centers = merged

    # ------------------------------------------------------------------
    # 3. Drop any delay cluster whose centre is within MERGE_RADIUS_M of
    #    a known station (the station's precise coordinate takes over)
    # ------------------------------------------------------------------
    MERGE_RADIUS_M = 300   # metres — same as cluster-assign radius

    filtered_delay = []
    for dc in delay_centers:
        too_close = False
        for (s_lat, s_lon, _) in station_centers:
            if haversine_meters(dc[0], dc[1], s_lat, s_lon) <= MERGE_RADIUS_M:
                too_close = True
                break
        if not too_close:
            filtered_delay.append(dc)

    dropped = len(delay_centers) - len(filtered_delay)
    if dropped:
        print(f"   Dropped {dropped} delay cluster(s) absorbed by station zones")

    # ------------------------------------------------------------------
    # 4. Assemble final cluster array
    #    Stations first (indices 0 … n_stations-1) so we know exactly
    #    which indices correspond to real stations.
    # ------------------------------------------------------------------
    final_centers = []
    station_cluster_ids = set()

    for i, (s_lat, s_lon, _) in enumerate(station_centers):
        final_centers.append([s_lat, s_lon])
        station_cluster_ids.add(i)

    for dc in filtered_delay:
        final_centers.append(dc)

    cluster_centers = np.array(final_centers) if final_centers else np.array([])

    print(f"   Station clusters  : {len(station_cluster_ids)}")
    print(f"   Delay clusters    : {len(filtered_delay)}")
    print(f"   Total clusters    : {len(cluster_centers)}")

    return cluster_centers, station_cluster_ids


# =============================================================================
# SEGMENT BUILDING - CORRECTED VERSION
# =============================================================================

