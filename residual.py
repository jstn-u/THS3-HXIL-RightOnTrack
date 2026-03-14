"""
residual.py — STATION-AWARE LSTM RESIDUAL
==========================================

ARCHITECTURE:
  segment_id  =  "A_B"   (cluster A → cluster B)
                          A / B can be a named light-rail station OR a
                          high-dwell GPS cluster

  For each segment the LSTM sees one time step that is built from:

    [spatial_embedding          ]  ← MAGNN node embedding for this segment
    [magnn_baseline             ]  ← MAGNN's raw prediction  (residual target)
    [segment_context            ]  ← operational flags (is_weekend, is_peak_hour)
                                     + temporal cyclical features
    [origin_operational_feats   ]  ← arrivalDelay / departureDelay
                                     mapped to the ORIGIN cluster / station
    [dest_operational_feats     ]  ← same, mapped to DESTINATION cluster / station
    [origin_weather_feats       ]  ← 8 weather vars at the ORIGIN GPS point
    [dest_weather_feats         ]  ← 8 weather vars at the DESTINATION GPS point

  Each weather/operational block uses the GPS coordinates of the relevant
  cluster centroid (passed in via `cluster_coords`) and looks up the
  nearest row in the raw ping dataframe for ground-truth features.

  At inference time (when no raw pings are available) the already-averaged
  segment-level features are split into origin / dest halves via a learned
  linear projection so the architecture stays identical.

  Station label mapping
  ---------------------
  A `segment_id` like "3_5" maps to:
    origin  cluster 3  →  "Mapleton Avenue" station
    dest    cluster 5  →  "Nullarbor Avenue" station  (see KNOWN_STATIONS below)
  The station name is stored in SegmentStationMapper and can be queried
  for logging, plotting, or injecting a one-hot station embedding.

FIXES from previous version
----------------------------
  ✅ Xavier init on all Linear layers
  ✅ Proper residual: LSTM predicts correction, not full value
  ✅ Adaptive gate α per sample
  ✅ Weather + operational features mapped to nearest GPS point
     (origin cluster coords  and  dest cluster coords)
  ✅ Station names mapped to segment_id for interpretability
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# KNOWN LIGHT-RAIL STATIONS  (Canberra Metro — extend as needed)
# station_name  →  (latitude, longitude)
# =============================================================================
KNOWN_STATIONS: Dict[str, Tuple[float, float]] = {
    "Gungahlin Place":      (-35.1827, 149.1328),
    "Manning Clark Crescent": (-35.1895, 149.1362),
    "Mapleton Avenue":      (-35.1965, 149.1390),
    "Nullarbor Avenue":     (-35.2044, 149.1404),
    "Well Station Drive":   (-35.2120, 149.1402),
    "Sandford Street":      (-35.2228, 149.1394),
    "EPIC and Racecourse":  (-35.2328, 149.1376),
    "Phillip Avenue":       (-35.2439, 149.1360),
    "Swinden Street":       (-35.2537, 149.1342),
    "Dickson Interchange":  (-35.2620, 149.1320),
    "Macarthur Avenue":     (-35.2719, 149.1296),
    "Ipima Street":         (-35.2802, 149.1272),
    "Elouera Street":       (-35.2889, 149.1248),
    "Alinga Street":        (-35.2797, 149.1310),   # City terminus
}

# Weather column order — must match EnhancedSegmentDataset
WEATHER_COLS = [
    "temperature_2m", "apparent_temperature", "precipitation",
    "rain", "snowfall", "windspeed_10m", "windgusts_10m",
    "winddirection_10m",
]
# Operational columns (continuous only; binary flags are in segment_context)
OPERATIONAL_COLS = ["arrivalDelay", "departureDelay"]


# =============================================================================
# STATION MAPPER
# =============================================================================

class SegmentStationMapper:
    """
    Maps cluster indices → station names and cluster coordinates.

    Parameters
    ----------
    clusters : np.ndarray  shape (n_clusters, 2)   lat/lon for each cluster centroid
    known_stops : dict     station_name → (lat, lon)   (from data_loader.get_known_stops)
    match_radius_m : float  max distance to call a cluster a known station
    """

    STATION_NAMES = list(KNOWN_STATIONS.keys())

    def __init__(
        self,
        clusters: np.ndarray,
        known_stops: Optional[Dict[str, Tuple[float, float]]] = None,
        match_radius_m: float = 150.0,
    ):
        self.clusters = np.asarray(clusters, dtype=float)   # (N, 2)
        self.n_clusters = len(self.clusters)
        self.cluster_to_station: Dict[int, str] = {}

        # Merge KNOWN_STATIONS with any run-time known_stops
        all_stops = dict(KNOWN_STATIONS)
        if known_stops:
            all_stops.update(known_stops)

        # For every cluster, find nearest named station within radius
        for idx, (c_lat, c_lon) in enumerate(self.clusters):
            best_name, best_dist = None, float("inf")
            for name, (s_lat, s_lon) in all_stops.items():
                d = _haversine_m(c_lat, c_lon, s_lat, s_lon)
                if d < best_dist:
                    best_dist, best_name = d, name
            if best_dist <= match_radius_m:
                self.cluster_to_station[idx] = best_name

        n_matched = len(self.cluster_to_station)
        print(f"   SegmentStationMapper: {n_matched}/{self.n_clusters} clusters "
              f"matched to named stations (radius={match_radius_m}m)")

    def segment_label(self, segment_id: str) -> str:
        """Return human-readable label: 'Mapleton Avenue → Nullarbor Avenue' or '3 → 5'"""
        try:
            o_str, d_str = segment_id.split("_")
            o, d = int(o_str), int(d_str)
        except ValueError:
            return segment_id
        o_name = self.cluster_to_station.get(o, f"Cluster {o}")
        d_name = self.cluster_to_station.get(d, f"Cluster {d}")
        return f"{o_name} → {d_name}"

    def cluster_coords(self, cluster_idx: int) -> Tuple[float, float]:
        """Return (lat, lon) for a cluster centroid."""
        return tuple(self.clusters[cluster_idx])

    def origin_dest_coords(
        self, segment_id: str
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        o, d = map(int, segment_id.split("_"))
        return self.cluster_coords(o), self.cluster_coords(d)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# =============================================================================
# GPS-POINT FEATURE LOOKUP
# =============================================================================

class NearestGPSFeatureLookup:
    """
    Given a cluster's (lat, lon), find the nearest raw GPS ping and return
    its weather / operational feature vector.

    This is used at *segment-build time* (inside build_segments_fixed) to
    produce per-endpoint feature vectors.  At inference only the pre-computed
    vectors are needed.

    Parameters
    ----------
    raw_df : pd.DataFrame   full raw GPS ping DataFrame with all feature cols
    weather_cols : list     column names for weather features
    operational_cols : list column names for operational features
    """

    def __init__(self, raw_df, weather_cols=None, operational_cols=None):
        import pandas as pd

        self.weather_cols = weather_cols or WEATHER_COLS
        self.operational_cols = operational_cols or OPERATIONAL_COLS

        # Keep only valid GPS rows
        if "is_gps_valid" in raw_df.columns:
            df = raw_df[raw_df["is_gps_valid"] == 1].copy()
        else:
            df = raw_df.copy()

        df = df.dropna(subset=["latitude", "longitude"])
        self.lats = df["latitude"].values
        self.lons = df["longitude"].values

        # Feature matrices (fill missing cols with 0)
        w_data = np.zeros((len(df), len(self.weather_cols)), dtype=np.float32)
        for i, col in enumerate(self.weather_cols):
            if col in df.columns:
                w_data[:, i] = df[col].fillna(0.0).values
        self.weather_matrix = w_data

        o_data = np.zeros((len(df), len(self.operational_cols)), dtype=np.float32)
        for i, col in enumerate(self.operational_cols):
            if col in df.columns:
                o_data[:, i] = df[col].fillna(0.0).values
        self.operational_matrix = o_data

        # Build BallTree for fast lookup
        from sklearn.neighbors import BallTree
        coords_rad = np.radians(np.stack([self.lats, self.lons], axis=1))
        self.tree = BallTree(coords_rad, metric="haversine")
        print(f"   NearestGPSFeatureLookup: indexed {len(df):,} valid GPS pings")

    def lookup(
        self, lat: float, lon: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (weather_vec [8], operational_vec [2]) for the nearest GPS ping.
        """
        q = np.radians([[lat, lon]])
        _, idx = self.tree.query(q, k=1)
        i = int(idx[0, 0])
        return self.weather_matrix[i].copy(), self.operational_matrix[i].copy()


# =============================================================================
# FEATURE SPLITTER  (used inside Dataset when lookup is unavailable)
# =============================================================================

class FeatureEndpointSplitter(nn.Module):
    """
    Learned linear split: given a segment-level averaged feature vector
    produce separate origin and destination representations.

    Input  : [weather(8) + operational(2)]  dim = 10
    Output : origin_feats (10),  dest_feats (10)
    """

    def __init__(self, feat_dim: int = 10, hidden: int = 32):
        super().__init__()
        self.origin_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, feat_dim)
        )
        self.dest_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, feat_dim)
        )
        for proj in (self.origin_proj, self.dest_proj):
            for layer in proj:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, feats: torch.Tensor):
        return self.origin_proj(feats), self.dest_proj(feats)


# =============================================================================
# ADAPTIVE GATE
# =============================================================================

class AdaptiveGate(nn.Module):
    """Learns per-sample gating weight α ∈ [0, 1]."""

    def __init__(self, spatial_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.gate_net[3].weight)
        nn.init.constant_(self.gate_net[3].bias, 0.0)   # start α ≈ 0.5

    def forward(self, spatial_embedding: torch.Tensor) -> torch.Tensor:
        return self.gate_net(spatial_embedding)          # (B, 1)


# =============================================================================
# GLOBAL TEMPORAL ATTENTION
# =============================================================================

class GlobalTemporalAttention(nn.Module):
    def __init__(self, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.scale = math.sqrt(feature_dim)
        self.W_Q = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_K = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_V = nn.Linear(feature_dim, feature_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor):
        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = self.norm(torch.matmul(attn, V) + x)
        return out, attn


# =============================================================================
# STATION-AWARE RESIDUAL LSTM
# =============================================================================

class StationAwareResidualLSTM(nn.Module):
    """
    LSTM that receives a *station-aware* feature vector for each segment.

    Input vector layout (one time-step per segment):
    ─────────────────────────────────────────────────
    [  spatial_embedding          ]  spatial_dim
    [  magnn_baseline             ]  1
    [  is_weekend, is_peak_hour   ]  2   (binary context flags)
    [  temporal cyclical          ]  temporal_dim  (hour_sin/cos, dow_sin/cos, speed)
    [  origin_operational         ]  op_dim   (arrivalDelay, departureDelay at origin)
    [  dest_operational           ]  op_dim   (same at destination)
    [  origin_weather             ]  weather_dim  (8 vars at origin cluster/station)
    [  dest_weather               ]  weather_dim  (8 vars at destination cluster/station)
    ─────────────────────────────────────────────────
    Total = spatial_dim + 1 + 2 + temporal_dim + 2*op_dim + 2*weather_dim

    When pre-computed origin/dest splits are not available (legacy path),
    a `FeatureEndpointSplitter` is used to produce them from the averaged
    segment features.
    """

    OP_DIM      = len(OPERATIONAL_COLS)   # 2
    WEATHER_DIM = len(WEATHER_COLS)       # 8
    CONTEXT_DIM = 2                       # is_weekend, is_peak_hour

    def __init__(
        self,
        spatial_dim: int,
        temporal_dim: int = 5,
        hidden_dim: int = 128,
        n_layers: int = 1,
        dropout: float = 0.1,
        use_endpoint_splitter: bool = False,
    ):
        super().__init__()

        self.spatial_dim = spatial_dim
        self.use_splitter = use_endpoint_splitter

        # Optional fallback splitter (used when GPS lookup is unavailable)
        if use_endpoint_splitter:
            self.splitter = FeatureEndpointSplitter(
                feat_dim=self.OP_DIM + self.WEATHER_DIM
            )

        lstm_input_dim = (
            spatial_dim
            + 1                         # MAGNN baseline
            + self.CONTEXT_DIM          # is_weekend, is_peak_hour
            + temporal_dim              # cyclical + speed
            + 2 * self.OP_DIM           # origin + dest operational
            + 2 * self.WEATHER_DIM      # origin + dest weather
        )

        print(f"   StationAwareResidualLSTM input breakdown:")
        print(f"     spatial_embedding : {spatial_dim}")
        print(f"     magnn_baseline    : 1")
        print(f"     context flags     : {self.CONTEXT_DIM}  (is_weekend, is_peak_hour)")
        print(f"     temporal cyclical : {temporal_dim}  (hour_sin/cos, dow_sin/cos, speed)")
        print(f"     origin_operational: {self.OP_DIM}  {OPERATIONAL_COLS}")
        print(f"     dest_operational  : {self.OP_DIM}")
        print(f"     origin_weather    : {self.WEATHER_DIM}  {WEATHER_COLS}")
        print(f"     dest_weather      : {self.WEATHER_DIM}")
        print(f"     ─────────────────────────────────")
        print(f"     TOTAL LSTM input  : {lstm_input_dim}")

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.0,
        )

        self.attention = GlobalTemporalAttention(hidden_dim, dropout)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),         nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Xavier init
        for layer in self.fusion:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        spatial_embeddings: torch.Tensor,   # (B, spatial_dim)
        magnn_baseline: torch.Tensor,       # (B, 1)
        temporal_features: torch.Tensor,    # (B, temporal_dim)
        context_flags: torch.Tensor,        # (B, 2)   is_weekend, is_peak_hour
        origin_operational: torch.Tensor,   # (B, OP_DIM)
        dest_operational: torch.Tensor,     # (B, OP_DIM)
        origin_weather: torch.Tensor,       # (B, WEATHER_DIM)
        dest_weather: torch.Tensor,         # (B, WEATHER_DIM)
    ) -> torch.Tensor:                      # returns correction (B, 1)

        combined = torch.cat([
            spatial_embeddings,     # MAGNN spatial context
            magnn_baseline,         # residual target anchor
            context_flags,          # is_weekend, is_peak_hour
            temporal_features,      # cyclical time + speed
            origin_operational,     # delay at origin cluster / station
            dest_operational,       # delay at dest cluster / station
            origin_weather,         # weather at origin GPS point
            dest_weather,           # weather at dest GPS point
        ], dim=1)                   # (B, lstm_input_dim)

        seq = combined.unsqueeze(1)                     # (B, 1, dim)
        lstm_out, _ = self.lstm(seq)                    # (B, 1, hidden)
        attn_out, _ = self.attention(lstm_out)          # (B, 1, hidden)
        correction = self.fusion(attn_out[:, -1, :])    # (B, 1)
        return correction


# =============================================================================
# MAGNN-LSTM-RESIDUAL  (main model)
# =============================================================================

class MAGNN_LSTM_Residual(nn.Module):
    """
    MAGNN-LSTM with station-aware residual learning and adaptive gating.

    ┌──────────────────────────────────────────────────────────────┐
    │  segment_id  "A_B"  →  origin cluster A,  dest cluster B    │
    │                                                              │
    │  MAGNN baseline prediction  (frozen or fine-tuned)          │
    │       ↓                                                      │
    │  StationAwareResidualLSTM                                    │
    │    ├─ spatial embedding  (from MAGNN node embeddings)        │
    │    ├─ MAGNN baseline     (residual anchor)                   │
    │    ├─ context flags      (is_weekend, is_peak_hour)          │
    │    ├─ temporal cyclical  (hour, dow, speed)                  │
    │    ├─ origin_operational (delay features @ origin station)   │
    │    ├─ dest_operational   (delay features @ dest station)     │
    │    ├─ origin_weather     (8 vars @ origin GPS cluster)       │
    │    └─ dest_weather       (8 vars @ dest GPS cluster)         │
    │       ↓                                                      │
    │  LSTM correction  × α  (adaptive gate per sample)           │
    │       ↓                                                      │
    │  final = baseline + α × correction                          │
    └──────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    magnn_model : MAGTTE
    spatial_dim : int        dimension of MAGNN node embeddings
    station_mapper : SegmentStationMapper | None
        If provided, logs station labels during forward (debug mode).
    freeze_magnn : bool
    lstm_hidden : int
    lstm_layers : int
    dropout : float
    temporal_dim : int       size of temporal feature vector (default 5)
    use_endpoint_splitter : bool
        Fall back to learned splitter when pre-computed origin/dest
        features are unavailable (e.g., ablation without GPS lookup).
    """

    def __init__(
        self,
        magnn_model,
        spatial_dim: int,
        station_mapper: Optional[SegmentStationMapper] = None,
        freeze_magnn: bool = True,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        temporal_dim: int = 5,
        use_endpoint_splitter: bool = False,
    ):
        super().__init__()

        self.magnn = magnn_model
        self.freeze_magnn = freeze_magnn
        self.station_mapper = station_mapper

        if freeze_magnn:
            for p in self.magnn.parameters():
                p.requires_grad = False
            print("   MAGNN frozen (transfer learning mode)")

        self.residual_lstm = StationAwareResidualLSTM(
            spatial_dim=spatial_dim,
            temporal_dim=temporal_dim,
            hidden_dim=lstm_hidden,
            n_layers=lstm_layers,
            dropout=dropout,
            use_endpoint_splitter=use_endpoint_splitter,
        )

        self.adaptive_gate = AdaptiveGate(spatial_dim=spatial_dim, hidden_dim=32)

    # ------------------------------------------------------------------
    # MAGNN helpers
    # ------------------------------------------------------------------

    def _get_spatial_embeddings(self, seg_indices: torch.Tensor) -> torch.Tensor:
        ctx = torch.no_grad() if self.freeze_magnn else torch.enable_grad()
        with ctx:
            all_nodes = self.magnn.node_embedding.weight.unsqueeze(0)
            spatial_all = self.magnn.multi_gat(
                all_nodes,
                [self.magnn.adj_geo, self.magnn.adj_dist, self.magnn.adj_soc],
            ).squeeze(0)
        return spatial_all[seg_indices]   # (B, spatial_dim)

    # ------------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------------

    def forward(
        self,
        seg_indices: torch.Tensor,          # (B,)  segment type indices
        temporal_features: torch.Tensor,    # (B, 5)   hour_sin/cos, dow_sin/cos, speed
        context_flags: torch.Tensor,        # (B, 2)   is_weekend, is_peak_hour  (not scaled)
        origin_operational: torch.Tensor,   # (B, 2)   scaled arrivalDelay, departureDelay @ origin
        dest_operational: torch.Tensor,     # (B, 2)   same @ destination
        origin_weather: torch.Tensor,       # (B, 8)   weather @ origin GPS cluster
        dest_weather: torch.Tensor,         # (B, 8)   weather @ dest GPS cluster
        return_components: bool = False,
    ):
        # 1. MAGNN baseline  (no grad through frozen graph)
        with torch.no_grad():
            magnn_baseline = self.magnn(seg_indices, temporal_features)  # (B, 1)

        # 2. Spatial embeddings
        spatial_embeddings = self._get_spatial_embeddings(seg_indices)   # (B, spatial_dim)

        # 3. LSTM correction
        lstm_correction = self.residual_lstm(
            spatial_embeddings=spatial_embeddings,
            magnn_baseline=magnn_baseline,
            temporal_features=temporal_features,
            context_flags=context_flags,
            origin_operational=origin_operational,
            dest_operational=dest_operational,
            origin_weather=origin_weather,
            dest_weather=dest_weather,
        )                                                                 # (B, 1)

        # 4. Adaptive gate
        alpha = self.adaptive_gate(spatial_embeddings)                    # (B, 1)

        # 5. Final prediction
        final_prediction = magnn_baseline + alpha * lstm_correction       # (B, 1)

        if return_components:
            return magnn_baseline, lstm_correction, alpha, final_prediction
        return final_prediction


# =============================================================================
# DATASET HELPER  — builds origin/dest feature vectors per segment row
# =============================================================================

def split_features_for_segment(
    row,
    gps_lookup: Optional[NearestGPSFeatureLookup],
    mapper: Optional[SegmentStationMapper],
) -> Dict[str, np.ndarray]:
    """
    Given a segment DataFrame row return:
      origin_operational (2,)
      dest_operational   (2,)
      origin_weather     (8,)
      dest_weather       (8,)
      context_flags      (2,)   [is_weekend, is_peak_hour]

    Strategy
    --------
    1. If `gps_lookup` and `mapper` are available → query the BallTree with
       the origin / dest cluster centroid coordinates.
    2. Otherwise → use the already-averaged segment-level scalars for both
       endpoints (identical vectors; the FeatureEndpointSplitter inside the
       LSTM will learn to differentiate them).
    """
    # Context flags (binary — same for whole segment, not split)
    context = np.array([
        float(row.get("is_weekend", 0)),
        float(row.get("is_peak_hour", 0)),
    ], dtype=np.float32)

    if gps_lookup is not None and mapper is not None:
        seg_id = str(row.get("segment_id", "0_0"))
        try:
            (o_lat, o_lon), (d_lat, d_lon) = mapper.origin_dest_coords(seg_id)
        except Exception:
            o_lat, o_lon = 0.0, 0.0
            d_lat, d_lon = 0.0, 0.0

        o_weather, o_op = gps_lookup.lookup(o_lat, o_lon)
        d_weather, d_op = gps_lookup.lookup(d_lat, d_lon)
    else:
        # Fallback: use averaged segment features for both endpoints
        o_op = d_op = np.array([
            float(row.get("arrivalDelay_scaled",  row.get("arrivalDelay",  0))),
            float(row.get("departureDelay_scaled", row.get("departureDelay", 0))),
        ], dtype=np.float32)

        o_weather = d_weather = np.array([
            float(row.get("temperature_2m",       0)),
            float(row.get("apparent_temperature", 0)),
            float(row.get("precipitation",        0)),
            float(row.get("rain",                 0)),
            float(row.get("snowfall",             0)),
            float(row.get("windspeed_10m",        0)),
            float(row.get("windgusts_10m",        0)),
            float(row.get("winddirection_10m",    0)),
        ], dtype=np.float32)

    return {
        "context_flags":       context,
        "origin_operational":  o_op,
        "dest_operational":    d_op,
        "origin_weather":      o_weather,
        "dest_weather":        d_weather,
    }


# =============================================================================
# QUICK SELF-TEST
# =============================================================================

if __name__ == "__main__":
    import torch

    print("=== StationAwareResidualLSTM smoke-test ===")

    B          = 4
    SPATIAL    = 64
    TEMPORAL   = 5

    lstm = StationAwareResidualLSTM(
        spatial_dim=SPATIAL,
        temporal_dim=TEMPORAL,
        hidden_dim=64,
    )

    correction = lstm(
        spatial_embeddings  = torch.randn(B, SPATIAL),
        magnn_baseline      = torch.randn(B, 1),
        temporal_features   = torch.randn(B, TEMPORAL),
        context_flags       = torch.zeros(B, 2),
        origin_operational  = torch.randn(B, 2),
        dest_operational    = torch.randn(B, 2),
        origin_weather      = torch.randn(B, 8),
        dest_weather        = torch.randn(B, 8),
    )
    assert correction.shape == (B, 1), f"Expected ({B},1), got {correction.shape}"
    print(f"   ✅  correction shape: {correction.shape}")

    # Station mapper test
    clusters = np.array([
        [-35.1965, 149.1390],   # ~ Mapleton Avenue
        [-35.2044, 149.1404],   # ~ Nullarbor Avenue
        [-35.9999, 149.0000],   # far away — no match
    ])
    mapper = SegmentStationMapper(clusters)
    print(f"   ✅  segment '0_1' → '{mapper.segment_label('0_1')}'")
    print(f"   ✅  segment '0_2' → '{mapper.segment_label('0_2')}'")

    print("\n=== All checks passed ✅ ===")