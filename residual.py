

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


KNOWN_STATIONS: Dict[str, Tuple[float, float]] = {
    "Gungahlin Place":        (-35.1827, 149.1328),
    "Manning Clark Crescent": (-35.1895, 149.1362),
    "Mapleton Avenue":        (-35.1965, 149.1390),
    "Nullarbor Avenue":       (-35.2044, 149.1404),
    "Well Station Drive":     (-35.2120, 149.1402),
    "Sandford Street":        (-35.2228, 149.1394),
    "EPIC and Racecourse":    (-35.2328, 149.1376),
    "Phillip Avenue":         (-35.2439, 149.1360),
    "Swinden Street":         (-35.2537, 149.1342),
    "Dickson Interchange":    (-35.2620, 149.1320),
    "Macarthur Avenue":       (-35.2719, 149.1296),
    "Ipima Street":           (-35.2802, 149.1272),
    "Elouera Street":         (-35.2889, 149.1248),
    "Alinga Street":          (-35.2797, 149.1310),
}

WEATHER_COLS = [
    "temperature_2m", "apparent_temperature", "precipitation",
    "rain", "snowfall", "windspeed_10m", "windgusts_10m",
    "winddirection_10m",
]
OPERATIONAL_COLS = ["arrivalDelay", "departureDelay", "dwellTime_sec"]

CONTEXT_FLAG_COLS = ["is_weekend", "is_peak_hour", "is_slowdown", "is_congested"]


class SegmentStationMapper:

    STATION_NAMES = list(KNOWN_STATIONS.keys())

    def __init__(
        self,
        clusters: np.ndarray,
        known_stops: Optional[Dict[str, Tuple[float, float]]] = None,
        match_radius_m: float = 150.0,
    ):
        self.clusters   = np.asarray(clusters, dtype=float)
        self.n_clusters = len(self.clusters)
        self.cluster_to_station: Dict[int, str] = {}

        all_stops = dict(KNOWN_STATIONS)
        if known_stops:
            all_stops.update(known_stops)

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
        try:
            o_str, d_str = segment_id.split("_")
            o, d = int(o_str), int(d_str)
        except ValueError:
            return segment_id
        o_name = self.cluster_to_station.get(o, f"Cluster {o}")
        d_name = self.cluster_to_station.get(d, f"Cluster {d}")
        return f"{o_name} → {d_name}"

    def cluster_coords(self, cluster_idx: int) -> Tuple[float, float]:
        return tuple(self.clusters[cluster_idx])

    def is_named(self, cluster_idx: int) -> bool:
        return cluster_idx in self.cluster_to_station

    def nearest_named_cluster(self, cluster_idx: int) -> int:
        """
        For an unnamed cluster, return the index of the nearest cluster
        that IS matched to a known station.  Used to borrow operational
        and weather features from the nearest real station instead of
        getting GPS pings from the middle of nowhere.

        Example: Macarthur Av (cl.6) → delay cl.18 (unnamed)
          cl.18 has no named station → nearest named = cl.1 (Dickson)
          → borrow Dickson's GPS coords for the feature lookup
        """
        if self.is_named(cluster_idx):
            return cluster_idx
        named_indices = [i for i in range(self.n_clusters)
                         if i in self.cluster_to_station]
        if not named_indices:
            return cluster_idx
        c_lat, c_lon = self.clusters[cluster_idx]
        best_idx, best_dist = cluster_idx, float("inf")
        for ni in named_indices:
            n_lat, n_lon = self.clusters[ni]
            d = _haversine_m(c_lat, c_lon, n_lat, n_lon)
            if d < best_dist:
                best_dist, best_idx = d, ni
        return best_idx

    def origin_dest_coords(
        self, segment_id: str, fallback_unnamed: bool = True
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Return (origin_lat_lon, dest_lat_lon) for a segment.

        fallback_unnamed=True (default):
          If an endpoint cluster is unnamed (delay cluster / transit cluster
          with no matched station), use the nearest NAMED cluster's coords
          instead.  This ensures the GPS feature lookup hits real station
          pings rather than random in-transit GPS noise.

          Example: segment "6_18"
            cl.6  = Macarthur Avenue  (named)   → use cl.6 coords  ✓
            cl.18 = delay cluster     (unnamed) → fallback to nearest named
                    nearest named = cl.1 (Dickson)  → use cl.1 coords
        """
        o, d = map(int, segment_id.split("_"))
        if fallback_unnamed:
            o_resolved = self.nearest_named_cluster(o)
            d_resolved = self.nearest_named_cluster(d)
        else:
            o_resolved, d_resolved = o, d
        return self.cluster_coords(o_resolved), self.cluster_coords(d_resolved)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R   = 6_371_000.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ  = math.radians(lat2 - lat1)
    Δλ  = math.radians(lon2 - lon1)
    a   = (math.sin(Δφ / 2) ** 2
           + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class NearestGPSFeatureLookup:
    def __init__(self, raw_df, weather_cols=None, operational_cols=None):
        import pandas as pd

        self.weather_cols     = weather_cols     or WEATHER_COLS
        self.operational_cols = operational_cols or OPERATIONAL_COLS

        if "is_gps_valid" in raw_df.columns:
            df = raw_df[raw_df["is_gps_valid"] == 1].copy()
        else:
            df = raw_df.copy()

        df = df.dropna(subset=["latitude", "longitude"])
        self.lats = df["latitude"].values
        self.lons  = df["longitude"].values

        ts_col = next((c for c in ["timestamp", "departureTime", "arrivalTime"]
                       if c in df.columns), None)
        if ts_col is not None:
            parsed = pd.to_datetime(df[ts_col], errors="coerce")
            self.timestamps_ns = parsed.values.astype("datetime64[ns]").view("int64")
        else:
            self.timestamps_ns = None

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

        from sklearn.neighbors import BallTree
        coords_rad  = np.radians(np.stack([self.lats, self.lons], axis=1))
        self.tree   = BallTree(coords_rad, metric="haversine")
        print(f"   NearestGPSFeatureLookup: indexed {len(df):,} valid GPS pings"
              + (f"  (timestamp col: '{ts_col}')" if ts_col else "  (no timestamp — geo-only)"))

    def lookup(
        self,
        lat:              float,
        lon:              float,
        k:                int   = 5,
        timestamp         = None,
        time_window_sec:  int   = 3600,
    ) -> Tuple[np.ndarray, np.ndarray]:

        import pandas as pd

        q = np.radians([[lat, lon]])

        if timestamp is not None and self.timestamps_ns is not None:
            k_geo = min(k * 20, len(self.lats))
            _, idx = self.tree.query(q, k=k_geo)
            idxs   = idx[0]

            try:
                ts_ns = int(np.datetime64(pd.Timestamp(timestamp), "ns").view("int64"))
            except Exception:
                ts_ns = None

            if ts_ns is not None:
                window_ns = int(time_window_sec) * 1_000_000_000
                time_mask = np.abs(self.timestamps_ns[idxs] - ts_ns) <= window_ns
                time_mask &= self.timestamps_ns[idxs] > 0
                time_idxs  = idxs[time_mask]

                if len(time_idxs) >= 1:
                    use = time_idxs[:k]
                else:
                    use = idxs[:k]
            else:
                use = idxs[:k]
        else:
            _, idx = self.tree.query(q, k=min(k, len(self.lats)))
            use    = idx[0]

        weather_vec = self.weather_matrix[use].mean(axis=0)
        op_vec      = self.operational_matrix[use].mean(axis=0)
        return weather_vec.copy(), op_vec.copy()


class FeatureEndpointSplitter(nn.Module):

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


class AdaptiveGate(nn.Module):

    OP_DIM      = len(OPERATIONAL_COLS)
    WEATHER_DIM = len(WEATHER_COLS)
    CTX_DIM     = len(CONTEXT_FLAG_COLS)

    def __init__(
        self,
        spatial_dim: int,
        hidden_dim:  int = 64,
    ):
        super().__init__()
        in_dim = spatial_dim + self.OP_DIM + self.WEATHER_DIM + self.CTX_DIM + 2
        self.gate_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        nn.init.xavier_uniform_(self.gate_net[5].weight, gain=0.1)
        nn.init.zeros_(self.gate_net[5].bias)

    def forward(
        self,
        spatial_embedding:  torch.Tensor,
        operational:        torch.Tensor,
        weather:            torch.Tensor,
        context_flags:      Optional[torch.Tensor] = None,
        baseline_deviation: Optional[torch.Tensor] = None,
        cum_magnn_error:    Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = [
            spatial_embedding,
            operational,
            weather,
        ]
        if context_flags is not None:
            parts.append(context_flags)
        else:
            parts.append(torch.zeros(
                spatial_embedding.size(0), self.CTX_DIM,
                device=spatial_embedding.device, dtype=spatial_embedding.dtype))
        if baseline_deviation is not None:
            parts.append(baseline_deviation)
        else:
            parts.append(torch.zeros(
                spatial_embedding.size(0), 1,
                device=spatial_embedding.device, dtype=spatial_embedding.dtype))
        if cum_magnn_error is not None:
            parts.append(cum_magnn_error)
        else:
            parts.append(torch.zeros(
                spatial_embedding.size(0), 1,
                device=spatial_embedding.device, dtype=spatial_embedding.dtype))
        return self.gate_net(torch.cat(parts, dim=1))

class GlobalTemporalAttention(nn.Module):
    def __init__(self, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.scale   = math.sqrt(feature_dim)
        self.W_Q     = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_K     = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_V     = nn.Linear(feature_dim, feature_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor):
        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)
        scores   = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn     = self.dropout(F.softmax(scores, dim=-1))
        out      = self.norm(torch.matmul(attn, V) + x)
        return out, attn


class StationAwareResidualLSTM(nn.Module):


    OP_DIM      = len(OPERATIONAL_COLS)
    WEATHER_DIM = len(WEATHER_COLS)
    CONTEXT_DIM = len(CONTEXT_FLAG_COLS)

    def __init__(
        self,
        spatial_dim:  int,
        temporal_dim: int = 5,
        hidden_dim:   int = 128,
        n_layers:     int = 2,
        dropout:      float = 0.1,
        use_endpoint_splitter: bool = False,
    ):
        super().__init__()

        self.spatial_dim = spatial_dim
        self.hidden_dim  = hidden_dim
        self.use_splitter = use_endpoint_splitter

        if use_endpoint_splitter:
            self.splitter = FeatureEndpointSplitter(
                feat_dim=self.OP_DIM + self.WEATHER_DIM
            )

        lstm_input_dim = (
            spatial_dim
            + 1
            + 1
            + self.CONTEXT_DIM
            + temporal_dim
            + self.OP_DIM
            + self.WEATHER_DIM
        )

        print(f"   StationAwareResidualLSTM input breakdown:")
        print(f"     spatial_embedding     : {spatial_dim}")
        print(f"     magnn_baseline        : 1  (MAGNN prediction — thing to correct)")
        print(f"     cumulative_magnn_error: 1  (running MAGNN error from prior steps in trip)")
        print(f"     context_flags         : {self.CONTEXT_DIM}  [is_weekend, is_peak_hour, has_prev_stop, is_delayed]")
        print(f"     temporal cyclical     : {temporal_dim}  (hour_sin/cos, dow_sin/cos, speed)")
        print(f"     operational           : {self.OP_DIM}  {OPERATIONAL_COLS}  (direct from segment row)")
        print(f"     weather               : {self.WEATHER_DIM}  (direct from segment row — no GPS lookup)")
        print(f"     ─────────────────────────────────")
        print(f"     TOTAL per step        : {lstm_input_dim}")
        print(f"     LSTM layers           : {n_layers}")

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        for k in range(n_layers):
            nn.init.orthogonal_(getattr(self.lstm, f'weight_hh_l{k}'))
            nn.init.xavier_uniform_(getattr(self.lstm, f'weight_ih_l{k}'))

        self.attention = GlobalTemporalAttention(hidden_dim, dropout)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),         nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        for layer in self.fusion:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.fusion[-1].weight, gain=0.5)
        nn.init.zeros_(self.fusion[-1].bias)

        self.last_hidden: Optional[torch.Tensor] = None

    def forward(
        self,
        spatial_embeddings:  torch.Tensor,
        magnn_baseline:      torch.Tensor,
        temporal_features:   torch.Tensor,
        context_flags:       torch.Tensor,
        origin_operational:  torch.Tensor,
        dest_operational:    torch.Tensor,
        origin_weather:      torch.Tensor,
        dest_weather:        torch.Tensor,
        cumulative_magnn_error: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        def _ensure_3d(t: torch.Tensor, T: int) -> torch.Tensor:
            if t.dim() == 3:
                return t
            elif t.dim() == 2:
                return t.unsqueeze(1).expand(-1, T, -1)
            else:
                return t.unsqueeze(0).unsqueeze(0).expand(1, T, -1)

        sp  = _ensure_3d(spatial_embeddings, 1)
        if sp.dim() == 3:
            B, T, _ = sp.shape
        else:
            B, T = sp.shape[0], 1

        sp  = _ensure_3d(spatial_embeddings, T)

        if magnn_baseline.dim() == 2:
            bl = magnn_baseline.unsqueeze(1).expand(B, T, 1)
        else:
            bl = magnn_baseline

        trip_bl_mean = bl.mean(dim=1, keepdim=True)
        bl_deviation = bl - trip_bl_mean
        cum_err = torch.cat([
            torch.zeros(B, 1, 1, device=bl.device, dtype=bl.dtype),
            bl_deviation[:, :-1, :].cumsum(dim=1)
        ], dim=1).clamp(-3.0, 3.0)

        tf   = _ensure_3d(temporal_features,   T)
        cf   = _ensure_3d(context_flags,        T)
        o_op = _ensure_3d(origin_operational,   T)
        o_wx = _ensure_3d(origin_weather,       T)

        combined = torch.cat([sp, bl, cum_err, cf, tf, o_op, o_wx], dim=2)

        lstm_out, _ = self.lstm(combined)
        attn_out, _ = self.attention(lstm_out)

        self.last_hidden = attn_out[:, -1, :]

        B, T, _ = attn_out.shape
        if T > 1:
            correction = self.fusion(attn_out.reshape(B * T, -1)).reshape(B, T, 1)
        else:
            correction = self.fusion(attn_out[:, -1, :])
        return correction


class MAGNN_LSTM_Residual(nn.Module):


    def __init__(
        self,
        magnn_model,
        spatial_dim:   int,
        station_mapper: Optional[SegmentStationMapper] = None,
        freeze_magnn:  bool  = True,
        lstm_hidden:   int   = 128,
        lstm_layers:   int   = 2,
        dropout:       float = 0.2,
        temporal_dim:  int   = 5,
        use_endpoint_splitter: bool = False,
    ):
        super().__init__()

        self.magnn          = magnn_model
        self.freeze_magnn   = freeze_magnn
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

        self.adaptive_gate = AdaptiveGate(
            spatial_dim=spatial_dim,
            hidden_dim=64,
        )

        num_seg = magnn_model.node_embedding.num_embeddings
        self.register_buffer(
            'segment_mean_sc',
            torch.zeros(num_seg, dtype=torch.float32)
        )

    def set_segment_stats(self, seg_medians_scaled: np.ndarray):

        pass


    def _get_spatial_embeddings(self, seg_indices: torch.Tensor) -> torch.Tensor:

        ctx = torch.no_grad() if self.freeze_magnn else torch.enable_grad()
        with ctx:
            all_nodes   = self.magnn.node_embedding.weight.unsqueeze(0)
            spatial_all = self.magnn.multi_gat(
                all_nodes,
                [self.magnn.adj_geo, self.magnn.adj_dist, self.magnn.adj_soc],
            ).squeeze(0)
        return spatial_all[seg_indices]


    def forward(
        self,
        seg_indices:        torch.Tensor,
        temporal_features:  torch.Tensor,
        context_flags:      torch.Tensor,
        origin_operational: torch.Tensor,
        dest_operational:   torch.Tensor,
        origin_weather:     torch.Tensor,
        dest_weather:       torch.Tensor,
        return_components:  bool = False,
        cumulative_magnn_error: Optional[torch.Tensor] = None,
    ):
        sequential = seg_indices.dim() == 2

        spatial_embeddings = self._get_spatial_embeddings(seg_indices)

        if sequential:
            B, T = seg_indices.shape
            flat_seg  = seg_indices.reshape(B * T)
            flat_temp = temporal_features.reshape(B * T, -1)
            with torch.no_grad():
                flat_bl = self.magnn(flat_seg, flat_temp)
            magnn_baseline = flat_bl.reshape(B, T, 1)
        else:
            with torch.no_grad():
                magnn_baseline = self.magnn(seg_indices, temporal_features)

        lstm_raw = self.residual_lstm(
            spatial_embeddings=spatial_embeddings,
            magnn_baseline=magnn_baseline if sequential else magnn_baseline,
            temporal_features=temporal_features,
            context_flags=context_flags,
            origin_operational=origin_operational,
            dest_operational=dest_operational,
            origin_weather=origin_weather,
            dest_weather=dest_weather,
            cumulative_magnn_error=cumulative_magnn_error,
        )
        if sequential:
            lstm_pred = magnn_baseline + lstm_raw

            sp   = spatial_embeddings

            correction_seq = lstm_raw
            cum_corr = torch.cat([
                torch.zeros(B, 1, 1, device=magnn_baseline.device),
                correction_seq[:, :-1, :].cumsum(dim=1)
            ], dim=1).clamp(-3.0, 3.0)

            trip_bl_mean = (magnn_baseline * torch.ones_like(magnn_baseline)
                            ).mean(dim=1, keepdim=True)
            bl_dev_seq   = magnn_baseline - trip_bl_mean

            alpha_steps = []
            for t in range(T):
                o_op_t = origin_operational[:, t, :] if origin_operational.dim()==3 else origin_operational
                o_wx_t = origin_weather[:, t, :] if origin_weather.dim()==3 else origin_weather
                ctx_t  = context_flags[:, t, :] if context_flags.dim()==3 else context_flags
                a = self.adaptive_gate(
                    sp[:, t, :],
                    o_op_t,
                    o_wx_t,
                    context_flags=ctx_t,
                    baseline_deviation=bl_dev_seq[:, t, :],
                    cum_magnn_error=cum_corr[:, t, :],
                )
                alpha_steps.append(a)
            alpha = torch.stack(alpha_steps, dim=1)

            bl_for_blend = magnn_baseline
            final_prediction = (1.0 - alpha) * bl_for_blend + alpha * lstm_pred
            self._last_correction = (lstm_pred - magnn_baseline)[:, -1, :]

            if return_components:
                return magnn_baseline, lstm_pred, alpha, final_prediction
            return final_prediction

        else:
            lstm_pred = magnn_baseline + lstm_raw

            bl_dev = magnn_baseline

            alpha = self.adaptive_gate(
                spatial_embeddings,
                origin_operational,
                origin_weather,
                context_flags=context_flags,
                baseline_deviation=bl_dev,
                cum_magnn_error=cumulative_magnn_error,
            )
            final_prediction = (1.0 - alpha) * magnn_baseline + alpha * lstm_pred
            self._last_correction = lstm_pred - magnn_baseline

            if return_components:
                return magnn_baseline, lstm_pred, alpha, final_prediction
            return final_prediction


def split_features_for_segment(
    row,
    gps_lookup:          Optional[NearestGPSFeatureLookup],
    mapper:              Optional[SegmentStationMapper],
    operational_scaler=None,
    weather_scaler=None,
    time_window_sec: int = 3600,
) -> Dict[str, np.ndarray]:

    arr_delay_raw = float(row.get("arrivalDelay",   row.get("arrivalDelay_scaled",  0)) or 0)
    dep_delay_raw = float(row.get("departureDelay", row.get("departureDelay_scaled", 0)) or 0)
    is_delayed_flag = int(abs(arr_delay_raw) > 20 or abs(dep_delay_raw) > 20)
    context = np.array([
        float(row.get("is_weekend",    0)),
        float(row.get("is_peak_hour",  0)),
        float(row.get("has_prev_stop", row.get("is_slowdown", 0))),
        float(row.get("is_delayed",    is_delayed_flag)),
    ], dtype=np.float32)

    departure_ts = (
        row.get("departure_time")
        or row.get("departureTime")
        or row.get("timestamp")
        or None
    )

    if gps_lookup is not None and mapper is not None:
        seg_id = str(row.get("segment_id", "0_0"))
        try:
            (o_lat, o_lon), (d_lat, d_lon) = mapper.origin_dest_coords(seg_id)
        except Exception:
            o_lat, o_lon = 0.0, 0.0
            d_lat, d_lon = 0.0, 0.0

        o_weather_raw, o_op_raw = gps_lookup.lookup(
            o_lat, o_lon,
            timestamp=departure_ts,
            time_window_sec=time_window_sec,
        )
        d_weather_raw, d_op_raw = gps_lookup.lookup(
            d_lat, d_lon,
            timestamp=departure_ts,
            time_window_sec=time_window_sec,
        )

        dwell_s = float(row.get("dwellTime_sec", 0) or 0)
        if operational_scaler is not None:
            try:
                o_op_scaled = operational_scaler.transform(o_op_raw[:2].reshape(1, -1))[0].astype(np.float32)
                d_op_scaled = operational_scaler.transform(d_op_raw[:2].reshape(1, -1))[0].astype(np.float32)
            except Exception:
                o_op_scaled = o_op_raw[:2].astype(np.float32)
                d_op_scaled = d_op_raw[:2].astype(np.float32)
        else:
            o_op_scaled = o_op_raw[:2].astype(np.float32)
            d_op_scaled = d_op_raw[:2].astype(np.float32)
        o_op = np.append(o_op_scaled, dwell_s).astype(np.float32)
        d_op = np.append(d_op_scaled, dwell_s).astype(np.float32)

        if weather_scaler is not None:
            try:
                o_weather = weather_scaler.transform(o_weather_raw[:8].reshape(1, -1))[0].astype(np.float32)
                d_weather = weather_scaler.transform(d_weather_raw[:8].reshape(1, -1))[0].astype(np.float32)
            except Exception:
                o_weather = o_weather_raw[:8].astype(np.float32)
                d_weather = d_weather_raw[:8].astype(np.float32)
        else:
            o_weather = o_weather_raw[:8].astype(np.float32)
            d_weather = d_weather_raw[:8].astype(np.float32)

    else:
        o_op = d_op = np.array([
            float(row.get("arrivalDelay_scaled",  row.get("arrivalDelay",  0))),
            float(row.get("departureDelay_scaled", row.get("departureDelay", 0))),
            float(row.get("dwellTime_sec", 0)),
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


if __name__ == "__main__":
    print("=== StationAwareResidualLSTM v2 smoke-test ===")

    B        = 4
    T_HIST   = 5
    SPATIAL  = 64
    TEMPORAL = 5
    CONTEXT  = 4

    lstm = StationAwareResidualLSTM(
        spatial_dim=SPATIAL,
        temporal_dim=TEMPORAL,
        hidden_dim=64,
        n_layers=2,
    )

    correction = lstm(
        spatial_embeddings  = torch.randn(B, T_HIST, SPATIAL),
        magnn_baseline      = torch.randn(B, 1),
        temporal_features   = torch.randn(B, T_HIST, TEMPORAL),
        context_flags       = torch.zeros(B, T_HIST, CONTEXT),
        origin_operational  = torch.randn(B, T_HIST, 2),
        dest_operational    = torch.randn(B, T_HIST, 2),
        origin_weather      = torch.randn(B, T_HIST, 8),
        dest_weather        = torch.randn(B, T_HIST, 8),
    )
    assert correction.shape == (B, 1), f"Expected ({B},1), got {correction.shape}"
    assert lstm.last_hidden is not None and lstm.last_hidden.shape == (B, 64)
    print(f"   ✅  correction shape: {correction.shape}")
    print(f"   ✅  last_hidden shape: {lstm.last_hidden.shape}")

    correction_legacy = lstm(
        spatial_embeddings  = torch.randn(B, SPATIAL),
        magnn_baseline      = torch.randn(B, 1),
        temporal_features   = torch.randn(B, TEMPORAL),
        context_flags       = torch.zeros(B, CONTEXT),
        origin_operational  = torch.randn(B, 2),
        dest_operational    = torch.randn(B, 2),
        origin_weather      = torch.randn(B, 8),
        dest_weather        = torch.randn(B, 8),
    )
    assert correction_legacy.shape == (B, 1)
    print(f"   ✅  legacy single-step correction shape: {correction_legacy.shape}")

    gate = AdaptiveGate(spatial_dim=SPATIAL)
    alpha = gate(
        torch.randn(B, SPATIAL),
        torch.randn(B, 2),
        torch.randn(B, 2),
        torch.randn(B, 8),
        torch.randn(B, 8),
    )
    assert alpha.shape == (B, 1)
    print(f"   ✅  AdaptiveGate alpha shape: {alpha.shape}")

    clusters = np.array([
        [-35.1965, 149.1390],
        [-35.2044, 149.1404],
        [-35.9999, 149.0000],
    ])
    mapper = SegmentStationMapper(clusters)
    print(f"   ✅  segment '0_1' → '{mapper.segment_label('0_1')}'")
    print(f"   ✅  segment '0_2' → '{mapper.segment_label('0_2')}'")

    print("\n=== All checks passed ✅ ===")
