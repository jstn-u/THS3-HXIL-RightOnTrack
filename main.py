"""
main.py
=======

THREE MODEL COMPARISON:
   1. MAGNN (baseline)
   2. MAGNN-LSTM-Residual (station-aware residual learning + adaptive gating)
   3. HierarchicalTripPredictor (local segment + global trip, cumulative correction)


Usage:
    python main.py                      # defaults (sample from config, hdbscan)
    python main.py 0.1                  # 10% sample, hdbscan
    python main.py 1.0 knn             # 100% sample, KNN clustering
    python main.py 0.5 dbscan          # 50% sample, DBSCAN clustering
    python main.py --compare-all        # Compare all three models
    python main.py --ablation           # Feature ablation studies
    python main.py 0.1 knn --compare-all

Supported clustering methods: hdbscan, dbscan, knn, gmm, kmeans
"""

import sys
import importlib

VALID_METHODS = {'hdbscan', 'dbscan', 'knn', 'gmm', 'kmeans'}

_sample_fraction_cli = None
_cluster_method      = 'hdbscan'
_mode                = None

for _arg in sys.argv[1:]:
    if _arg in ('--compare-all', '--ablation'):
        _mode = _arg
    elif _arg.lower() in VALID_METHODS:
        _cluster_method = _arg.lower()
    else:
        try:
            _sample_fraction_cli = float(_arg)
        except ValueError:
            print(f"Unknown argument '{_arg}' — ignored.")
            print(f"Valid methods: {', '.join(sorted(VALID_METHODS))}")
            print(f"Valid flags:   --compare-all, --ablation")
            print(f"Or pass a number for sample fraction (e.g. 0.1)")

_cluster_module = importlib.import_module(f'cluster_{_cluster_method}')
event_driven_clustering_fixed = _cluster_module.event_driven_clustering_fixed

ALGORITHM_NAME = _cluster_method.upper()

from config import Config, DEVICE, print_section, haversine_meters
from data_loader import (load_data_fixed, load_train_test_val_data_fixed,
                         get_known_stops)
from segments import (build_segments_fixed, build_adjacency_matrices_fixed,
                      aggregate_segments_into_paths)
from visualizations import (plot_clusters, plot_segments,
                            plot_segment_statistics)
from model import (SegmentDataset, masked_collate_fn,
                   train_magtte, SimpleMLP, train_simple,
                   EnhancedSegmentDataset, enhanced_collate_fn,
                   PathDataset, path_collate_fn,
                   TripDataset, trip_collate_fn,
                   train_trip_level_predictor,
                   DwellPredictor,
                   MAGTTE)
from residual import (MAGNN_LSTM_Residual,
                      NearestGPSFeatureLookup,
                      SegmentStationMapper,
                      split_features_for_segment)

import numpy as np
import pandas as pd
import os
import json
import warnings
import time
from datetime import datetime
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats as scipy_stats
import torch

warnings.filterwarnings('ignore')



def _attach_gps_lookup(dataset: EnhancedSegmentDataset,
                       raw_df,
                       clusters,
                       known_stops: dict):
    if not hasattr(dataset, 'gps_lookup') or dataset.gps_lookup is None:
        dataset.gps_lookup     = NearestGPSFeatureLookup(raw_df)
        dataset.station_mapper = SegmentStationMapper(
            np.asarray(clusters), known_stops=known_stops
        )
    return dataset

def _make_enhanced_loaders(train_segments, val_segments, test_segments,
                            segment_types, config,
                            train_df, clusters, known_stops,
                            collate_fn=None):

    if collate_fn is None:
        collate_fn = enhanced_collate_fn

    train_ds = EnhancedSegmentDataset(train_segments, segment_types,
                                      fit_scalers=True)
    val_ds   = EnhancedSegmentDataset(val_segments, segment_types,
                                      fit_scalers=False,
                                      target_scaler=train_ds.target_scaler,
                                      speed_scaler=train_ds.speed_scaler,
                                      operational_scaler=train_ds.operational_scaler,
                                      weather_scaler=train_ds.weather_scaler)
    test_ds  = EnhancedSegmentDataset(test_segments, segment_types,
                                      fit_scalers=False,
                                      target_scaler=train_ds.target_scaler,
                                      speed_scaler=train_ds.speed_scaler,
                                      operational_scaler=train_ds.operational_scaler,
                                      weather_scaler=train_ds.weather_scaler)

    # Attach GPS lookup to all three datasets (val/test reuse train lookup)
    _attach_gps_lookup(train_ds, train_df, clusters, known_stops)
    val_ds.gps_lookup      = train_ds.gps_lookup
    val_ds.station_mapper  = train_ds.station_mapper
    test_ds.gps_lookup     = train_ds.gps_lookup
    test_ds.station_mapper = train_ds.station_mapper

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True,  num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size,
                              shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size,
                              shuffle=False, num_workers=0, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, train_ds


def _make_trip_loaders(train_segments, val_segments, test_segments,
                       segment_types, config,
                       train_df, clusters, known_stops,
                       max_trip_length: int = 15):

    gps_lookup     = NearestGPSFeatureLookup(train_df)
    station_mapper = SegmentStationMapper(
        np.asarray(clusters), known_stops=known_stops)

    train_ds = TripDataset(train_segments, segment_types,
                           max_trip_length=max_trip_length,
                           fit_scalers=True)
    val_ds   = TripDataset(val_segments, segment_types,
                           max_trip_length=max_trip_length,
                           fit_scalers=False,
                           target_scaler=train_ds.target_scaler,
                           speed_scaler=train_ds.speed_scaler,
                           operational_scaler=train_ds.operational_scaler,
                           weather_scaler=train_ds.weather_scaler,
                           trip_target_scaler=train_ds.trip_target_scaler,
                           day_target_scaler=train_ds.day_target_scaler)
    test_ds  = TripDataset(test_segments, segment_types,
                           max_trip_length=max_trip_length,
                           fit_scalers=False,
                           target_scaler=train_ds.target_scaler,
                           speed_scaler=train_ds.speed_scaler,
                           operational_scaler=train_ds.operational_scaler,
                           weather_scaler=train_ds.weather_scaler,
                           trip_target_scaler=train_ds.trip_target_scaler,
                           day_target_scaler=train_ds.day_target_scaler)

    for ds in (train_ds, val_ds, test_ds):
        ds.gps_lookup     = gps_lookup
        ds.station_mapper = station_mapper

    trip_batch = max(4, config.batch_size // 4)

    train_loader = DataLoader(train_ds, batch_size=trip_batch,
                              shuffle=True,  num_workers=0,
                              collate_fn=trip_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=trip_batch,
                              shuffle=False, num_workers=0,
                              collate_fn=trip_collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=trip_batch,
                              shuffle=False, num_workers=0,
                              collate_fn=trip_collate_fn)

    return train_loader, val_loader, test_loader, train_ds

def _unpack_residual_trip_batch(batch, device):
    (seg_idx, temporal, context_flags,
     origin_op, dest_op,
     origin_wx, dest_wx,
     seg_targets, _trip_target,
     _cum_prior, _day_target, _day_rank,
     lengths, mask) = batch

    return (
        seg_idx.to(device),        # (B, T)
        temporal.to(device),       # (B, T, 5)
        context_flags.to(device),  # (B, T, 4)
        origin_op.to(device),      # (B, T, 3)
        dest_op.to(device),        # (B, T, 3)
        origin_wx.to(device),      # (B, T, 8)
        dest_wx.to(device),        # (B, T, 8)
        seg_targets.to(device),    # (B, T, 1)
        lengths.to(device),        # (B,)
        mask.to(device),           # (B, T)
    )

def train_magnn_lstm_residual(train_loader, val_loader, test_loader,
                              adj_geo, adj_dist, adj_soc,
                              segment_types, scaler,
                              output_folder, device, cfg,
                              clusters=None,
                              known_stops=None,
                              pretrained_magnn_path=None,
                              freeze_magnn=True):
    print_section("MAGNN-LSTM-RESIDUAL TRAINING  (Trip-Sequential)")
    num_segments = len(segment_types)

    os.makedirs(output_folder, exist_ok=True)

    # ── Build MAGNN base ──────────────────────────────────────────────────────
    magnn_base = MAGTTE(num_segments,
                        cfg.n_heads, cfg.node_embed_dim, cfg.gat_hidden,
                        cfg.lstm_hidden, cfg.historical_dim,
                        cfg.dropout).to(device)
    magnn_base.set_adjacency_matrices(adj_geo, adj_dist, adj_soc)

    _NODE_KEYS = {
        "segment_mean_sc",
        "magnn.adj_geo", "magnn.adj_dist", "magnn.adj_soc",
        "magnn.node_embedding.weight",
        "magnn.historical_embed.embedding.weight",
    }

    if pretrained_magnn_path and os.path.exists(pretrained_magnn_path):
        magnn_base.load_state_dict(
            torch.load(pretrained_magnn_path, map_location=device))
        print(f"Loaded pre-trained MAGNN from {pretrained_magnn_path}")

    station_mapper = None
    if clusters is not None:
        station_mapper = SegmentStationMapper(
            np.asarray(clusters), known_stops=known_stops or {})

    travel_model = MAGNN_LSTM_Residual(
        magnn_base,
        spatial_dim=cfg.gat_hidden,
        station_mapper=station_mapper,
        freeze_magnn=freeze_magnn,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.2,
        temporal_dim=5,
    ).to(device)

    try:
        ds  = train_loader.dataset
        sdf = ds.segments_df.copy() if hasattr(ds, 'segments_df') else None
        if sdf is not None and 'seg_type_idx' in sdf.columns and 'duration_scaled' in sdf.columns:
            medians   = sdf.groupby('seg_type_idx')['duration_scaled'].median()
            seg_means = np.zeros(num_segments, dtype=np.float32)
            for idx, m in medians.items():
                if 0 <= int(idx) < num_segments:
                    seg_means[int(idx)] = float(m)
            travel_model.set_segment_stats(seg_means)
            print(f"segment_mean_sc set from TripDataset segments_df")
    except Exception as ex:
        print(f"segment_mean_sc: could not compute ({ex}), using zeros")

    # ── Dwell model ───────────────────────────────────────────────────────────
    dwell_model = DwellPredictor(
        num_segments=num_segments,
        embed_dim=16,
        dropout=0.2,
    ).to(device)

    n_travel = sum(p.numel() for p in travel_model.parameters() if p.requires_grad)
    n_dwell  = sum(p.numel() for p in dwell_model.parameters()  if p.requires_grad)
    print(f"   MAGNN frozen: {freeze_magnn}")
    print(f"   Travel model trainable params: {n_travel:,}")
    print(f"   Dwell  model trainable params: {n_dwell:,}")

    lr = getattr(cfg, 'residual_learning_rate', 5e-4)
    wd = getattr(cfg, 'lstm_weight_decay',      1e-6)
    opt_travel = optim.Adam(
        filter(lambda p: p.requires_grad, travel_model.parameters()),
        lr=lr, weight_decay=wd)
    opt_dwell = optim.Adam(dwell_model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched_travel = optim.lr_scheduler.ReduceLROnPlateau(
        opt_travel, mode='min', factor=0.5, patience=5)
    sched_dwell  = optim.lr_scheduler.ReduceLROnPlateau(
        opt_dwell,  mode='min', factor=0.5, patience=5)

    best_val   = float('inf')
    patience_c = 0
    best_ckpt  = os.path.join(output_folder, 'magnn_lstm_residual_best.pth')

    _use_amp = hasattr(device, 'type') and device.type == 'cuda'
    if _use_amp:
        from torch.cuda.amp import autocast, GradScaler
        _amp_ctx       = autocast
        _scaler_travel = GradScaler()
    else:
        _amp_ctx       = None   # falsy → nullcontext used inline
        _scaler_travel = None   # falsy → plain .backward() used inline

    print(f"   LR={lr}  WD={wd}  AMP={'on (CUDA)' if _use_amp else 'off'}\n")

    for epoch in range(1, cfg.n_epochs + 1):
        travel_model.train()
        dwell_model.train()
        t_loss = t_alpha = t_corr = t_n = 0.0
        t_alpha_min = 1.0
        t_alpha_max = 0.0
        d_loss = d_n = 0.0

        for batch in train_loader:
            (seg_idx, temporal, ctx,
             o_op, d_op, o_wx, d_wx,
             seg_tgts, lengths, mask) = _unpack_residual_trip_batch(batch, device)

            B, T = seg_idx.shape
            cum_err = torch.zeros(B, 1, device=device)

            _fwd_ctx = (_amp_ctx() if _amp_ctx else __import__('contextlib').nullcontext())
            with _fwd_ctx:
                bl, lstm_pred, alpha, pred = travel_model(
                    seg_idx, temporal,
                    context_flags=ctx,
                    origin_operational=o_op,
                    dest_operational=d_op,
                    origin_weather=o_wx,
                    dest_weather=d_wx,
                    return_components=True,
                    cumulative_magnn_error=cum_err,
                )
                v = mask.float().unsqueeze(-1)   # (B, T, 1)
                n_valid = v.sum().clamp(min=1)

                loss_pred   = (F.smooth_l1_loss(pred,      seg_tgts, reduction='none') * v).sum() / n_valid
                loss_direct = (F.smooth_l1_loss(lstm_pred, seg_tgts, reduction='none') * v).sum() / n_valid

                with torch.no_grad():
                    magnn_err    = (bl.detach() - seg_tgts).abs()
                    spread       = magnn_err[mask].median().clamp(min=0.15)
                    alpha_target = torch.sigmoid(magnn_err / spread - 1.5)

                loss_gate = (F.mse_loss(alpha, alpha_target, reduction='none') * v).sum() / n_valid
                loss_t = loss_pred + 0.8*loss_direct + 0.3*loss_gate

            if not torch.isnan(loss_t) and not torch.isinf(loss_t):
                if _scaler_travel is not None:
                    opt_travel.zero_grad(set_to_none=True)
                    _scaler_travel.scale(loss_t).backward()
                    _scaler_travel.unscale_(opt_travel)
                    torch.nn.utils.clip_grad_norm_(travel_model.parameters(), 1.0)
                    _scaler_travel.step(opt_travel)
                    _scaler_travel.update()
                else:
                    opt_travel.zero_grad(set_to_none=True)
                    loss_t.backward()
                    torch.nn.utils.clip_grad_norm_(travel_model.parameters(), 1.0)
                    opt_travel.step()
                t_loss      += loss_pred.item()
                t_alpha     += (alpha.detach() * v).sum().item() / n_valid.item()
                t_alpha_min  = min(t_alpha_min, alpha.detach().min().item())
                t_alpha_max  = max(t_alpha_max, alpha.detach().max().item())
                t_corr      += ((lstm_pred - bl).abs().detach() * v).sum().item() / n_valid.item()
                t_n         += 1

            for t in range(T):
                valid_t = mask[:, t]
                if not valid_t.any():
                    continue
                pred_d = dwell_model(seg_idx[:, t], temporal[:, t], ctx[:, t])
                tgt_t  = seg_tgts[:, t]
                vt     = valid_t.float().unsqueeze(1)
                loss_d = (F.smooth_l1_loss(pred_d, tgt_t, reduction='none') * vt).sum() / vt.sum().clamp(1)
                if not torch.isnan(loss_d) and not torch.isinf(loss_d):
                    opt_dwell.zero_grad()
                    loss_d.backward()
                    opt_dwell.step()
                    d_loss += loss_d.item()
                    d_n    += 1

        t_loss /= max(t_n, 1)
        d_loss /= max(d_n, 1)
        avg_alpha = t_alpha / max(t_n, 1)
        avg_corr  = t_corr  / max(t_n, 1)

        travel_model.eval()
        dwell_model.eval()
        vt_preds, vt_tgts = [], []
        vt_loss = vt_n = 0.0

        with torch.no_grad():
            for batch in val_loader:
                (seg_idx, temporal, ctx,
                 o_op, d_op, o_wx, d_wx,
                 seg_tgts, lengths, mask) = _unpack_residual_trip_batch(batch, device)

                B, T = seg_idx.shape
                cum_err_val = torch.zeros(B, 1, device=device)

                _, _, _, pred = travel_model(
                    seg_idx, temporal,
                    context_flags=ctx,
                    origin_operational=o_op,
                    dest_operational=d_op,
                    origin_weather=o_wx,
                    dest_weather=d_wx,
                    return_components=True,
                    cumulative_magnn_error=cum_err_val,
                )
                v       = mask.float().unsqueeze(-1)
                n_valid = v.sum().clamp(1)
                loss_t  = (F.smooth_l1_loss(pred, seg_tgts, reduction='none') * v).sum() / n_valid
                if not torch.isnan(loss_t):
                    vt_loss += loss_t.item()
                    vt_n    += 1
                vt_preds.append(pred[mask].cpu().numpy())
                vt_tgts.append(seg_tgts[mask].cpu().numpy())

        vt_loss /= max(vt_n, 1)
        sched_travel.step(vt_loss)
        sched_dwell.step(vt_loss)

        vt_r2 = float('nan')
        if vt_preds:
            vp  = scaler.inverse_transform(np.concatenate(vt_preds).reshape(-1, 1))
            vtt = scaler.inverse_transform(np.concatenate(vt_tgts).reshape(-1, 1))
            try:
                vt_r2 = r2_score(vtt, vp)
            except Exception:
                pass

        if epoch % 5 == 0 or epoch == 1:
            corr_s = avg_corr * (scaler.scale_[0] if hasattr(scaler, 'scale_') else 1.0)
            print(f"  Epoch {epoch:>3}/{cfg.n_epochs}  "
                  f"pred={t_loss:.4f}  val_R²={vt_r2:.4f}  "
                  f"α={avg_alpha:.3f} [{t_alpha_min:.2f},{t_alpha_max:.2f}]  "
                  f"corr={corr_s:.2f}s")

        if not np.isnan(vt_loss) and vt_loss < best_val:
            best_val   = vt_loss
            patience_c = 0
            torch.save({
                'travel_model': travel_model.state_dict(),
                'dwell_model':  dwell_model.state_dict(),
            }, best_ckpt)
        else:
            patience_c += 1
            if patience_c >= cfg.early_stopping_patience:
                print(f"\n  ⏹️  Early stopping at epoch {epoch}")
                break

    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        t_filtered = {k: v for k, v in ckpt['travel_model'].items()
                      if k not in _NODE_KEYS}
        travel_model.load_state_dict(t_filtered, strict=False)
        dwell_model.load_state_dict(ckpt['dwell_model'])

    print_section("MAGNN-LSTM-RESIDUAL — FINAL RESULTS")

    def _eval(loader, name):
        travel_model.eval()
        dwell_model.eval()
        all_preds, all_tgts = [], []
        t_alphas, t_corrs   = [], []

        with torch.no_grad():
            for batch in loader:
                (seg_idx, temporal, ctx,
                 o_op, d_op, o_wx, d_wx,
                 seg_tgts, lengths, mask) = _unpack_residual_trip_batch(batch, device)

                B, T = seg_idx.shape
                cum_err_e = torch.zeros(B, 1, device=device)

                bl, lstm_p, alpha, pred = travel_model(
                    seg_idx, temporal,
                    context_flags=ctx,
                    origin_operational=o_op,
                    dest_operational=d_op,
                    origin_weather=o_wx,
                    dest_weather=d_wx,
                    return_components=True,
                    cumulative_magnn_error=cum_err_e,
                )
                all_preds.append(pred[mask].cpu().numpy())
                all_tgts.append(seg_tgts[mask].cpu().numpy())
                t_alphas.append(alpha[mask].cpu().numpy())
                t_corrs.append((lstm_p[mask] - bl[mask]).cpu().numpy())

        if not all_preds:
            print(f"   {name}: no data")
            return {}

        p = scaler.inverse_transform(np.concatenate(all_preds).reshape(-1, 1))
        t = scaler.inverse_transform(np.concatenate(all_tgts).reshape(-1, 1))
        r2   = r2_score(t, p)
        rmse = float(np.sqrt(mean_squared_error(t, p)))
        mae  = float(mean_absolute_error(t, p))
        mv   = t.flatten() > 0
        mape = (float(np.mean(np.abs((t.flatten()[mv] - p.flatten()[mv]) / t.flatten()[mv])) * 100)
                if mv.any() else float('nan'))
        print(f"  ── {name} ──  R²={r2:.4f}  RMSE={rmse:.2f}s  MAE={mae:.2f}s  MAPE={mape:.2f}%")

        if t_alphas:
            a_all = np.concatenate(t_alphas)
            c_raw = scaler.inverse_transform(np.concatenate(t_corrs).reshape(-1, 1))
            print(f"   α={a_all.mean():.3f}  correction={c_raw.mean():.2f}±{c_raw.std():.2f}s"
                  f"  (neg = LSTM predicting below-median, valid)")

        return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape,
                'preds': p.flatten().tolist(), 'actual': t.flatten().tolist()}

    results = {
        'Train': _eval(train_loader, 'Train'),
        'Val':   _eval(val_loader,   'Val'),
        'Test':  _eval(test_loader,  'Test'),
    }

    test_res = results.get('Test', {})
    if test_res.get('preds'):
        print(f"\n  {'Idx':>4}  {'Actual(s)':>10}  {'Pred(s)':>10}  "
              f"{'Error(s)':>9}  {'Error%':>7}")
        print("  " + "-" * 48)
        for i in range(min(20, len(test_res['actual']))):
            a, p_val = test_res['actual'][i], test_res['preds'][i]
            err  = p_val - a
            epct = (err / a * 100) if a != 0 else 0.0
            print(f"  {i:>3}  {a:>10.2f}  {p_val:>10.2f}  "
                  f"{err:>9.2f}  {epct:>6.2f}%")

    print_section("SEGMENT DIAGNOSTIC — MAGNN vs LSTM vs Final (first 10 test segments)")
    print(f"  {'#':>3}  {'Seg':>6}  {'Label':<40}  "
          f"{'Actual':>8}  {'MAGNN':>8}  {'LSTM':>8}  "
          f"{'α':>6}  {'Final':>8}  {'Err':>7}  Note")
    print("  " + "-" * 118)

    travel_model.eval()
    diag_count     = 0
    seg_types_list = list(segment_types)
    idx_to_seg     = {i: s for i, s in enumerate(seg_types_list)}

    with torch.no_grad():
        for batch in test_loader:
            if diag_count >= 10:
                break
            (seg_idx, temporal, ctx,
             o_op, d_op, o_wx, d_wx,
             seg_tgts, lengths, mask) = _unpack_residual_trip_batch(batch, device)

            B, T = seg_idx.shape
            cum_err_d = torch.zeros(B, 1, device=device)

            bl_t, lstm_t, alpha_t, pred_t = travel_model(
                seg_idx, temporal,
                context_flags=ctx,
                origin_operational=o_op,
                dest_operational=d_op,
                origin_weather=o_wx,
                dest_weather=d_wx,
                return_components=True,
                cumulative_magnn_error=cum_err_d,
            )

            for t in range(T):
                if diag_count >= 10:
                    break
                for b in range(B):
                    if diag_count >= 10:
                        break
                    if not mask[b, t].item():
                        continue
                    s_idx   = int(seg_idx[b, t].item())
                    seg_id  = idx_to_seg.get(s_idx, f"?_{s_idx}")
                    label   = (station_mapper.segment_label(seg_id)
                               if station_mapper else seg_id)[:40]
                    actual_s = float(scaler.inverse_transform(
                        seg_tgts[b, t].cpu().numpy().reshape(1, 1))[0, 0])
                    bl_s    = float(scaler.inverse_transform(
                        bl_t[b, t].cpu().numpy().reshape(1, 1))[0, 0])
                    lstm_s  = float(scaler.inverse_transform(
                        lstm_t[b, t].cpu().numpy().reshape(1, 1))[0, 0])
                    alpha_v = float(alpha_t[b, t, 0].item())
                    final_s = float(scaler.inverse_transform(
                        pred_t[b, t].cpu().numpy().reshape(1, 1))[0, 0])
                    err_s   = final_s - actual_s
                    note    = "below-med" if actual_s < float(scaler.inverse_transform([[0]])[0,0]) else ""
                    print(f"  {diag_count:>3}  {seg_id:>6}  {label:<40}  "
                          f"{actual_s:>8.1f}  {bl_s:>8.1f}  {lstm_s:>8.1f}  "
                          f"{alpha_v:>6.3f}  {final_s:>8.1f}  {err_s:>7.1f}  {note}")
                    diag_count += 1

    print("  " + "-" * 118)
    print("  α: 0=trust MAGNN, 1=trust LSTM  |  neg err = pred faster than actual (fine if near 0)")

    return results, travel_model


def main():

    config = Config()

    print_section("MAGNN TRAINING CONFIGURATION")
    print(f"  Device: {DEVICE}")
    print(f"  Algorithm: {ALGORITHM_NAME}")
    print(f"  Data sampling: {config.sample_fraction * 100}%")
    print(f"  Iterations: {config.n_iterations}")
    print(f"  Epochs per iteration: {config.n_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print("=" * 80)

    for iteration in range(1, config.n_iterations + 1):
        print_section(f"ITERATION {iteration}/{config.n_iterations}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_folder = f"outputs/{ALGORITHM_NAME}_{timestamp}_i{iteration}"
        os.makedirs(output_folder, exist_ok=True)
        print(f"Output: {output_folder}")

        try:
            # Start pipeline timing
            pipeline_start = time.time()
            
            # Data loading
            data_load_start = time.time()
            train_df, test_df, val_df = load_train_test_val_data_fixed(
                data_folder=config.data_folder,
                sample_fraction=config.sample_fraction
            )
            data_load_time = time.time() - data_load_start

            if len(train_df) == 0:
                print("No training data")
                continue

            known_stops_train = get_known_stops(train_df)
            known_stops_test  = get_known_stops(test_df)
            known_stops_val   = get_known_stops(val_df)
            known_stops = {**known_stops_test, **known_stops_val,
                           **known_stops_train}
            print(f"   Known stops: {len(known_stops)} "
                  f"(train={len(known_stops_train)}, "
                  f"test={len(known_stops_test)}, "
                  f"val={len(known_stops_val)})")

            clustering_start = time.time()
            clusters, station_cluster_ids = event_driven_clustering_fixed(
                train_df, known_stops=known_stops
            )
            clustering_time = time.time() - clustering_start
            if len(clusters) == 0:
                print("No clusters")
                continue

            segment_start = time.time()
            train_segments = build_segments_fixed(train_df, clusters)
            if len(train_segments) == 0:
                print("No segments")
                continue

            test_segments = build_segments_fixed(test_df, clusters)
            val_segments = build_segments_fixed(val_df, clusters)
            segment_time = time.time() - segment_start
            val_segments  = build_segments_fixed(val_df,  clusters)

            print_section("GENERATING VISUALISATIONS")
            plot_clusters(
                clusters, {},
                algorithm_name=ALGORITHM_NAME,
                save_path=os.path.join(
                    output_folder, f'{ALGORITHM_NAME.lower()}-clusters.png'))
            plot_segments(
                train_segments, clusters, max_segments=100,
                algorithm_name=ALGORITHM_NAME,
                save_path=os.path.join(
                    output_folder, f'{ALGORITHM_NAME.lower()}-segments.png'))
            plot_segment_statistics(
                train_segments,
                algorithm_name=ALGORITHM_NAME,
                save_path=os.path.join(
                    output_folder,
                    f'{ALGORITHM_NAME.lower()}-segment_stats.png'))
            adj_start = time.time()
            adj_geo, adj_dist, adj_soc, segment_types = \
                build_adjacency_matrices_fixed(
                    train_segments, clusters, known_stops=known_stops)
            adj_time = time.time() - adj_start

            if adj_geo is None:
                print("Adjacency failed")
                continue

            dataset_start = time.time()
            train_dataset = SegmentDataset(
                train_segments, segment_types, fit_scalers=True)
            val_dataset = SegmentDataset(
                val_segments, segment_types, fit_scalers=False,
                target_scaler=train_dataset.target_scaler,
                speed_scaler=train_dataset.speed_scaler)
            test_dataset = SegmentDataset(
                test_segments, segment_types, fit_scalers=False,
                target_scaler=train_dataset.target_scaler,
                speed_scaler=train_dataset.speed_scaler)

            train_loader = DataLoader(
                train_dataset, batch_size=config.batch_size,
                shuffle=True, num_workers=0, collate_fn=masked_collate_fn)
            val_loader = DataLoader(
                val_dataset, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=masked_collate_fn)
            test_loader = DataLoader(
                test_dataset, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=masked_collate_fn)
            dataset_time = time.time() - dataset_start

            print(f"\nData Summary:")
            print(f"   Segment types: {len(segment_types)}")
            print(f"   Train: {len(train_dataset):,}  "
                  f"Val: {len(val_dataset):,}  "
                  f"Test: {len(test_dataset):,}")

            if len(train_loader) == 0:
                print("Training loader empty — skipping")
                continue

            # Training
            training_start = time.time()
            results, _ = train_magtte(
                train_loader, val_loader, test_loader,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_dataset.target_scaler,
                output_folder, DEVICE, config
            )
            training_time = time.time() - training_start
            
            total_pipeline_time = time.time() - pipeline_start
            
            test_res = results.get('Test', {})
            inference_timing = test_res.get('inference_timing', {})
            
            print(f"\n{'='*60}")
            print(f"TIMING METRICS — {ALGORITHM_NAME}")
            print(f"{'='*60}")
            print(f"  Data Loading:              {data_load_time*1000:.2f} ms")
            print(f"  Clustering Time:           {clustering_time*1000:.2f} ms")
            print(f"  Segment Building:          {segment_time*1000:.2f} ms")
            print(f"  Adjacency Matrices:        {adj_time*1000:.2f} ms")
            print(f"  Dataset Creation:          {dataset_time*1000:.2f} ms")
            print(f"  Training Time:             {training_time:.2f} s")
            print(f"  ---")
            print(f"  Avg Model Forward:         {inference_timing.get('avg_model_forward_ms', 0):.4f} ms")
            print(f"  Total Inference Time:      {inference_timing.get('total_inference_time_s', 0):.4f} s")
            print(f"  Throughput:                {inference_timing.get('throughput_samples_per_sec', 0):.2f} samples/sec")
            print(f"  Avg Latency per Sample:    {inference_timing.get('avg_latency_per_sample_ms', 0):.4f} ms")
            print(f"  ---")
            print(f"  Total Pipeline Time:       {total_pipeline_time:.2f} s ({total_pipeline_time*1000:.2f} ms)")
            print(f"{'='*60}")


            DATA_DRIVEN_METHODS = {'knn', 'hdbscan', 'dbscan'}

            metrics_config = {
                'n_epochs':       config.n_epochs,
                'batch_size':     config.batch_size,
                'learning_rate':  config.learning_rate,
                'sample_fraction': config.sample_fraction,
                'n_clusters':     len(clusters),
                'cluster_count_method': (
                    'data-driven' if _cluster_method in DATA_DRIVEN_METHODS
                    else 'requested'),
                'n_segment_types': len(segment_types),
                'node_embed_dim': config.node_embed_dim,
                'gat_hidden':     config.gat_hidden,
                'lstm_hidden':    config.lstm_hidden,
                'historical_dim': config.historical_dim,
            }

            metrics_out = {
                'graph_method': ALGORITHM_NAME.lower(),
                'config': metrics_config,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'timing': {
                    'data_load_time_ms': round(data_load_time * 1000, 4),
                    'clustering_time_ms': round(clustering_time * 1000, 4),
                    'segment_building_time_ms': round(segment_time * 1000, 4),
                    'adjacency_time_ms': round(adj_time * 1000, 4),
                    'dataset_creation_time_ms': round(dataset_time * 1000, 4),
                    'training_time_s': round(training_time, 4),
                    'avg_model_forward_ms': round(inference_timing.get('avg_model_forward_ms', 0), 4),
                    'total_inference_time_s': round(inference_timing.get('total_inference_time_s', 0), 4),
                    'throughput_samples_per_sec': round(inference_timing.get('throughput_samples_per_sec', 0), 2),
                    'avg_latency_per_sample_ms': round(inference_timing.get('avg_latency_per_sample_ms', 0), 4),
                    'total_pipeline_time_ms': round(total_pipeline_time * 1000, 4),
                }
            }

            for dataset_name, res in [('Train', results.get('Train')),
                                      ('Val', results.get('Val')),
                                      ('Test', results.get('Test'))]:
                if res:
                    metrics_out[dataset_name] = {
                        'R2':   f"{res['r2']:.4f}",
                        'RMSE': f"{res['rmse']:.2f} seconds",
                        'MAE':  f"{res['mae']:.2f} seconds",
                        'MAPE': f"{res['mape']:.2f}%",
                    }

            json_path = os.path.join(output_folder, 'metrics.json')
            with open(json_path, 'w') as f:
                json.dump(metrics_out, f, indent=2)
            print(f"\nMetrics saved → {json_path}")
            print(f"Iteration {iteration} complete")

        except Exception as e:
            print(f"\nError in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            continue

def print_three_way_comparison_table(magnn_results, residual_results,
                                     mtl_results, split_name):
    magnn_m    = magnn_results.get(split_name, {})
    residual_m = residual_results.get(split_name, {})
    mtl_m      = mtl_results.get(split_name, {})

    print(f"\n{'=' * 120}")
    print(f"{split_name.upper()} SET - THREE-WAY COMPARISON")
    print(f"  MAGNN + Residual : per-segment duration [SEG]  (~50s scale)")
    print(f"  TripPredictor    : total trip duration  [TRIP] (~300s scale) — different task")
    print(f"{'=' * 120}")
    print(f"{'Metric':<10} {'MAGNN [SEG]':<22} {'Residual [SEG]':<22} "
          f"{'TripPredictor [TRIP]':<22} {'Best (seg task)':<20}")
    print(f"{'-' * 120}")

    for display_name, metric_key, higher_better, fmt in [
        ('R²',   'r2',   True,  lambda v: f"{v:.4f}"),
        ('RMSE', 'rmse', False, lambda v: f"{v:.2f}s"),
        ('MAE',  'mae',  False, lambda v: f"{v:.2f}s"),
        ('MAPE', 'mape', False, lambda v: f"{v:.2f}%"),
    ]:
        mv = magnn_m.get(metric_key, float('nan'))
        rv = residual_m.get(metric_key, float('nan'))
        tv = mtl_m.get(metric_key, float('nan'))

        best_seg = ("Residual ✓"
                    if (not np.isnan(mv) and not np.isnan(rv) and
                        (rv > mv if higher_better else rv < mv))
                    else "MAGNN ✓" if not np.isnan(mv) else "N/A")

        print(f"{display_name:<10} "
              f"{(fmt(mv) if not np.isnan(mv) else 'N/A'):<22} "
              f"{(fmt(rv) if not np.isnan(rv) else 'N/A'):<22} "
              f"{(fmt(tv) if not np.isnan(tv) else 'N/A'):<22} "
              f"{best_seg:<20}")

    print(f"{'=' * 120}\n")

def main_with_three_way_comparison():

    config = Config()

    print_section("THREE-WAY COMPARISON MODE")
    print(f"  Device: {DEVICE}   Algorithm: {ALGORITHM_NAME}")
    print(f"\n  Models:")
    print(f"    1. MAGNN (baseline segment predictor)")
    print(f"    2. MAGNN-LSTM-Residual (station-aware, best per-segment predictor)")
    print(f"    3. Hierarchical Trip Predictor (total trip duration, cumulative correction)")
    print("=" * 80)

    all_magnn_results    = []
    all_residual_results = []
    all_mtl_results      = []

    for iteration in range(1, config.n_iterations + 1):
        print_section(f"ITERATION {iteration}/{config.n_iterations}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_folder = (f"outputs/{ALGORITHM_NAME}_{timestamp}"
                         f"_i{iteration}_three_way")
        os.makedirs(output_folder, exist_ok=True)

        try:
            print("\n[1/10] Loading data...")
            train_df, test_df, val_df = load_train_test_val_data_fixed(
                data_folder=config.data_folder,
                sample_fraction=config.sample_fraction
            )
            known_stops_train = get_known_stops(train_df)
            known_stops_test  = get_known_stops(test_df)
            known_stops_val   = get_known_stops(val_df)
            known_stops = {**known_stops_test, **known_stops_val,
                           **known_stops_train}

            print("[2/10] Clustering stops...")
            clusters, station_cluster_ids = event_driven_clustering_fixed(
                train_df, known_stops=known_stops)

            print("[3/10] Building segments...")
            train_segments = build_segments_fixed(train_df, clusters)
            test_segments  = build_segments_fixed(test_df,  clusters)
            val_segments   = build_segments_fixed(val_df,   clusters)

            print("[3b/10] Generating visualisations...")
            plot_clusters(
                clusters, {}, algorithm_name=ALGORITHM_NAME,
                save_path=os.path.join(
                    output_folder,
                    f'{ALGORITHM_NAME.lower()}-clusters.png'))
            plot_segments(
                train_segments, clusters, max_segments=100,
                algorithm_name=ALGORITHM_NAME,
                save_path=os.path.join(
                    output_folder,
                    f'{ALGORITHM_NAME.lower()}-segments.png'))
            plot_segment_statistics(
                train_segments, algorithm_name=ALGORITHM_NAME,
                save_path=os.path.join(
                    output_folder,
                    f'{ALGORITHM_NAME.lower()}-segment_stats.png'))

            print("[4/10] Building adjacency matrices...")
            adj_geo, adj_dist, adj_soc, segment_types = \
                build_adjacency_matrices_fixed(
                    train_segments, clusters, known_stops=known_stops)

            print("[5/10] Preparing MAGNN datasets...")
            train_ds_magnn = SegmentDataset(
                train_segments, segment_types, fit_scalers=True)
            val_ds_magnn = SegmentDataset(
                val_segments, segment_types, fit_scalers=False,
                target_scaler=train_ds_magnn.target_scaler,
                speed_scaler=train_ds_magnn.speed_scaler)
            test_ds_magnn = SegmentDataset(
                test_segments, segment_types, fit_scalers=False,
                target_scaler=train_ds_magnn.target_scaler,
                speed_scaler=train_ds_magnn.speed_scaler)

            train_loader_magnn = DataLoader(
                train_ds_magnn, batch_size=config.batch_size,
                shuffle=True, num_workers=0, collate_fn=masked_collate_fn)
            val_loader_magnn = DataLoader(
                val_ds_magnn, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=masked_collate_fn)
            test_loader_magnn = DataLoader(
                test_ds_magnn, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=masked_collate_fn)

            print("[6/10] Preparing Trip datasets "
                  "(trip_id + same-day grouping, sorted by departure_time)...")
            (train_loader_trip, val_loader_trip, test_loader_trip,
             train_ds_trip) = _make_trip_loaders(
                train_segments, val_segments, test_segments,
                segment_types, config,
                train_df, clusters, known_stops,
                max_trip_length=15,
            )

            print(f"\nDataset Summary:")
            print(f"   Trip train: {len(train_ds_trip):,} trips  "
                  f"val: {len(val_loader_trip.dataset):,}  "
                  f"test: {len(test_loader_trip.dataset):,}")
            if train_ds_trip.station_mapper is not None:
                sm = train_ds_trip.station_mapper
                n_matched = len(sm.cluster_to_station)
                print(f"   Clusters matched to named stations: "
                      f"{n_matched}/{sm.n_clusters}")

            print("\n[7/10] Training MAGNN (baseline)...")
            print("-" * 80)
            magnn_results, magnn_model = train_magtte(
                train_loader_magnn, val_loader_magnn, test_loader_magnn,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_ds_magnn.target_scaler,
                output_folder, DEVICE, config
            )
            all_magnn_results.append(magnn_results)
            magnn_checkpoint = os.path.join(output_folder, 'magtte_best.pth')

            print("\n[8/10] Training MAGNN-LSTM-Residual (trip-sequential)...")
            print("-" * 80)
            residual_results, residual_model = train_magnn_lstm_residual(
                train_loader_trip, val_loader_trip, test_loader_trip,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_ds_trip.target_scaler,
                output_folder, DEVICE, config,
                clusters=clusters,
                known_stops=known_stops,
                pretrained_magnn_path=magnn_checkpoint,
                freeze_magnn=True,
            )
            all_residual_results.append(residual_results)

            print("\n[9/10] Training Hierarchical Trip Predictor "
                  "(cumulative correction, local+global MTL)...")
            print("-" * 80)
            mtl_results, mtl_model = train_trip_level_predictor(
                train_loader_trip, val_loader_trip, test_loader_trip,
                segment_types, train_ds_trip.trip_target_scaler,
                output_folder, DEVICE, config,
                pretrained_residual_model=residual_model,
                seg_scaler=train_ds_trip.target_scaler,
                day_scaler=train_ds_trip.day_target_scaler,
                n_epochs_override=50,
            )
            all_mtl_results.append(mtl_results)

            print("\n[10/10] Generating comparison report...")
            print_section(
                f"ITERATION {iteration} — THREE-WAY COMPARISON")
            for split in ['Train', 'Val', 'Test']:
                print_three_way_comparison_table(
                    magnn_results, residual_results, mtl_results, split)

            comparison_json = {
                'iteration': iteration,
                'algorithm': ALGORITHM_NAME,
                'config': {
                    'n_epochs':        config.n_epochs,
                    'batch_size':      config.batch_size,
                    'learning_rate':   config.learning_rate,
                    'sample_fraction': config.sample_fraction,
                },
                'magnn': {
                    sp: {k: float(v) for k, v in
                         magnn_results.get(sp, {}).items()
                         if k in ('r2', 'rmse', 'mae', 'mape')}
                    for sp in ('Train', 'Val', 'Test')
                },
                'residual': {
                    sp: {k: float(v) for k, v in
                         residual_results.get(sp, {}).items()
                         if k in ('r2', 'rmse', 'mae', 'mape', 'alpha')}
                    for sp in ('Train', 'Val', 'Test')
                },
                'dualtask_mtl': {
                    sp: {k: float(v) for k, v in
                         mtl_results.get(sp, {}).items()
                         if k in ('r2', 'rmse', 'mae', 'mape')}
                    for sp in ('Train', 'Val', 'Test')
                },
            }

            json_path = os.path.join(output_folder,
                                     'three_way_comparison.json')
            with open(json_path, 'w') as f:
                json.dump(comparison_json, f, indent=2)
            print(f"Comparison saved → {json_path}\n")

        except Exception as e:
            print(f"\nError in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_magnn_results and all_residual_results and all_mtl_results:
        n_iters = len(all_magnn_results)
        print_section("FINAL SUMMARY — AVERAGED ACROSS ALL ITERATIONS")

        for split in ['Train', 'Val', 'Test']:
            print(f"\n{'=' * 120}")
            print(f"{split.upper()} SET — AVERAGE OVER {n_iters} ITERATION(S)")
            print(f"  MAGNN + Residual → per-segment [SEG] | TripPredictor → trip-total [TRIP]")
            print(f"{'=' * 120}")
            print(f"{'Metric':<10} {'MAGNN [SEG]':<22} {'Residual [SEG]':<22} "
                  f"{'TripPredictor [TRIP]':<22} {'Best (seg task)':<15}")
            print(f"{'-' * 120}")

            for display_name, metric_key, higher_better in [
                ('R²',   'r2',   True),
                ('RMSE', 'rmse', False),
                ('MAE',  'mae',  False),
                ('MAPE', 'mape', False),
            ]:
                mv_vals = [r.get(split, {}).get(metric_key, float('nan'))
                           for r in all_magnn_results]
                rv_vals = [r.get(split, {}).get(metric_key, float('nan'))
                           for r in all_residual_results]
                tv_vals = [r.get(split, {}).get(metric_key, float('nan'))
                           for r in all_mtl_results]

                ma, ms = np.nanmean(mv_vals), np.nanstd(mv_vals)
                ra, rs = np.nanmean(rv_vals), np.nanstd(rv_vals)
                ta, ts = np.nanmean(tv_vals), np.nanstd(tv_vals)

                def _fmts(avg, std):
                    if metric_key in ('rmse', 'mae'):
                        return f"{avg:.2f}±{std:.2f}s"
                    if metric_key == 'mape':
                        return f"{avg:.2f}±{std:.2f}%"
                    return f"{avg:.4f}±{std:.4f}"

                best_seg = ("Residual ✓"
                            if (not np.isnan(ma) and not np.isnan(ra) and
                                (ra > ma if higher_better else ra < ma))
                            else "MAGNN" if not np.isnan(ma) else "N/A")

                print(f"{display_name:<10} {_fmts(ma,ms):<22} "
                      f"{_fmts(ra,rs):<22} {_fmts(ta,ts):<22} "
                      f"{best_seg:<15}")

            print(f"{'=' * 120}\n")

        if n_iters >= 3:
            print_section("STATISTICAL SIGNIFICANCE TESTING  "
                          f"(n={n_iters} independent runs, Test set)")
            print("  Per the methodology (§4.4.4): two non-parametric tests are used.")
            print("  (A) Wilcoxon Signed-Rank — paired, each baseline vs MA-GNN-LSTM")
            print("  (B) Kruskal-Wallis H     — across all model groups simultaneously")
            print("  Two-tailed α=0.05  |  p<0.05 → *, p<0.01 → **, p<0.001 → ***\n")

            _sig_label = lambda p: (
                '***' if p < 0.001 else '**' if p < 0.01
                else '*' if p < 0.05 else 'n.s.')

            split = 'Test'

            # ── (A) Wilcoxon Signed-Rank: each baseline vs MA-GNN-LSTM (MTL) ──
            print(f"  (A) WILCOXON SIGNED-RANK TEST  "
                  f"— each baseline vs MA-GNN-LSTM (proposed)")
            print(f"  H₀: baseline model = MA-GNN-LSTM  (paired, n={n_iters} runs)")
            print(f"  {'Baseline':<25} {'Metric':<8} "
                  f"{'ΔMean (prop−base)':>20}  {'W stat':>8}  {'p-value':>10}  "
                  f"{'Sig':>5}  {'Cohen d':>9}")
            print(f"  {'-' * 100}")

            # Proposed = MTL (MA-GNN-LSTM); baselines = MAGNN, Residual
            _wilc_comps = [
                ("MAGNN (baseline)",    all_magnn_results),
                ("Residual (baseline)", all_residual_results),
            ]

            for baseline_name, baseline_res in _wilc_comps:
                for display_name, metric_key, higher_better in [
                    ('R²',   'r2',   True),
                    ('RMSE', 'rmse', False),
                    ('MAE',  'mae',  False),
                    ('MAPE', 'mape', False),
                ]:
                    v_prop = np.array([r.get(split, {}).get(metric_key,
                                       float('nan')) for r in all_mtl_results],
                                      dtype=float)
                    v_base = np.array([r.get(split, {}).get(metric_key,
                                       float('nan')) for r in baseline_res],
                                      dtype=float)
                    ok = ~(np.isnan(v_prop) | np.isnan(v_base))
                    vp, vb = v_prop[ok], v_base[ok]
                    if len(vp) < 3:
                        continue

                    # Di = Proposed_i − Baseline_i  (positive = proposed is better)
                    diff = vp - vb
                    delta = float(np.mean(diff))

                    # Compute W+ and W- manually per methodology spec
                    nonzero = diff[diff != 0]
                    if len(nonzero) == 0:
                        p_wilc, w_stat = 1.0, 0.0
                    else:
                        abs_ranks = scipy_stats.rankdata(np.abs(nonzero))
                        w_plus  = float(np.sum(abs_ranks[nonzero > 0]))
                        w_minus = float(np.sum(abs_ranks[nonzero < 0]))
                        w_stat  = min(w_plus, w_minus)
                        try:
                            _, p_wilc = scipy_stats.wilcoxon(
                                vp, vb, alternative='two-sided',
                                zero_method='wilcox')
                        except Exception:
                            p_wilc = float('nan')

                    # Cohen's d (paired)
                    std_d   = float(np.std(diff, ddof=1))
                    cohen_d = (delta / std_d if std_d > 1e-12 else float('nan'))

                    sig   = _sig_label(p_wilc) if not np.isnan(p_wilc) else 'N/A'
                    p_s   = f"{p_wilc:.4f}"    if not np.isnan(p_wilc) else "N/A"
                    d_s   = f"{cohen_d:+.3f}"  if not np.isnan(cohen_d) else "N/A"
                    w_s   = f"{w_stat:.1f}"

                    if metric_key in ('rmse', 'mae'):
                        d_str = f"{delta:+.2f}s"
                    elif metric_key == 'mape':
                        d_str = f"{delta:+.2f}%"
                    else:
                        d_str = f"{delta:+.4f}"

                    print(f"  {baseline_name:<25} {display_name:<8} "
                          f"{d_str:>20}  {w_s:>8}  {p_s:>10}  "
                          f"{sig:>5}  {d_s:>9}")

                print()

            # ── (B) Kruskal-Wallis H: across all 3 model groups ───────────────
            print(f"\n  (B) KRUSKAL-WALLIS H TEST  "
                  f"— across all model groups (k=3, n={n_iters} per group)")
            print(f"  H₀: all models have the same performance distribution")
            print(f"  {'Metric':<8} {'H statistic':>14}  {'df':>4}  "
                  f"{'p-value':>10}  {'Sig':>5}  Interpretation")
            print(f"  {'-' * 80}")

            for display_name, metric_key, higher_better in [
                ('R²',   'r2',   True),
                ('RMSE', 'rmse', False),
                ('MAE',  'mae',  False),
                ('MAPE', 'mape', False),
            ]:
                g_magnn = np.array([r.get(split, {}).get(metric_key, float('nan'))
                                    for r in all_magnn_results],    dtype=float)
                g_resid = np.array([r.get(split, {}).get(metric_key, float('nan'))
                                    for r in all_residual_results], dtype=float)
                g_mtl   = np.array([r.get(split, {}).get(metric_key, float('nan'))
                                    for r in all_mtl_results],      dtype=float)

                # Drop NaNs per-group (keep complete cases)
                g_magnn = g_magnn[~np.isnan(g_magnn)]
                g_resid = g_resid[~np.isnan(g_resid)]
                g_mtl   = g_mtl[~np.isnan(g_mtl)]

                if min(len(g_magnn), len(g_resid), len(g_mtl)) < 3:
                    print(f"  {display_name:<8}  insufficient data")
                    continue

                try:
                    h_stat, p_kw = scipy_stats.kruskal(g_magnn, g_resid, g_mtl)
                except Exception as e:
                    print(f"  {display_name:<8}  error: {e}")
                    continue

                df  = 2   # k − 1 = 3 − 1
                sig = _sig_label(p_kw)
                p_s = f"{p_kw:.4f}"
                h_s = f"{h_stat:.4f}"

                if p_kw < 0.05:
                    interp = "≥1 model significantly differs"
                else:
                    interp = "no significant difference"

                print(f"  {display_name:<8} {h_s:>14}  {df:>4}  "
                      f"{p_s:>10}  {sig:>5}  {interp}")

            print(f"\n  NOTE: A significant Kruskal-Wallis result confirms that at")
            print(f"  least one model differs. Wilcoxon (A) identifies which pairs.")


    print_section("THREE-WAY COMPARISON COMPLETE")
    print(f"  Results saved in: outputs/{ALGORITHM_NAME}_*_three_way/")
    print("=" * 80)

def print_ablation_comparison_table(full_results, no_op_results,
                                    no_weather_results, baseline_results,
                                    split_name):
    """Print formatted ablation study comparison table."""

    full_m       = full_results.get(split_name, {})
    no_op_m      = no_op_results.get(split_name, {})
    no_weather_m = no_weather_results.get(split_name, {})
    baseline_m   = baseline_results.get(split_name, {})

    print(f"\n{'=' * 130}")
    print(f"{split_name.upper()} SET — ABLATION STUDY COMPARISON")
    print(f"{'=' * 130}")
    print(f"{'Metric':<10} {'Full':<20} {'No Operational':<20} "
          f"{'No Weather':<20} {'Baseline':<20} {'Best':<15}")
    print(f"{'-' * 130}")

    for display_name, metric_key, higher_better in [
        ('R²',   'r2',   True),
        ('RMSE', 'rmse', False),
        ('MAE',  'mae',  False),
        ('MAPE', 'mape', False),
    ]:
        fv  = full_m.get(metric_key, float('nan'))
        ov  = no_op_m.get(metric_key, float('nan'))
        wv  = no_weather_m.get(metric_key, float('nan'))
        bv  = baseline_m.get(metric_key, float('nan'))

        def _fmt(v):
            if metric_key in ('rmse', 'mae'): return f"{v:.2f}s"
            if metric_key == 'mape':          return f"{v:.2f}%"
            return f"{v:.4f}"

        if not any(np.isnan(x) for x in [fv, ov, wv, bv]):
            best = (max if higher_better else min)(fv, ov, wv, bv)
            best_model = ("Full ✓"       if best == fv
                          else "No Op ✓"      if best == ov
                          else "No Weather ✓" if best == wv
                          else "Baseline ✓")
        else:
            best_model = "N/A"

        print(f"{display_name:<10} {_fmt(fv):<20} {_fmt(ov):<20} "
              f"{_fmt(wv):<20} {_fmt(bv):<20} {best_model:<15}")

    print(f"{'=' * 130}\n")


def main_with_ablation_study():
    """Ablation study: feature importance using MAGNN-LSTM-Residual."""

    config = Config()

    print_section("ABLATION STUDY MODE — MAGNN-LSTM-MTL (Full Model)")
    print(f"  Device: {DEVICE}   Algorithm: {ALGORITHM_NAME}")
    print(f"\n  All 4 variants train the COMPLETE MAGNN-LSTM-MTL model")
    print(f"  (Residual encoder → MTL trip predictor) with features ablated.")
    print(f"\n  Variants:")
    print(f"    1. Full            — All features (weather + operational)")
    print(f"    2. No Operational  — arrivalDelay/departureDelay/is_weekend/is_peak zeroed")
    print(f"    3. No Weather      — All weather columns zeroed")
    print(f"    4. Baseline        — Spatial + temporal only (both groups zeroed)")
    print("=" * 80)

    all_full_results          = []
    all_no_operational_results = []
    all_no_weather_results    = []
    all_baseline_results      = []

    WEATHER_COLS = ['temperature_2m', 'apparent_temperature', 'precipitation',
                    'rain', 'snowfall', 'windspeed_10m', 'windgusts_10m',
                    'winddirection_10m']

    for iteration in range(1, config.n_iterations + 1):
        print_section(f"ITERATION {iteration}/{config.n_iterations}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_folder = (f"outputs/{ALGORITHM_NAME}_{timestamp}"
                         f"_i{iteration}_ablation")
        os.makedirs(output_folder, exist_ok=True)

        try:
            print("\n[1/9] Loading data...")
            train_df, test_df, val_df = load_train_test_val_data_fixed(
                data_folder=config.data_folder,
                sample_fraction=config.sample_fraction
            )
            known_stops_train = get_known_stops(train_df)
            known_stops_test  = get_known_stops(test_df)
            known_stops_val   = get_known_stops(val_df)
            known_stops = {**known_stops_test, **known_stops_val,
                           **known_stops_train}
            print(f"   Known stops: {len(known_stops)}")

            print("[2/9] Clustering stops...")
            clusters, _ = event_driven_clustering_fixed(
                train_df, known_stops=known_stops)

            print("[3/9] Building segments...")
            train_segments = build_segments_fixed(train_df, clusters)
            test_segments  = build_segments_fixed(test_df,  clusters)
            val_segments   = build_segments_fixed(val_df,   clusters)

            print("[4/9] Building adjacency matrices...")
            adj_geo, adj_dist, adj_soc, segment_types = \
                build_adjacency_matrices_fixed(
                    train_segments, clusters, known_stops=known_stops)

            print("\n[5/9] Training base MAGNN for transfer learning...")
            print("-" * 80)
            train_ds_magnn = SegmentDataset(
                train_segments, segment_types, fit_scalers=True)
            val_ds_magnn = SegmentDataset(
                val_segments, segment_types, fit_scalers=False,
                target_scaler=train_ds_magnn.target_scaler,
                speed_scaler=train_ds_magnn.speed_scaler)
            test_ds_magnn = SegmentDataset(
                test_segments, segment_types, fit_scalers=False,
                target_scaler=train_ds_magnn.target_scaler,
                speed_scaler=train_ds_magnn.speed_scaler)

            tl_m = DataLoader(train_ds_magnn, batch_size=config.batch_size,
                              shuffle=True,  num_workers=0,
                              collate_fn=masked_collate_fn)
            vl_m = DataLoader(val_ds_magnn, batch_size=config.batch_size,
                              shuffle=False, num_workers=0,
                              collate_fn=masked_collate_fn)
            tstl_m = DataLoader(test_ds_magnn, batch_size=config.batch_size,
                                shuffle=False, num_workers=0,
                                collate_fn=masked_collate_fn)

            _, magnn_model = train_magtte(
                tl_m, vl_m, tstl_m,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_ds_magnn.target_scaler,
                output_folder, DEVICE, config
            )
            magnn_checkpoint = os.path.join(output_folder, 'magtte_best.pth')

            def _ablation_loaders(tr_seg, va_seg, te_seg):
                return _make_trip_loaders(
                    tr_seg, va_seg, te_seg,
                    segment_types, config,
                    train_df, clusters, known_stops,
                    max_trip_length=15,
                )

            def _run_full_mtl(tl, vl, tstl, ds_train, tag):
                # Stage 1: residual LSTM (encoder warm-up)
                _, residual_mdl = train_magnn_lstm_residual(
                    tl, vl, tstl,
                    adj_geo, adj_dist, adj_soc,
                    segment_types, ds_train.target_scaler,
                    output_folder, DEVICE, config,
                    clusters=clusters,
                    known_stops=known_stops,
                    pretrained_magnn_path=magnn_checkpoint,
                    freeze_magnn=True,
                )
                mtl_res, _ = train_trip_level_predictor(
                    tl, vl, tstl,
                    segment_types, ds_train.trip_target_scaler,
                    output_folder, DEVICE, config,
                    pretrained_residual_model=residual_mdl,
                    seg_scaler=ds_train.target_scaler,
                    day_scaler=ds_train.day_target_scaler,
                    n_epochs_override=50,
                )
                return mtl_res, None

            print("\n[6/9] Training MAGNN-LSTM-MTL — FULL (all features)...")
            print("-" * 80)
            tl, vl, tstl, ds = _ablation_loaders(
                train_segments, val_segments, test_segments)
            full_results, _ = _run_full_mtl(tl, vl, tstl, ds, "full")
            all_full_results.append(full_results)
            print("Full model trained")

            print("\n[7/9] Training MAGNN-LSTM-MTL — NO OPERATIONAL "
                  "(arrivalDelay / departureDelay / is_weekend / is_peak_hour zeroed)...")
            print("-" * 80)
            tr_no_op = train_segments.copy()
            va_no_op = val_segments.copy()
            te_no_op = test_segments.copy()
            for df in (tr_no_op, va_no_op, te_no_op):
                df['arrivalDelay']   = 0.0
                df['departureDelay'] = 0.0
                df['is_weekend']     = 0.0
                df['is_peak_hour']   = 0.0

            tl, vl, tstl, ds = _ablation_loaders(tr_no_op, va_no_op, te_no_op)
            no_op_results, _ = _run_full_mtl(tl, vl, tstl, ds, "no_op")
            all_no_operational_results.append(no_op_results)
            print("No-operational model trained")

            print("\n[8/9] Training MAGNN-LSTM-MTL — NO WEATHER "
                  "(all weather columns zeroed)...")
            print("-" * 80)
            tr_no_w = train_segments.copy()
            va_no_w = val_segments.copy()
            te_no_w = test_segments.copy()
            for df in (tr_no_w, va_no_w, te_no_w):
                for col in WEATHER_COLS:
                    if col in df.columns:
                        df[col] = 0.0

            tl, vl, tstl, ds = _ablation_loaders(tr_no_w, va_no_w, te_no_w)
            no_weather_results, _ = _run_full_mtl(tl, vl, tstl, ds,
                                                  "no_weather")
            all_no_weather_results.append(no_weather_results)
            print("No-weather model trained")

            print("\n[9/9] Training MAGNN-LSTM-MTL — BASELINE "
                  "(spatial + temporal only; all op + weather zeroed)...")
            print("-" * 80)
            tr_bl = train_segments.copy()
            va_bl = val_segments.copy()
            te_bl = test_segments.copy()
            for df in (tr_bl, va_bl, te_bl):
                df['arrivalDelay']   = 0.0
                df['departureDelay'] = 0.0
                df['is_weekend']     = 0.0
                df['is_peak_hour']   = 0.0
                for col in WEATHER_COLS:
                    if col in df.columns:
                        df[col] = 0.0

            tl, vl, tstl, ds = _ablation_loaders(tr_bl, va_bl, te_bl)
            baseline_results, _ = _run_full_mtl(tl, vl, tstl, ds, "baseline")
            all_baseline_results.append(baseline_results)
            print("Baseline model trained")

            print_section(
                f"ITERATION {iteration} — ABLATION STUDY RESULTS")
            for split in ('Train', 'Val', 'Test'):
                print_ablation_comparison_table(
                    full_results, no_op_results,
                    no_weather_results, baseline_results, split)

            ablation_json = {
                'iteration': iteration,
                'algorithm': ALGORITHM_NAME,
                'model':     'MAGNN-LSTM-MTL (full model, trip-level)',
                'config': {
                    'n_epochs':        config.n_epochs,
                    'batch_size':      config.batch_size,
                    'learning_rate':   config.learning_rate,
                    'sample_fraction': config.sample_fraction,
                },
            }
            _keep = ('r2', 'rmse', 'mae', 'mape', 'alpha',
                     'correction_mean', 'correction_std')
            for key, res in [('full',           full_results),
                              ('no_operational', no_op_results),
                              ('no_weather',     no_weather_results),
                              ('baseline',       baseline_results)]:
                ablation_json[key] = {
                    sp: {k: float(v) for k, v in res.get(sp, {}).items()
                         if k in _keep}
                    for sp in ('Train', 'Val', 'Test')
                }

            json_path = os.path.join(output_folder, 'ablation_study.json')
            with open(json_path, 'w') as f:
                json.dump(ablation_json, f, indent=2)
            print(f"Ablation results saved → {json_path}\n")

        except Exception as e:
            print(f"\nError in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if (all_full_results and all_no_operational_results
            and all_no_weather_results and all_baseline_results):

        n_iters = len(all_full_results)
        print_section("ABLATION STUDY — FINAL SUMMARY  "
                      f"(MAGNN-LSTM-MTL, n={n_iters} iterations)")

        for split in ('Train', 'Val', 'Test'):
            print(f"\n{'=' * 130}")
            print(f"{split.upper()} — AVERAGE OVER {n_iters} ITERATION(S)")
            print(f"{'=' * 130}")
            print(f"{'Metric':<10} {'Full':<22} {'No Operational':<22} "
                  f"{'No Weather':<22} {'Baseline':<22} {'Best':<15}")
            print(f"{'-' * 130}")

            for display_name, metric_key, higher_better in [
                ('R²',   'r2',   True),
                ('RMSE', 'rmse', False),
                ('MAE',  'mae',  False),
                ('MAPE', 'mape', False),
            ]:
                fa = np.nanmean([r.get(split, {}).get(metric_key, float('nan'))
                                 for r in all_full_results])
                oa = np.nanmean([r.get(split, {}).get(metric_key, float('nan'))
                                 for r in all_no_operational_results])
                wa = np.nanmean([r.get(split, {}).get(metric_key, float('nan'))
                                 for r in all_no_weather_results])
                ba = np.nanmean([r.get(split, {}).get(metric_key, float('nan'))
                                 for r in all_baseline_results])
                fs = np.nanstd([r.get(split, {}).get(metric_key, float('nan'))
                                for r in all_full_results])
                os_ = np.nanstd([r.get(split, {}).get(metric_key, float('nan'))
                                 for r in all_no_operational_results])
                ws = np.nanstd([r.get(split, {}).get(metric_key, float('nan'))
                                for r in all_no_weather_results])
                bs = np.nanstd([r.get(split, {}).get(metric_key, float('nan'))
                                for r in all_baseline_results])

                def _fmts(avg, std):
                    if metric_key in ('rmse', 'mae'):
                        return f"{avg:.2f}±{std:.2f}s"
                    if metric_key == 'mape':
                        return f"{avg:.2f}±{std:.2f}%"
                    return f"{avg:.4f}±{std:.4f}"

                if not any(np.isnan(x) for x in [fa, oa, wa, ba]):
                    best = (max if higher_better else min)(fa, oa, wa, ba)
                    best_model = ("Full ✓"       if best == fa
                                  else "No Op ✓"      if best == oa
                                  else "No Weather ✓" if best == wa
                                  else "Baseline ✓")
                else:
                    best_model = "N/A"

                print(f"{display_name:<10} {_fmts(fa,fs):<22} "
                      f"{_fmts(oa,os_):<22} {_fmts(wa,ws):<22} "
                      f"{_fmts(ba,bs):<22} {best_model:<15}")

            print(f"{'=' * 130}\n")

        print_section("STATISTICAL SIGNIFICANCE TESTING  "
                      f"(n={n_iters} independent runs, Test set)")
        print("  Per the methodology (§4.4.3 / §4.4.4): two non-parametric tests.")
        print("  (A) Wilcoxon Signed-Rank — paired, each ablated variant vs Full model")
        print("  (B) Kruskal-Wallis H     — across all 4 ablation variants simultaneously")
        print("  Two-tailed α=0.05  |  p<0.05 → *, p<0.01 → **, p<0.001 → ***\n")

        _sig_label = lambda p: (
            '***' if p < 0.001 else '**' if p < 0.01
            else '*' if p < 0.05 else 'n.s.')

        split = 'Test'

        # ── (A) Wilcoxon Signed-Rank: each ablated variant vs Full MAGNN-LSTM-MTL
        print(f"  (A) WILCOXON SIGNED-RANK TEST  "
              f"— each ablated variant vs Full model (proposed)")
        print(f"  H₀: ablated variant = Full model  (paired, n={n_iters} runs)")
        print(f"  Di = Full_i − Ablated_i  "
              f"(positive = Full is better)")
        print(f"  {'Ablated Variant':<28} {'Metric':<8} "
              f"{'ΔMean (Full−Abl)':>18}  {'W stat':>8}  {'p-value':>10}  "
              f"{'Sig':>5}  {'Cohen d':>9}")
        print(f"  {'-' * 103}")

        _wilc_abl_comps = [
            ("No Operational",  all_no_operational_results),
            ("No Weather",      all_no_weather_results),
            ("Baseline",        all_baseline_results),
        ]

        for ablation_name, ablated_res in _wilc_abl_comps:
            for display_name, metric_key, higher_better in [
                ('R²',   'r2',   True),
                ('RMSE', 'rmse', False),
                ('MAE',  'mae',  False),
                ('MAPE', 'mape', False),
            ]:
                v_full = np.array([r.get(split, {}).get(metric_key,
                                   float('nan')) for r in all_full_results],
                                  dtype=float)
                v_abl  = np.array([r.get(split, {}).get(metric_key,
                                   float('nan')) for r in ablated_res],
                                  dtype=float)
                ok = ~(np.isnan(v_full) | np.isnan(v_abl))
                vf, va = v_full[ok], v_abl[ok]
                if len(vf) < 3:
                    continue

                # Di = Full_i − Ablated_i
                diff  = vf - va
                delta = float(np.mean(diff))

                # W+ and W- per methodology spec (§4.4.4.A)
                nonzero = diff[diff != 0]
                if len(nonzero) == 0:
                    p_wilc, w_stat = 1.0, 0.0
                else:
                    abs_ranks = scipy_stats.rankdata(np.abs(nonzero))
                    w_plus  = float(np.sum(abs_ranks[nonzero > 0]))
                    w_minus = float(np.sum(abs_ranks[nonzero < 0]))
                    w_stat  = min(w_plus, w_minus)
                    try:
                        _, p_wilc = scipy_stats.wilcoxon(
                            vf, va, alternative='two-sided',
                            zero_method='wilcox')
                    except Exception:
                        p_wilc = float('nan')

                std_d   = float(np.std(diff, ddof=1))
                cohen_d = (delta / std_d if std_d > 1e-12 else float('nan'))

                sig = _sig_label(p_wilc) if not np.isnan(p_wilc) else 'N/A'
                p_s = f"{p_wilc:.4f}"    if not np.isnan(p_wilc) else "N/A"
                d_s = f"{cohen_d:+.3f}"  if not np.isnan(cohen_d) else "N/A"
                w_s = f"{w_stat:.1f}"

                if metric_key in ('rmse', 'mae'):
                    d_str = f"{delta:+.2f}s"
                elif metric_key == 'mape':
                    d_str = f"{delta:+.2f}%"
                else:
                    d_str = f"{delta:+.4f}"

                print(f"  {ablation_name:<28} {display_name:<8} "
                      f"{d_str:>18}  {w_s:>8}  {p_s:>10}  "
                      f"{sig:>5}  {d_s:>9}")

            print()

        # ── (B) Kruskal-Wallis H: across all 4 ablation variants ─────────────
        print(f"\n  (B) KRUSKAL-WALLIS H TEST  "
              f"— across all ablation variants (k=4, n={n_iters} per group)")
        print(f"  H₀: all variants have the same performance distribution")
        print(f"  {'Metric':<8} {'H statistic':>14}  {'df':>4}  "
              f"{'p-value':>10}  {'Sig':>5}  Interpretation")
        print(f"  {'-' * 80}")

        for display_name, metric_key, higher_better in [
            ('R²',   'r2',   True),
            ('RMSE', 'rmse', False),
            ('MAE',  'mae',  False),
            ('MAPE', 'mape', False),
        ]:
            def _grp(result_list):
                arr = np.array([r.get(split, {}).get(metric_key, float('nan'))
                                for r in result_list], dtype=float)
                return arr[~np.isnan(arr)]

            g_full  = _grp(all_full_results)
            g_noop  = _grp(all_no_operational_results)
            g_now   = _grp(all_no_weather_results)
            g_bl    = _grp(all_baseline_results)

            if min(len(g_full), len(g_noop), len(g_now), len(g_bl)) < 3:
                print(f"  {display_name:<8}  insufficient data")
                continue

            try:
                h_stat, p_kw = scipy_stats.kruskal(g_full, g_noop, g_now, g_bl)
            except Exception as e:
                print(f"  {display_name:<8}  error: {e}")
                continue

            df  = 3   # k − 1 = 4 − 1
            sig = _sig_label(p_kw)
            p_s = f"{p_kw:.4f}"
            h_s = f"{h_stat:.4f}"

            if p_kw < 0.05:
                interp = "≥1 variant significantly differs"
            else:
                interp = "no significant difference"

            print(f"  {display_name:<8} {h_s:>14}  {df:>4}  "
                  f"{p_s:>10}  {sig:>5}  {interp}")

        print(f"\n  NOTE: Kruskal-Wallis confirms whether removing features has any")
        print(f"  global effect. Wilcoxon (A) pinpoints which ablations matter.")


        # ── Feature importance analysis ───────────────────────────────────────
        print_section("FEATURE IMPORTANCE ANALYSIS  (MAGNN-LSTM-MTL)")
        for split in ('Test',):
            full_r2     = np.nanmean([r.get(split, {}).get('r2', float('nan'))
                                      for r in all_full_results])
            no_op_r2    = np.nanmean([r.get(split, {}).get('r2', float('nan'))
                                      for r in all_no_operational_results])
            no_w_r2     = np.nanmean([r.get(split, {}).get('r2', float('nan'))
                                      for r in all_no_weather_results])
            baseline_r2 = np.nanmean([r.get(split, {}).get('r2', float('nan'))
                                      for r in all_baseline_results])

            op_contribution = full_r2 - no_op_r2
            w_contribution  = full_r2 - no_w_r2
            combined        = full_r2 - baseline_r2

            print(f"\n{split.upper()} SET — Feature Contribution Analysis")
            print(f"  Full Model (All features):          {full_r2:.4f}")
            print(f"  No Operational (Weather only):      {no_op_r2:.4f}")
            print(f"  No Weather (Operational only):      {no_w_r2:.4f}")
            print(f"  Baseline (Spatial + Temporal only): {baseline_r2:.4f}")
            print(f"\n  ΔR² Contributions (Full − Ablated):")
            print(f"    Operational features:    {op_contribution:+.4f}")
            print(f"    Weather features:        {w_contribution:+.4f}")
            print(f"    Combined (Op + Weather): {combined:+.4f}")

            if op_contribution > w_contribution:
                print(f"\n Operational features contribute MORE "
                      f"(Δ={op_contribution - w_contribution:+.4f})")
            elif w_contribution > op_contribution:
                print(f"\n  Weather features contribute MORE "
                      f"(Δ={w_contribution - op_contribution:+.4f})")
            else:
                print(f"\n  Both feature groups contribute equally")

            # Bootstrap 95% CI for ΔR² (uses all n iterations as resampling pool)
            if n_iters >= 5:
                rng = np.random.default_rng(42)
                n_boot = 5000
                full_vals = np.array([r.get(split, {}).get('r2', float('nan'))
                                      for r in all_full_results])
                nop_vals  = np.array([r.get(split, {}).get('r2', float('nan'))
                                      for r in all_no_operational_results])
                now_vals  = np.array([r.get(split, {}).get('r2', float('nan'))
                                      for r in all_no_weather_results])
                bl_vals   = np.array([r.get(split, {}).get('r2', float('nan'))
                                      for r in all_baseline_results])

                idx = rng.integers(0, n_iters, size=(n_boot, n_iters))
                op_boot = (full_vals[idx].mean(1) - nop_vals[idx].mean(1))
                w_boot  = (full_vals[idx].mean(1) - now_vals[idx].mean(1))
                co_boot = (full_vals[idx].mean(1) - bl_vals[idx].mean(1))

                def _ci(arr):
                    lo, hi = np.nanpercentile(arr, [2.5, 97.5])
                    return f"[{lo:+.4f}, {hi:+.4f}]"

                print(f"\n  95% Bootstrap CI (n_boot={n_boot} resamples):")
                print(f"    ΔR² Operational: {op_contribution:+.4f}  "
                      f"CI={_ci(op_boot)}")
                print(f"    ΔR² Weather:     {w_contribution:+.4f}  "
                      f"CI={_ci(w_boot)}")
                print(f"    ΔR² Combined:    {combined:+.4f}  "
                      f"CI={_ci(co_boot)}")

    print_section("ABLATION STUDY COMPLETE")
    print(f"  Total iterations: {len(all_full_results)}")
    print(f"  Model:   MAGNN-LSTM-MTL (full pipeline, trip-level predictions)")
    print(f"  Results: outputs/{ALGORITHM_NAME}_*_ablation/")
    print("=" * 80)

if __name__ == '__main__':
    if _sample_fraction_cli is not None:
        Config.sample_fraction = _sample_fraction_cli

    print(f"🔧 Clustering method : {ALGORITHM_NAME}")
    print(f"🔧 Sample fraction   : {Config.sample_fraction}")

    if _mode == '--compare-all':
        main_with_three_way_comparison()
    elif _mode == '--ablation':
        main_with_ablation_study()
    else:
        main()
