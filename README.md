# THS3‑HXIL‑RightOnTrack — MAGNN‑LSTM‑MTL

This repository implements an end‑to‑end spatiotemporal learning pipeline for rail/transit travel‑time modeling. The core workflow is:

1) Load GPS + operational + weather datasets (train/val/test)
2) Cluster “stop / dwell / slowdown” locations (multiple algorithms supported)
3) Convert raw GPS traces into **segments** (origin→destination pairs) with engineered context features
4) Build **graph adjacency matrices** over segment types (geo / distance / social)
5) Train and evaluate MAGNN/MAGTTE + LSTM multi‑task variants
6) Export metrics, plots, and timing comparisons into `outputs/`


---

## Repository structure

Top-level modules:

- `main.py`
  - Primary end‑to‑end pipeline for a single clustering method.
- `compare_algorithms.py`
  - Runs multiple clustering algorithms and produces a unified comparison (metrics + timing).
- `test_clusters.py`
  - Cluster/method experimentation harness (historically used for sweeps).

Core pipeline modules:

- `config.py`
  - Shared configuration, printing helpers (`print_section`), distance helpers (`haversine_meters`), device selection.
- `data_loader.py`
  - CSV loading, chronological sorting/sampling, and known-stop extraction.
- `segments.py`
  - Convert raw GPS pings into segment records and build adjacency matrices over segment types.
- `visualizations.py`
  - Cluster plots, segment plots, and summary plots.

Modeling modules:

- `model.py`
  - Dataset/dataloader utilities, training loop(s), evaluation utilities, and MAGNN/MAGTTE model definitions.
- `mtl.py`
  - Multi‑task learning heads, loss functions, and hierarchical trip prediction components.
- `residual.py`
  - Station/cluster mapping, nearest‑GPS feature lookup, and residual modeling utilities.

Clustering adapters (all expose the same public API):

- `cluster_hdbscan.py`
- `cluster_dbscan.py`
- `cluster_kmeans.py`
- `cluster_gmm.py`
- `cluster_knn.py`

Data / outputs folders:

- `data/` – expected CSV inputs
- `outputs/` – run artifacts (plots, metrics JSON, comparisons)
- `DeepTTE/` – optional baseline model folder (separate pipeline)

---

## Data

### Input files

Expected CSVs (default locations):

- `data/train_data.csv`
- `data/validation_data.csv`
- `data/test_data.csv`
- `data/social_function.csv` (used to build the social adjacency matrix)

The loader expects at minimum:

- Geographic coordinates: `latitude`, `longitude`
- A timestamp column (used for chronological sorting; exact name depends on your loader configuration)

Many parts of the pipeline use optional columns when present:

- Motion/event indicators: `speed_mps`, `is_long_dwell`, `is_slow_speed`, `is_slowdown`
- Delay/operations: `arrivalDelay`, `departureDelay`, `dwellTime_sec`
- Weather: columns like `temperature_2m`, `windspeed_10m`, etc.
- Categorical/binary context: `is_weekend`, `is_peak_hour`

If a required column is missing for a stage, that stage will either fall back to a default behavior or drop the feature (depends on the function).

---

## Clustering layer (`cluster_*.py`)

### Unified adapter API

Each clustering module must expose:

```python
event_driven_clustering_fixed(df, known_stops=None, **optional)
	-> (cluster_centers: np.ndarray shape (N,2),
		station_cluster_ids: set[int])
```

- `cluster_centers[i] = [lat, lon]`
- `station_cluster_ids` indicates which indices in `cluster_centers` correspond to known physical stations (only used by some algorithms).

This shared API allows `main.py` and `compare_algorithms.py` to switch clustering algorithms without changing downstream code.

### Algorithm notes

- **DBSCAN (`cluster_dbscan.py`)**
  - Fully data‑driven number of clusters.
  - Uses a k‑distance elbow method to select `eps`.
  - Ignores `known_stops` (prints a note).
- **KMEANS (`cluster_kmeans.py`)**
  - Uses a default `n_clusters` inside the module.
  - Ignores `known_stops`.
- **GMM (`cluster_gmm.py`)**
  - Uses a default `n_components` / `n_clusters` inside the module.
  - Ignores `known_stops`.
- **HDBSCAN (`cluster_hdbscan.py`)**
  - Density‑based with optional station injection/merging.
  - Requires the `hdbscan` Python package.
- **KNN (`cluster_knn.py`)**
  - Graph clustering of sparse delay/stop events with station injection.

---

## Segmentation (`segments.py`)

Segmentation converts raw GPS points into **origin→destination segments**. 

Key steps:
1. Assign each GPS ping to the nearest cluster (within a configured radius).
2. Identify segment boundaries when the assigned cluster changes OR when authoritative stop metadata indicates a stop sequence transition.
3. Aggregate per-segment features:
   - Target: `duration_sec` (from origin to destination)
   - Operational features (delays, dwell)
   - Weather features
   - Binary flags: weekend/peak hour and event flags

Segmentation is executed for train/val/test and produces a segment dataframe plus a “segment type” encoding used for the graph.

---

## Adjacency matrices / graph construction

`segments.py` builds adjacency matrices over **segment types**. 

Common matrices:

- `adj_geo`: geographic similarity between segment types
- `adj_dist`: connectivity/transition similarity
- `adj_soc`: social-function similarity (based on `data/social_function.csv`)

These matrices are used by graph layers in MAGNN/MAGTTE.

---

## Modeling

### Core model utilities (`model.py`)

`model.py` contains most of the training/evaluation utilities and model definitions. It typically provides:

- Dataset / dataloader wrappers
- Model construction (MAGTTE + graph attention components)
- Training loops (epoch iteration, checkpointing)
- Evaluation helpers returning regression metrics and predictions

### Multi-task learning (`mtl.py`)

`mtl.py` contains components for multi‑task learning such as:

- Multi‑head prediction (e.g., per‑segment + per‑trip)
- Combined loss functions
- Sequence encoders (LSTM) and hierarchical predictors

### Residual / station mapping (`residual.py`)

Utilities for mapping anonymous clusters to known stations and constructing residual features via nearest‑neighbour lookups.

---

## Training & evaluation

Training flow:

1. Build segments + adjacency
2. Create datasets/dataloaders
3. Train for `epochs`
4. Evaluate (train/val/test)
5. Save `metrics.json` + plots

Evaluation returns:

- `loss`
- `r2`, `rmse`, `mae`, `mape`
---

## Inference timing metrics

Timing metrics are intended to separate:

- **Pure model forward time** (batch forward pass)
- **Total inference time** (entire test set)
- **Clustering overhead**
- Derived throughput/latency
- **Total pipeline time** (end‑to‑end)

Standard keys used in comparisons:

- `avg_model_forward_ms`
- `total_inference_time_s`
- `clustering_time_ms`
- `throughput_samples_per_sec`
- `avg_latency_per_sample_ms`
- `total_pipeline_time_ms`

These are surfaced in per‑algorithm `metrics.json` and aggregate comparison CSV/JSON.

---

## Outputs & file formats

All artifacts are written under `outputs/`.

Common patterns:

- Single algorithm run:
  - `outputs/<ALGO>_<timestamp>_i1/`
- Algorithm comparison run:
  - `outputs/algorithm_comparison_<timestamp>/`
	- `KMEANS/metrics.json`, `KNN/metrics.json`, ...
	- `comparison_results.csv`
	- `comparison_results.json`

Per algorithm, expected artifacts should include:

- `metrics.json`
- `*-clusters.png`
- `*-segments.png`
- `*-segment_stats.png`

### `metrics.json` 


```json
{
  "config": { "algorithm": "KMEANS", "sample_fraction": 1.0, "epochs": 50, "batch_size": 64 },
  "results": { "Train": {"r2": 0.0}, "Val": {"r2": 0.0}, "Test": {"r2": 0.0, "inference_timing": { /* ... */ } } },
  "timing": {
	"clustering_time_ms": 0.0,
	"total_pipeline_time_ms": 0.0
  }
}
```

---

## Experiment runners

### `main.py`

Runs a single clustering method end‑to‑end.

Typical usage:

```bash
python main.py <sample_fraction> <clustering algorithm>
python main.py 0.1                    # 10% random sample, hdbscan
python main.py 1.0 knn                # 100% sample, KNN
python main.py 0.5 dbscan             # 50% sample, DBSCAN
python main.py --compare-all          # three-way model comparison
python main.py --ablation             # feature ablation study
python main.py 0.1 knn --compare-all  # all options combined
python main.py 0.1 knn --ablation     # all options combined
```

Where the second argument is usually `sample_fraction`.

### `compare_algorithms.py`

Runs multiple clustering algorithms in one experiment folder and aggregates results.

```bash
python compare_algorithms.py 1.0
```

Depending on your branch/version, it may support:

- `--resume` to skip algorithms that already have completed `metrics.json`
- `--run-folder <path>` to continue in an existing output folder

### `test_clusters.py`

Used for targeted cluster experiments and debugging. If you want it to run the *full MAGNN‑LSTM‑MTL pipeline for all clusters* (rather than cluster-size sweeps), treat it as a convenience wrapper around the same pipeline functions used by `main.py`.

```bash
    python test_clusters.py                  # defaults from config
    python test_clusters.py 0.1              # 10% sample fraction
    python test_clusters.py 0.15             # 15% sample fraction
```
## Troubleshooting

- **It’s running on 20% instead of 100%**
  - Ensure you invoked `sample_fraction=1.0`.
  - Confirm the script you ran actually parses that argument as `sample_fraction`.
  - Ensure you didn’t resume from a folder created with 0.2 earlier.

- **HDBSCAN killed / OOM**
  - HDBSCAN can be memory heavy on large event sets.
  - Consider sampling fraction, reducing event points, or running on a larger machine.

- **Resume marked an incomplete run as complete**
  - Resume logic should validate the contents of `metrics.json`, not just its presence.

