# T.R2_LSTM_A — Travel Time Prediction via Transfer Learning

**Paper:** *"An instance-based transfer learning model with attention mechanism for freight train travel time prediction in the China–Europe railway express"*
**DOI:** [https://doi.org/10.1016/j.eswa.2024.123989](https://doi.org/10.1016/j.eswa.2024.123989)

This project implements the paper's proposed model — **T.R2_LSTM_A** — as a standalone system. Results are intended for external comparison against a separate project. No baseline models or internal comparisons are included.

---

## Table of Contents

1. [What This Model Does (Plain English)](#1-what-this-model-does-plain-english)
2. [The Problem: Why Transfer Learning?](#2-the-problem-why-transfer-learning)
3. [How the Data Flows Through the System](#3-how-the-data-flows-through-the-system)
4. [The Three Core Ideas](#4-the-three-core-ideas)
5. [Algorithm Walkthrough: Step by Step](#5-algorithm-walkthrough-step-by-step)
6. [File-by-File Breakdown](#6-file-by-file-breakdown)
7. [Hyperparameters (Paper Table 5)](#7-hyperparameters-paper-table-5)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [How to Run](#9-how-to-run)
10. [Output Format](#10-output-format)
11. [Computational Cost](#11-computational-cost)

---

## 1. What This Model Does (Plain English)

Given a bus trip that has passed through several stops, **predict how long (in seconds) the next segment of the trip will take**.

The model looks at the sequence of stops a trip has already visited — GPS coordinates, speed, time of day, day of week, weather, etc. — and outputs a single number: the predicted `travel_time_sec` for the last segment in the sequence.

---

## 2. The Problem: Why Transfer Learning?

We have two datasets from **different domains** (e.g., different routes, time periods, or cities):

| Dataset | Role | Size | Description |
|---|---|---|---|
| `train_data.csv` | **Source domain** | Large | Abundant historical data, but from a different context |
| `test_data.csv` | **Target domain** | Small | The context we actually care about, but limited data |

A model trained only on the small target domain would underfit. A model trained only on the large source domain would learn the wrong patterns. **Transfer learning** bridges this gap: it uses the source domain's volume while gradually learning to trust the target domain's patterns more.

The specific technique is **TrAdaBoost.R2** — a boosting-based transfer learning algorithm that assigns weights to every training sample. Over multiple rounds, it:
- **Downweights** source samples that disagree with the target domain
- **Upweights** target samples that the current model gets wrong

This way, the model automatically figures out which source data is useful and which is misleading.

---

## 3. How the Data Flows Through the System

```
Raw CSV files (GPS observations)
        │
        ▼
┌─────────────────────────────────────────────────┐
│  STEP 1: Sequence Construction (data_loader.py) │
│                                                 │
│  Raw rows are grouped by (trip_id, stop).       │
│  Multiple GPS pings per stop are aggregated     │
│  into one feature vector per stop.              │
│  Stops are arranged into a time-ordered         │
│  sequence per trip.                             │
│                                                 │
│  Result: 3D array (trips, stops, features)      │
│  Shape example: (2260, 12, 34)                  │
│    - 2,260 trips                                │
│    - 12 stops per trip (padded/truncated)        │
│    - 34 features per stop                       │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  STEP 2: Min-Max Normalization (Equation 10)    │
│                                                 │
│  Both X (features) and y (target) are scaled    │
│  to [0, 1] using MinMaxScaler fit on the        │
│  source domain.                                 │
│                                                 │
│  Why: LSTM training converges faster when all   │
│  inputs are on the same scale. The y_scaler is  │
│  saved to inverse-transform predictions back    │
│  to seconds later.                              │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  STEP 3: T.R2_LSTM_A Training (tr2.py)          │
│                                                 │
│  Two-stage boosted ensemble. See Section 5      │
│  below for the full algorithm walkthrough.      │
│                                                 │
│  Input: scaled X and y from source + target     │
│  Output: an ensemble of LSTM models with        │
│          learned weights                        │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  STEP 4: Predict & Inverse-Transform            │
│                                                 │
│  The ensemble predicts on each split.           │
│  Predictions are in [0, 1] (normalized).        │
│  y_scaler.inverse_transform() converts them     │
│  back to raw seconds for meaningful metrics.    │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  STEP 5: Evaluate (evaluate.py)                 │
│                                                 │
│  Compute R², MAE, MAPE, MSE, RMSE on raw        │
│  seconds. Save results.json.                    │
└─────────────────────────────────────────────────┘
```

### What the model sees (input)

Each training sample is one **trip** represented as a 2D matrix:

```
         Feature 1   Feature 2   ...   Feature 34
Stop 1  [  0.23,      0.87,      ...,   0.12    ]
Stop 2  [  0.31,      0.65,      ...,   0.45    ]
  ...
Stop 12 [  0.55,      0.44,      ...,   0.78    ]
```

The 34 features per stop include things like GPS coordinates, speed, distance, time-of-day encodings, day-of-week flags, and other engineered features. The LSTM reads this sequence stop-by-stop, building up an understanding of how the trip is progressing.

### What the model predicts (output)

A single scalar: `travel_time_sec` — the travel time in seconds for the last segment of the trip. After inverse-transforming from [0, 1] back to raw seconds, this is directly comparable to the actual recorded travel time.

### What the metrics measure

All metrics compare **predicted seconds vs. actual seconds** on held-out data the model never saw during training:
- **MAE:** average absolute error in seconds
- **MAPE:** average percentage error (e.g., "off by 30% on average")
- **RMSE:** penalizes large errors more heavily than MAE
- **R²:** how much variance the model explains (1.0 = perfect, 0.0 = no better than predicting the mean)

---

## 4. The Three Core Ideas

### Idea 1: LSTM_A — Bidirectional LSTM with Pairwise Attention

The base prediction model (`lstm_a.py`). Three steps:

**Step A — Bidirectional LSTM (Equation 6):**
Two stacked Bidirectional LSTM layers process the stop sequence in both directions. At each stop `t`, the hidden state captures information from all past stops (forward) and all future stops (backward):

```
h_t = [ h_forward_t ⊕ h_backward_t ]
```

This gives the model context about the entire trip, not just what came before.

**Step B — Pairwise Attention (Equations 7-9):**
For every pair of stops `(t, t')`, the model computes an attention score: "how relevant is stop `t'` when predicting from the perspective of stop `t`?"

```
q(t,t') = tanh(W_q · h_t + W_k · h_t' + b)     ← combine the two stops
α(t,t') = sigmoid(W · q(t,t') + b_α)             ← raw attention score
α_t     = softmax(α(t, :))                        ← normalize across all t'
```

This is different from standard attention — it explicitly models the **interaction** between every pair of stops, not just query-key similarity.

**Step C — Context Aggregation:**
The attention-weighted hidden states are averaged into a single context vector, which is fed to a Dense output layer to produce the travel time prediction.

### Idea 2: AdaBoost.R2 — Ensemble of Weak Learners

Instead of training one LSTM, the algorithm trains **Q = 5 LSTMs per step** and combines them into an ensemble via AdaBoost.R2:

1. Train LSTM #1 on weighted data
2. Compute per-sample errors
3. Increase weights on samples that were predicted poorly
4. Train LSTM #2 on the reweighted data (it focuses on the hard cases)
5. Repeat for Q rounds
6. Combine all Q models using **weighted median** prediction

The result is an ensemble that's more robust than any single LSTM.

### Idea 3: TrAdaBoost.R2 — Two-Stage Transfer Learning

The outer loop that manages source vs. target domain weights across **S = 5 steps**:

**Stage 1 (Plain LSTM):** Coarse transfer. Each step:
- Trains an AdaBoost.R2 ensemble of Q plain LSTMs
- Computes errors on source samples
- **Downweights source samples** with high error (they're misleading)
- Gradually increases the total weight on target samples

**Stage 2 (LSTM_A):** Fine-tuning. Source weights are **frozen** from Stage 1. Each step:
- Trains an AdaBoost.R2 ensemble of Q LSTM_A models
- Updates **only target weights** — focusing on hard target samples
- Source domain provides stable background knowledge

At the end, the best-performing ensemble (lowest target error) is selected as the final model.

---

## 5. Algorithm Walkthrough: Step by Step

Here's exactly what happens when you run `python main.py`:

```
1.  Load train_data.csv, test_data.csv, validation_data.csv, holdout_data.csv
2.  Compute MAX_SEQUENCE_LENGTH from the 95th percentile of stops-per-trip
3.  For each CSV:
      - Group raw GPS rows by (trip_id, trip_stop_sequence)
      - Aggregate into one feature vector per stop
      - Pad/truncate to MAX_SEQUENCE_LENGTH stops
      - Extract target = travel_time_sec of last stop
4.  Optionally subsample (controlled by SAMPLE_FRACTION)
5.  Min-Max scale X and y to [0, 1], fit on source domain

6.  Initialize sample weights: w_i = 1/(n+m) for all n source + m target samples

    ── STAGE 1: S=5 steps with Plain LSTM ──────────────────────
7.  For each TrAdaBoost step t = 1..5:
      a. Run inner AdaBoost.R2 with Q=5 plain LSTM estimators:
           - Train LSTM on weighted combined data
           - Compute weighted error
           - If error ≥ 0.5: stop (model is worse than random)
           - Otherwise: add to ensemble, reweight samples
      b. Estimate target error via 3-fold cross-validation
      c. Compute δ_t = 1/(1 + √(2·ln(n)/S))      — always < 1
      d. Update source weights: w_i *= δ_t^(error_i)
           High-error source → multiplied by small number → downweighted
           Low-error source → multiplied by ~1 → kept
      e. Renormalize: target weight fraction increases each step

    ── STAGE 2: S=5 steps with LSTM_A ─────────────────────────
8.  Freeze source weights from end of Stage 1
9.  For each TrAdaBoost step t = 1..5:
      a. Run inner AdaBoost.R2 with Q=5 LSTM_A estimators
      b. Estimate target error via 3-fold cross-validation
      c. Update only target weights (if error < 0.5):
           β_t = error/(1 - error)
           Well-predicted targets → downweighted (focus on hard cases)
      d. Renormalize target weights

10. Select best ensemble: argmin(target_error) across all 10 steps
11. Predict on train/val/holdout using the best ensemble's weighted median
12. Inverse-transform predictions from [0, 1] back to seconds
13. Compute R², MAE, MAPE, MSE, RMSE on raw seconds
14. Save results.json to outputs/T.R2_LSTM_A_{timestamp}/
```

---

## 6. File-by-File Breakdown

```
T.R2_LSTM_A/
│
├── main.py              Entry point. Parses SAMPLE_FRACTION, calls train.py,
│                        prints results. Run with: python main.py 0.1
│
├── train.py             Training pipeline orchestrator. Loads data, scales it,
│                        trains the model, inverse-transforms predictions,
│                        computes metrics, saves results.json.
│
├── tr2.py               The TrAdaBoost.R2 algorithm (Paper Algorithm 1).
│                        Implements both stages, weight updates, inner
│                        AdaBoost.R2 loop, and ensemble prediction.
│
├── lstm_a.py            Neural network definitions:
│                        - LSTM_A: Bidirectional LSTM + Pairwise Attention
│                        - Plain LSTM: simpler Stage 1 learner
│                        - KerasLSTMRegressor: sklearn-compatible wrapper
│
├── data_loader.py       Data loading and preprocessing:
│                        - CSV → sequence construction (grouping by trip)
│                        - Dynamic MAX_SEQUENCE_LENGTH from data
│                        - Min-Max scaling for X and y
│
├── evaluate.py          Metric computation (R², MAE, MAPE, MSE, RMSE)
│                        and JSON result formatting.
│
├── data/
│   ├── train_data.csv        Source domain (large)
│   ├── test_data.csv         Target domain
│   ├── validation_data.csv   Validation during training
│   └── holdout_data.csv      Final held-out test set
│
└── outputs/
    └── T.R2_LSTM_A_{timestamp}/
        └── results.json      Metrics + config for each run
```

---

## 7. Hyperparameters (Paper Table 5)

| Parameter | Value | What It Controls |
|---|---|---|
| LSTM units | 16 | Hidden state dimension in each LSTM layer |
| Dropout | 0.5 | Regularization rate between layers |
| Optimizer | Adam | Gradient descent variant |
| Learning rate | 0.001 | Adam step size |
| Epochs | 150 | Training iterations per base learner |
| Batch size | 128 | Samples per gradient update |
| S (TrAdaBoost steps) | 5 | Outer transfer learning iterations per stage |
| Q (AdaBoost estimators) | 5 | Inner boosting rounds per TrAdaBoost step |
| F (CV folds) | 3 | Cross-validation folds for error estimation |
| Attention units | 16 | Internal dimension of the pairwise attention layer |
| Sequence length | 12 | Stops per trip (computed from 95th percentile of data) |

---

## 8. Evaluation Metrics

All metrics are computed on **raw seconds** after inverse-transforming predictions from [0, 1] back to the original scale.

| Metric | Formula | Interpretation |
|---|---|---|
| **MAE** (Eq. 11) | `mean(\|ŷ - y\|)` | Average error in seconds |
| **MAPE** (Eq. 12) | `mean(\|ŷ - y\| / y) × 100` | Average percentage error |
| **MSE** (Eq. 13) | `mean((ŷ - y)²)` | Penalizes large errors quadratically |
| **RMSE** (Eq. 14) | `√MSE` | Error in seconds, sensitive to outliers |
| **R²** | `1 - SS_res/SS_tot` | Variance explained (1.0 = perfect) |

---

## 9. How to Run

```bash
# Smoke test (1% of data — fast, for checking the pipeline works)
python main.py 0.01

# Quick test (10% of data)
python main.py 0.1

# Full run (100% of data — very slow, see Section 11)
python main.py 1.0
```

The `SAMPLE_FRACTION` can also be set directly in `main.py` (line 44) instead of passing it as a command-line argument.

### Data splits used for evaluation

| Split | File | Role |
|---|---|---|
| Train | `train_data.csv` | Source domain — used for training |
| Validation | `validation_data.csv` | Monitor overfitting during training |
| Test | `holdout_data.csv` | **Final evaluation** — never seen during training |

---

## 10. Output Format

Each run creates a timestamped folder under `outputs/`:

```
outputs/T.R2_LSTM_A_20260308_191328/results.json
```

The JSON file contains:

```json
{
  "method": "T.R2_LSTM_A",
  "config": {
    "sample_fraction": 1.0,
    "sequence_length": 12,
    "n_features": 34,
    "lstm_units": 16,
    "dropout": 0.5,
    "n_estimators": 5,
    "epochs": 150,
    "batch_size": 128,
    "tr2_steps": 5,
    "learning_rate": 0.001,
    "train_trips": 2260,
    "val_trips": 846,
    "test_trips": 850
  },
  "timestamp": "2026-03-08 19:13:28",
  "Train": {
    "R2": "0.8521",
    "RMSE": "12.34 seconds",
    "MAE": "8.50 seconds",
    "MAPE": "25.30%"
  },
  "Val": {
    "R2": "0.7102",
    "RMSE": "15.26 seconds",
    "MAE": "11.04 seconds",
    "MAPE": "32.51%"
  },
  "Test": {
    "R2": "0.7458",
    "RMSE": "14.29 seconds",
    "MAE": "10.05 seconds",
    "MAPE": "30.88%"
  }
}
```

*(Values above are illustrative, not actual results.)*

---

## 11. Computational Cost

The T.R2_LSTM_A algorithm trains many LSTM models internally as part of its boosting ensemble. Here's the breakdown:

```
Stage 1:  S × Q = 5 × 5 = 25 plain LSTM models
Stage 2:  S × Q = 5 × 5 = 25 LSTM_A models
CV error: S × F = 5 × 3 = 15 models per stage (for error estimation)
                              ──────
                    Total:    ~80 LSTM model fits
                    Each:     150 epochs
```

These are **not** baseline comparison models. They are all internal building blocks of the single T.R2_LSTM_A prediction. The boosting ensemble needs multiple weak learners to produce one strong prediction — this is fundamental to how AdaBoost works.

### Estimated wall-clock times (approximate)

| `SAMPLE_FRACTION` | Data size | Estimated time |
|---|---|---|
| 0.01 | ~23 trips | 5–15 minutes |
| 0.1 | ~230 trips | 30–90 minutes |
| 0.5 | ~1,130 trips | 3–8 hours |
| 1.0 | ~2,260 trips | 8–24+ hours |

Times depend heavily on hardware. A GPU will be significantly faster than CPU-only.

---

## Reference

```bibtex
@article{T.R2_LSTM_A,
  title   = {An instance-based transfer learning model with attention mechanism
             for freight train travel time prediction in the China–Europe
             railway express},
  journal = {Expert Systems with Applications},
  year    = {2024},
  doi     = {10.1016/j.eswa.2024.123989}
}
```

