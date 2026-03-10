"""
tr2.py — Two-Stage TrAdaBoost.R2 (Paper Section 3.2, Algorithm 1)

Implements the T.R2 transfer learning wrapper that combines:
  Stage 1: AdaBoost.R2 with Q plain LSTM base learners — adjusts source/target weights
  Stage 2: Freeze source weights, switch to Q LSTM_A learners, fine-tune target weights

Key equations from the paper:
  - Weight initialization: w_i^1 = 1/(n+m) for all instances
  - Source weight decay:   w_i^(t+1) = w_i^t * δ_t^(e_i^t)   where δ_t ∈ (0,1)
  - Target weight update per AdaBoost.R2: β_t = error_t / (1 - error_t)
  - δ_t chosen so target weight sum = m/(n+m) + t/(S-1) * (1 - m/(n+m))

Fix notes vs prior version:
  - δ_t is now always < 1, so high-error source samples are correctly DOWNweighted
  - Inner AdaBoost.R2 loop with Q estimators is now implemented per TrAdaBoost step
  - F-fold cross-validation used for error estimation on target data
"""

import numpy as np
from sklearn.model_selection import KFold
from lstm_a import build_lstm_a, build_plain_lstm, KerasLSTMRegressor


# Number of CV folds for error estimation (Paper Algorithm 1 Step 3)
N_FOLDS = 3


class TrAdaBoostR2:
    """
    Two-Stage TrAdaBoost.R2 — Paper Algorithm 1 (T.R2_LSTM_A only).

    Stage 1: AdaBoost.R2 with Q plain LSTM learners per TrAdaBoost step S.
    Stage 2: Freeze source weights, AdaBoost.R2 with Q LSTM_A learners per step.

    Parameters
    ----------
    n_estimators : int
        Q — number of AdaBoost.R2 base learners per TrAdaBoost step (paper Table 5: 5)
    n_steps : int
        S — number of TrAdaBoost iterations (paper Table 5: 5)
    lstm_units : int
        LSTM hidden units (paper Table 5: 16)
    dropout : float
        Dropout rate (paper Table 5: 0.5)
    epochs : int
        Training epochs per base learner (paper Table 5: 150)
    batch_size : int
        Batch size (paper Table 5: 128)
    input_shape : tuple
        (timesteps, features) for LSTM input
    verbose : int
        Verbosity level
    """

    def __init__(self, n_estimators=5, n_steps=5,
                 lstm_units=16, dropout=0.5, epochs=150, batch_size=128,
                 input_shape=None, verbose=1):
        self.n_estimators = n_estimators
        self.n_steps = n_steps
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.verbose = verbose

        # Storage
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        self.stage1_histories_ = []
        self.stage2_histories_ = []

    def _create_learner(self, build_fn):
        """Create a base learner with the given builder function."""
        return KerasLSTMRegressor(
            build_fn=build_fn,
            input_shape=self.input_shape,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            lstm_units=self.lstm_units,
            dropout=self.dropout,
        )

    def _compute_adjusted_error(self, y_true, y_pred):
        """
        Compute adjusted error e_i per instance (AdaBoost.R2 linear error).
        e_i = |y_true_i - y_pred_i| / max(|y_true - y_pred|)
        Returns array of per-instance errors in [0, 1].
        """
        abs_error = np.abs(y_true - y_pred)
        max_error = abs_error.max()
        if max_error == 0:
            return np.zeros_like(abs_error)
        return abs_error / max_error

    def _cv_error_on_target(self, build_fn, X_target, y_target, sample_weight_target):
        """
        F-fold cross-validation error on target data (Paper Algorithm 1 Step 3).

        Instead of evaluating error on the training set (which is optimistic),
        we use K-fold CV on the target portion to get an honest error estimate.
        """
        n_target = len(X_target)
        if n_target < N_FOLDS:
            # Too few samples for CV — fall back to in-sample
            learner = self._create_learner(build_fn)
            learner.fit(X_target, y_target, sample_weight=sample_weight_target)
            y_pred = learner.predict(X_target)
            errors = self._compute_adjusted_error(y_target, y_pred)
            return np.average(errors, weights=sample_weight_target)

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        oof_errors = np.zeros(n_target)

        for train_idx, val_idx in kf.split(X_target):
            learner = self._create_learner(build_fn)
            w_train = sample_weight_target[train_idx]
            # Normalize fold weights
            w_train = w_train / w_train.sum() * len(w_train)
            learner.fit(X_target[train_idx], y_target[train_idx], sample_weight=w_train)
            y_pred_fold = learner.predict(X_target[val_idx])
            oof_errors[val_idx] = self._compute_adjusted_error(
                y_target[val_idx], y_pred_fold
            )

        return np.average(oof_errors, weights=sample_weight_target)

    def _run_adaboost_r2(self, build_fn, X, y, sample_weights, n_source,
                          X_val=None, y_val=None):
        """
        Inner AdaBoost.R2 loop — trains Q base learners and returns ensemble.

        Paper Algorithm 1 calls AdaBoost.R2 with Q estimators inside each
        TrAdaBoost step. This function implements that inner loop.

        Parameters
        ----------
        build_fn : callable — model builder (build_plain_lstm or build_lstm_a)
        X, y : combined source+target data
        sample_weights : current instance weights
        n_source : number of source samples (for separating source/target)
        X_val, y_val : optional validation data

        Returns
        -------
        ensemble : list of (learner, weight) tuples
        histories : list of training history dicts
        target_error : weighted error on target portion (from CV)
        """
        N = len(X)
        w = sample_weights.copy()
        ensemble = []
        histories = []

        for q in range(self.n_estimators):
            # Normalize weights
            w_norm = w / w.sum()

            # Train base learner
            learner = self._create_learner(build_fn)
            val_data = None
            if X_val is not None and y_val is not None:
                val_data = (X_val, y_val)
            learner.fit(X, y, sample_weight=w_norm, validation_data=val_data)
            if hasattr(learner, 'history'):
                histories.append(learner.history.history)

            # Compute per-instance errors
            y_pred = learner.predict(X)
            errors = self._compute_adjusted_error(y, y_pred)

            # Weighted error (AdaBoost.R2)
            weighted_error = np.sum(w_norm * errors)

            # Avoid degenerate cases
            if weighted_error >= 0.5:
                if self.verbose:
                    print(f"      Estimator {q+1}: error={weighted_error:.4f} >= 0.5, stopping ensemble")
                if len(ensemble) == 0:
                    ensemble.append((learner, 1.0))
                break
            if weighted_error == 0:
                ensemble.append((learner, 1.0))
                break

            # AdaBoost.R2 learner weight: β_q = error / (1 - error)
            beta_q = weighted_error / (1 - weighted_error)
            learner_weight = np.log(1.0 / max(beta_q, 1e-10))
            ensemble.append((learner, learner_weight))

            if self.verbose:
                print(f"      Estimator {q+1}/{self.n_estimators}: "
                      f"error={weighted_error:.4f}, weight={learner_weight:.4f}")

            # Update instance weights for next estimator
            # Well-predicted instances get lower weight
            for i in range(N):
                w[i] *= np.power(beta_q, 1.0 - errors[i])

        # Compute target error using F-fold CV
        target_w = sample_weights[n_source:] / sample_weights[n_source:].sum()
        target_error = self._cv_error_on_target(
            build_fn, X[n_source:], y[n_source:], target_w
        )

        return ensemble, histories, target_error

    def _ensemble_predict(self, ensemble, X):
        """Weighted median prediction from AdaBoost.R2 ensemble."""
        if len(ensemble) == 1:
            return ensemble[0][0].predict(X)

        predictions = np.array([learner.predict(X) for learner, _ in ensemble])
        weights = np.array([w for _, w in ensemble])

        # Weighted median
        n_samples = X.shape[0]
        result = np.zeros(n_samples)
        for i in range(n_samples):
            sorted_idx = np.argsort(predictions[:, i])
            sorted_preds = predictions[sorted_idx, i]
            sorted_weights = weights[sorted_idx]
            cumsum = np.cumsum(sorted_weights)
            median_idx = np.searchsorted(cumsum, cumsum[-1] / 2.0)
            result[i] = sorted_preds[min(median_idx, len(sorted_preds) - 1)]
        return result

    def fit(self, X_source, y_source, X_target, y_target,
            X_val=None, y_val=None):
        """
        Train the Two-Stage TrAdaBoost.R2 model (T.R2_LSTM_A).

        Parameters
        ----------
        X_source : ndarray — Source domain features (3D: trips, timesteps, features)
        y_source : ndarray — Source domain targets
        X_target : ndarray — Target domain features
        y_target : ndarray — Target domain targets
        X_val, y_val : optional validation data

        Returns self
        """
        n = len(X_source)
        m = len(X_target)
        N = n + m

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"TrAdaBoost.R2 — T.R2_LSTM_A")
            print(f"Source: {n}, Target: {m}, Total: {N}")
            print(f"AdaBoost estimators (Q): {self.n_estimators}")
            print(f"TrAdaBoost steps (S): {self.n_steps}")
            print(f"CV folds (F): {N_FOLDS}")
            print(f"{'='*60}")

        # Combine source + target
        X_combined = np.concatenate([X_source, X_target], axis=0)
        y_combined = np.concatenate([y_source, y_target])

        # Initialize equal weights (Paper Algorithm 1, Step 2)
        weights = np.ones(N) / N

        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []

        # ─── STAGE 1: TrAdaBoost with plain LSTM ────────────────────────
        if self.verbose:
            print(f"\n--- STAGE 1: TrAdaBoost with plain LSTM (S={self.n_steps} steps × Q={self.n_estimators} learners) ---")

        for t in range(self.n_steps):
            if self.verbose:
                print(f"\n  Step {t+1}/{self.n_steps}")
                print(f"    Source weight sum: {weights[:n].sum():.4f}")
                print(f"    Target weight sum: {weights[n:].sum():.4f}")

            # Run inner AdaBoost.R2 with Q estimators
            ensemble, histories, target_error = self._run_adaboost_r2(
                build_fn=build_plain_lstm,
                X=X_combined, y=y_combined,
                sample_weights=weights, n_source=n,
            )
            self.stage1_histories_.extend(histories)
            self.estimators_.append(ensemble)
            self.estimator_errors_.append(target_error)

            if self.verbose:
                print(f"    Target CV error: {target_error:.4f}")

            # Source weight update: w_i^(t+1) = w_i^t * δ_t^(e_i^t)
            # δ_t must be in (0, 1) so high-error source samples are DOWNweighted
            # Paper: δ_t = 1 / (1 + sqrt(2 * ln(n) / S))
            delta_t = 1.0 / (1.0 + np.sqrt(2.0 * np.log(max(n, 2)) / max(self.n_steps, 1)))

            # Get predictions from this step's ensemble for error calculation
            y_pred_combined = self._ensemble_predict(ensemble, X_combined)
            errors = self._compute_adjusted_error(y_combined, y_pred_combined)

            # Update source weights: higher error → multiplied by δ_t^e_i → lower weight
            # Since δ_t < 1 and e_i ∈ [0,1]: δ_t^0 = 1 (no change), δ_t^1 = δ_t (max reduction)
            for i in range(n):
                weights[i] *= np.power(delta_t, errors[i])

            # Normalize to achieve desired target fraction schedule
            target_frac = self._compute_target_fraction(n, m, t, self.n_steps)
            weights = self._renormalize_weights(weights, n, target_frac)

        # ─── STAGE 2: Fine-tune with LSTM_A (freeze source weights) ─────
        if self.verbose:
            print(f"\n--- STAGE 2: Fine-tune with LSTM_A (S={self.n_steps} steps × Q={self.n_estimators} learners) ---")
            print(f"  Source weights frozen")

        frozen_source_weights = weights[:n].copy()

        for t in range(self.n_steps):
            # Restore frozen source weights each step
            weights[:n] = frozen_source_weights

            if self.verbose:
                print(f"\n  Step {t+1}/{self.n_steps}")
                print(f"    Source weight sum (frozen): {weights[:n].sum():.4f}")
                print(f"    Target weight sum: {weights[n:].sum():.4f}")

            # Run inner AdaBoost.R2 with Q LSTM_A estimators
            ensemble, histories, target_error = self._run_adaboost_r2(
                build_fn=build_lstm_a,
                X=X_combined, y=y_combined,
                sample_weights=weights, n_source=n,
                X_val=X_val, y_val=y_val,
            )
            self.stage2_histories_.extend(histories)
            self.estimators_.append(ensemble)
            self.estimator_errors_.append(target_error)

            if self.verbose:
                print(f"    Target CV error: {target_error:.4f}")

            # Update ONLY target weights using AdaBoost.R2 rule
            y_pred_combined = self._ensemble_predict(ensemble, X_combined)
            errors = self._compute_adjusted_error(y_combined, y_pred_combined)

            # Only update if model is better than random (error < 0.5).
            # When target_error >= 0.5, beta_t >= 1, which would upweight
            # low-error samples — the opposite of AdaBoost's intent.
            if 0 < target_error < 0.5:
                beta_t = target_error / (1 - target_error)
                for i in range(n, N):
                    weights[i] *= np.power(beta_t, 1.0 - errors[i])

            # Normalize target weights (source stays frozen)
            target_sum = weights[n:].sum()
            if target_sum > 0:
                source_total = frozen_source_weights.sum()
                weights[n:] = weights[n:] / target_sum * (1.0 - source_total)

        # Select best step: argmin(target_error) per paper
        self.best_idx_ = int(np.argmin(self.estimator_errors_))
        self.best_ensemble_ = self.estimators_[self.best_idx_]

        if self.verbose:
            print(f"\n  Best step: index {self.best_idx_}, "
                  f"error {self.estimator_errors_[self.best_idx_]:.4f}")

        return self

    def _compute_target_fraction(self, n, m, t, S):
        """
        Target weight fraction schedule.
        Paper: target_frac = m/(n+m) + t/(S-1) * (1 - m/(n+m))
        Increases from m/(n+m) to 1.0 over S steps.
        """
        if S <= 1:
            return m / (n + m)
        frac = m / (n + m) + t / (S - 1) * (1 - m / (n + m))
        return np.clip(frac, 0.01, 0.99)

    def _renormalize_weights(self, weights, n, target_frac):
        """Renormalize weights so target portion sums to target_frac."""
        source_sum = weights[:n].sum()
        target_sum = weights[n:].sum()
        if source_sum > 0 and target_sum > 0:
            weights[:n] = weights[:n] / source_sum * (1.0 - target_frac)
            weights[n:] = weights[n:] / target_sum * target_frac
        return weights

    def predict(self, X):
        """
        Predict using the best ensemble selected by TrAdaBoost.R2.
        Uses weighted median from the AdaBoost.R2 ensemble.
        """
        return self._ensemble_predict(self.best_ensemble_, X)

    def get_convergence_history(self):
        """Return training loss histories for analysis."""
        return {
            "stage1": self.stage1_histories_,
            "stage2": self.stage2_histories_,
            "errors": self.estimator_errors_,
        }


if __name__ == "__main__":
    np.random.seed(42)
    n_features = 10
    X_src = np.random.randn(50, 4, n_features).astype(np.float32)
    y_src = np.random.randn(50).astype(np.float32) * 50 + 100
    X_tgt = np.random.randn(20, 4, n_features).astype(np.float32)
    y_tgt = np.random.randn(20).astype(np.float32) * 50 + 80

    print("Testing T.R2_LSTM_A...")
    model = TrAdaBoostR2(
        n_estimators=2, n_steps=2,
        input_shape=(4, n_features), epochs=3, batch_size=16, verbose=1
    )
    model.fit(X_src, y_src, X_tgt, y_tgt)
    preds = model.predict(X_tgt[:5])
    print(f"T.R2_LSTM_A predictions: {preds}")
    print("\nTrAdaBoost.R2 module tested successfully!")
