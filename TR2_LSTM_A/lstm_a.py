"""
lstm_a.py — LSTM_A: Custom Bidirectional Attention LSTM (Paper Section 3.3)

Implements the core base learner described in the paper:
  Step A: Bidirectional two-layer LSTM (Equation 6)
  Step B: Pairwise attention matrix computation (Equations 7-9)
  Step C: Attention-weighted context for final prediction

Also provides:
  - Plain LSTM (used internally as the Stage 1 base learner in T.R2)

No baseline models (CLSTM, LSTM_Attention, etc.) are included here.

Paper hyperparameters (Table 5):
  LSTM neurons: 16, Dropout: 0.5, Optimizer: Adam, Epochs: 150, Batch: 128
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info/warnings

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, backend as K


# ─────────────────────────────────────────────────────────────────────────────
# Custom Attention Layer — Paper Equations 7, 8, 9
# ─────────────────────────────────────────────────────────────────────────────

class PairwiseAttentionLayer(layers.Layer):
    """
    Custom attention mechanism from Paper Section 3.3.

    For every pair of timesteps t and t', compute:
      q(t,t') = tanh(W_q * h_t + W_k * h_t' + b_q)         (Eq. 7)
      α(t,t') = sigmoid(W * q(t,t') + b_α)                  (Eq. 8)
      α_t     = softmax(α(t,:)) over all t'                  (Eq. 9)

    The context vector is then the attention-weighted sum of hidden states.
    """

    def __init__(self, units, **kwargs):
        super(PairwiseAttentionLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # input_shape: (batch, timesteps, hidden_dim)
        hidden_dim = input_shape[-1]

        # W_q: projects h_t (Eq. 7)
        self.W_q = self.add_weight(
            name="W_q", shape=(hidden_dim, self.units),
            initializer="glorot_uniform", trainable=True
        )
        # W_k: projects h_t' (Eq. 7)
        self.W_k = self.add_weight(
            name="W_k", shape=(hidden_dim, self.units),
            initializer="glorot_uniform", trainable=True
        )
        # b_q: bias for q computation (Eq. 7)
        self.b_q = self.add_weight(
            name="b_q", shape=(self.units,),
            initializer="zeros", trainable=True
        )
        # W: projects q to scalar attention score (Eq. 8)
        self.W = self.add_weight(
            name="W", shape=(self.units, 1),
            initializer="glorot_uniform", trainable=True
        )
        # b_alpha: bias for attention score (Eq. 8)
        self.b_alpha = self.add_weight(
            name="b_alpha", shape=(1,),
            initializer="zeros", trainable=True
        )
        super(PairwiseAttentionLayer, self).build(input_shape)

    def call(self, h):
        """
        Parameters
        ----------
        h : Tensor of shape (batch, timesteps, hidden_dim)
            Concatenated bidirectional LSTM hidden states.

        Returns
        -------
        context : Tensor of shape (batch, hidden_dim)
            Attention-weighted context vector.
        """
        # h shape: (batch, T, D)
        T = tf.shape(h)[1]

        # Compute projections for all timesteps
        # h_q: (batch, T, units) — projection of h_t
        h_q = tf.matmul(h, self.W_q)  # (batch, T, units)
        # h_k: (batch, T, units) — projection of h_t'
        h_k = tf.matmul(h, self.W_k)  # (batch, T, units)

        # Expand dims for broadcasting pairwise computation
        # h_q_expanded: (batch, T, 1, units)
        h_q_expanded = tf.expand_dims(h_q, axis=2)
        # h_k_expanded: (batch, 1, T, units)
        h_k_expanded = tf.expand_dims(h_k, axis=1)

        # Equation 7: q(t,t') = tanh(W_q * h_t + W_k * h_t' + b_q)
        # q shape: (batch, T, T, units)
        q = tf.nn.tanh(h_q_expanded + h_k_expanded + self.b_q)

        # Equation 8: α(t,t') = sigmoid(W * q(t,t') + b_α)
        # score shape: (batch, T, T, 1) → squeeze → (batch, T, T)
        score = tf.nn.sigmoid(tf.matmul(q, self.W) + self.b_alpha)
        score = tf.squeeze(score, axis=-1)  # (batch, T, T)

        # Equation 9: α_t = softmax(α(t,:)) — normalize over t' dimension
        alpha = tf.nn.softmax(score, axis=-1)  # (batch, T, T)

        # Apply attention: weighted sum of h over t' for each t
        # context_per_t: (batch, T, D)
        context_per_t = tf.matmul(alpha, h)  # (batch, T, D)

        # Aggregate over timesteps (mean pooling over t)
        context = tf.reduce_mean(context_per_t, axis=1)  # (batch, D)

        return context

    def get_config(self):
        config = super(PairwiseAttentionLayer, self).get_config()
        config.update({"units": self.units})
        return config


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────

def build_lstm_a(input_shape, lstm_units=16, dropout=0.5, attention_units=16,
                 learning_rate=0.001):
    """
    Build the LSTM_A model — Paper Section 3.3, the core base learner.

    Architecture:
      1. Bidirectional LSTM (2 layers) → h_t = [h_forward ⊕ h_backward] (Eq. 6)
      2. Pairwise Attention Layer (Eqs. 7-9)
      3. Dropout
      4. Dense output layer

    Parameters
    ----------
    input_shape : tuple
        (timesteps, features)
    lstm_units : int
        Number of LSTM units (paper Table 5: 16)
    dropout : float
        Dropout rate (paper Table 5: 0.5)
    attention_units : int
        Internal dimension of attention layer
    learning_rate : float
        Adam optimizer learning rate

    Returns
    -------
    keras.Model
    """
    inputs = layers.Input(shape=input_shape)

    # Step A: Bidirectional two-layer LSTM (Equation 6)
    # Layer 1: Bidirectional LSTM, return sequences for layer 2
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, activation="tanh", return_sequences=True,
                    name="lstm_fwd_1"),
        backward_layer=layers.LSTM(lstm_units, activation="tanh",
                                   return_sequences=True, go_backwards=True,
                                   name="lstm_bwd_1"),
        name="bilstm_1"
    )(inputs)
    x = layers.Dropout(dropout)(x)

    # Layer 2: Bidirectional LSTM, return sequences for attention
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, activation="tanh", return_sequences=True,
                    name="lstm_fwd_2"),
        backward_layer=layers.LSTM(lstm_units, activation="tanh",
                                   return_sequences=True, go_backwards=True,
                                   name="lstm_bwd_2"),
        name="bilstm_2"
    )(x)
    x = layers.Dropout(dropout)(x)
    # After Bidirectional: h_t = [h_forward_t ⊕ h_backward_t], shape = (batch, T, 2*lstm_units)

    # Step B & C: Pairwise Attention (Equations 7-9)
    context = PairwiseAttentionLayer(units=attention_units, name="pairwise_attention")(x)

    # Output layer — linear activation for regression on [0,1] normalized target
    context = layers.Dropout(dropout)(context)
    outputs = layers.Dense(1, name="output")(context)

    model = Model(inputs=inputs, outputs=outputs, name="LSTM_A")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model


def build_plain_lstm(input_shape, lstm_units=16, dropout=0.5, learning_rate=0.001):
    """
    Plain single-layer LSTM — used as the Stage 1 base learner in T.R2
    before switching to LSTM_A in Stage 2.

    Architecture: LSTM → Dropout → Dense(1)
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(lstm_units, activation="tanh", return_sequences=False,
                    name="lstm")(inputs)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="PlainLSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Keras wrapper for sklearn compatibility (TrAdaBoost)
# ─────────────────────────────────────────────────────────────────────────────

class KerasLSTMRegressor:
    """
    Sklearn-compatible wrapper for Keras LSTM models.
    Supports sample_weight for boosting algorithms.
    Expects 3D input: (samples, timesteps, features).
    """

    def __init__(self, build_fn, input_shape, epochs=150, batch_size=128,
                 verbose=0, **build_kwargs):
        self.build_fn = build_fn
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.build_kwargs = build_kwargs
        self.model = None

    def fit(self, X, y, sample_weight=None, validation_data=None):
        self.model = self.build_fn(self.input_shape, **self.build_kwargs)

        kwargs = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "verbose": self.verbose,
        }
        if sample_weight is not None:
            kwargs["sample_weight"] = sample_weight
        if validation_data is not None:
            kwargs["validation_data"] = validation_data

        self.history = self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()

    def get_params(self, deep=True):
        return {
            "build_fn": self.build_fn,
            "input_shape": self.input_shape,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "verbose": self.verbose,
            **self.build_kwargs
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


if __name__ == "__main__":
    # Quick test with dummy 3D data (proper sequence input)
    np.random.seed(42)
    X_dummy = np.random.randn(100, 8, 34).astype(np.float32)
    y_dummy = np.random.randn(100).astype(np.float32)

    print("Testing LSTM_A model...")
    model_a = build_lstm_a(input_shape=(8, 34))
    model_a.summary()
    model_a.fit(X_dummy, y_dummy, epochs=2, batch_size=32, verbose=1)
    preds = model_a.predict(X_dummy[:5], verbose=0)
    print(f"LSTM_A predictions: {preds.flatten()}")

    print("\nTesting Plain LSTM model (Stage 1 learner)...")
    model_lstm = build_plain_lstm(input_shape=(8, 34))
    model_lstm.fit(X_dummy, y_dummy, epochs=2, batch_size=32, verbose=0)
    print("Plain LSTM OK")

    print("\nAll models tested successfully!")
