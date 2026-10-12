"""Yoo Jaeyoon's SimpleRNN: independent 128-step windows, six logits."""
from tensorflow import keras


def build_rnn(input_shape=(128, 9), units=32, dropout=0.0, seed=2026):
    if tuple(input_shape) != (128, 9):
        raise ValueError("RNN expects (128,9) sequences")
    if not isinstance(units, int) or isinstance(units, bool) or units < 1:
        raise ValueError("units must be a positive integer")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0,1)")
    return keras.Sequential([
        keras.Input(shape=input_shape, dtype="float32"),
        keras.layers.SimpleRNN(units, activation="tanh", use_bias=True,
            kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal",
            bias_initializer="zeros", stateful=False, return_sequences=False,
            dropout=0.0, recurrent_dropout=0.0, unroll=False, name="simple_rnn"),
        keras.layers.Dropout(dropout, seed=seed + 1000, name="head_dropout"),
        keras.layers.Dense(6, name="logits"),
    ], name="sequence_rnn")
