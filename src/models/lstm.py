"""Bongheon's sequence LSTM. Independent windows, six output logits."""
from tensorflow import keras


def build_lstm(input_shape=(128, 9), units=64, dropout=0.0, seed=2026):
    if tuple(input_shape) != (128, 9):
        raise ValueError("LSTM expects (128,9) sequences")
    if not isinstance(units, int) or isinstance(units, bool) or units < 1:
        raise ValueError("units must be a positive integer")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0,1)")
    return keras.Sequential([
        keras.Input(shape=input_shape, dtype="float32"),
        keras.layers.LSTM(units, activation="tanh", recurrent_activation="sigmoid",
                          use_bias=True, unit_forget_bias=True, dropout=0.0,
                          recurrent_dropout=0.0, stateful=False,
                          return_sequences=False, unroll=False, name="lstm"),
        keras.layers.Dense(32, activation="relu", name="hidden"),
        keras.layers.Dropout(dropout, seed=seed + 1000, name="head_dropout"),
        keras.layers.Dense(6, name="logits"),
    ], name="sequence_lstm")
