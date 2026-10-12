"""Plan E04 reference implementation for the RNN owner's comparison workflow."""
from tensorflow import keras


def build_cnn(input_shape=(128, 9), seed=2026):
    if tuple(input_shape) != (128, 9):
        raise ValueError("CNN expects (128,9)")
    return keras.Sequential([
        keras.Input(shape=input_shape, dtype="float32"),
        keras.layers.Conv1D(32, 5, padding="same", activation="relu", name="conv1"),
        keras.layers.MaxPooling1D(2, name="pool1"),
        keras.layers.Conv1D(64, 5, padding="same", activation="relu", name="conv2"),
        keras.layers.GlobalAveragePooling1D(name="time_average"),
        keras.layers.Dense(32, activation="relu", name="hidden"),
        keras.layers.Dropout(0.2, seed=seed + 1000, name="head_dropout"),
        keras.layers.Dense(6, name="logits"),
    ], name="sequence_cnn")
