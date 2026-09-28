"""The two MLP baselines owned by part 02. Outputs are six logits."""
from tensorflow import keras


def build_mlp(input_shape=(561,), dropout=0.0):
    shape = tuple(input_shape)
    if not (len(shape) == 1 and shape[0] > 0) and shape != (128, 9):
        raise ValueError("MLP expects feature vectors or (128,9) sequences")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0,1)")
    layers = [keras.Input(shape=shape, dtype="float32")]
    if len(shape) == 2:
        layers.append(keras.layers.Flatten(name="flatten_sequence"))
    for i, width in enumerate((128, 64), start=1):
        layers.append(keras.layers.Dense(width, activation="relu", name=f"hidden_{i}"))
        if dropout:
            layers.append(keras.layers.Dropout(dropout, name=f"dropout_{i}"))
    layers.append(keras.layers.Dense(6, name="logits"))
    return keras.Sequential(layers, name="sequence_mlp" if len(shape) == 2 else "feature_mlp")
