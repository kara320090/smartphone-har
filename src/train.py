"""Common trainer: accepts any Keras classifier with (B,6) logits.

Reconstruction models use task_type='reconstruction' and targets equal to input.
No official test evaluation is performed here.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from .adapters.uci_reference import CHANNELS, CLASSES, load_dataset
from .contracts import TrainingData, prepare_data
from .models import build_mlp
from .runtime import configure_runtime, environment_info, file_sha256, write_json
from .training_metrics import classification_metrics, softmax


def make_dataset(x, y, batch_size, training, seed, threads=4):
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).cache()
    if training:
        dataset = dataset.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.deterministic = True
    options.threading.private_threadpool_size = threads
    options.threading.max_intra_op_parallelism = 1
    return dataset.with_options(options).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def predict_batches(model, x, batch_size=256):
    values = np.concatenate([model(x[i:i + batch_size], training=False).numpy()
                             for i in range(0, len(x), batch_size)])
    if not np.isfinite(values).all():
        raise ValueError("Model output contains NaN or Inf")
    return values


class ValidationF1(keras.callbacks.Callback):
    def __init__(self, x, y):
        super().__init__()
        self.x, self.y = x, y

    def on_epoch_end(self, epoch, logs=None):
        logs["val_macro_f1"] = classification_metrics(
            self.y, predict_batches(self.model, self.x))["macro_f1"]


class FiniteLoss(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        if not np.isfinite(logs["loss"]):
            raise FloatingPointError("Non-finite training loss; run is not complete")

    def on_epoch_end(self, epoch, logs=None):
        if not np.isfinite(logs["val_loss"]):
            raise FloatingPointError("Non-finite validation loss; run is not complete")


def _write_predictions(path, data, logits):
    probabilities = softmax(logits)
    headers = ["sample_id", "subject", "true_label", "predicted_label"]
    headers += [f"logit_{i}" for i in range(6)] + [f"probability_{i}" for i in range(6)]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        for i in range(len(logits)):
            writer.writerow([data.validation_ids[i], int(data.validation_subjects[i]),
                             int(data.y_validation[i]), int(probabilities[i].argmax()),
                             *logits[i].tolist(), *probabilities[i].tolist()])


def train_model(model, data: TrainingData, train_config):
    """Train and return an immutable run directory; never overwrite prior runs.

    Seed model construction first with configure_runtime(seed), then call this.
    All run IDs, raw row IDs, validation predictions and environment are saved.
    """
    config = dict(train_config)
    data.validate()
    seed = int(config.get("seed", 2026))
    threads = int(config.get("threads", 4))
    configure_runtime(seed, threads)
    task_type = config.get("task_type", "classification")
    if task_type not in ("classification", "reconstruction"):
        raise ValueError(f"Unsupported task_type: {task_type}")
    if tuple(model.input_shape[1:]) != data.x_fit.shape[1:]:
        raise ValueError("Model and prepared input shapes differ")
    expected_output = (6,) if task_type == "classification" else data.x_fit.shape[1:]
    if tuple(model.output_shape[1:]) != expected_output:
        raise ValueError(f"Expected output {expected_output}, got {model.output_shape}")
    if task_type == "classification" and getattr(model.layers[-1], "activation", None) == keras.activations.softmax:
        raise ValueError("Classifier must output logits, not softmax probabilities")
    epochs = int(config.get("epochs", 40))
    batch_size = int(config.get("batch_size", 64))
    patience = int(config.get("patience", 6))
    learning_rate = float(config.get("learning_rate", 0.001))
    if min(epochs, batch_size, threads) < 1 or patience < 0 or learning_rate <= 0:
        raise ValueError("Invalid training hyperparameters")
    out = Path(config["run_directory"])
    out.mkdir(parents=True, exist_ok=False)
    # Write the effective defaults, rather than depending on library defaults.
    config.update(seed=seed, threads=threads, task_type=task_type, epochs=epochs,
                  batch_size=batch_size, patience=patience, learning_rate=learning_rate,
                  optimizer="Adam", beta_1=0.9, beta_2=0.999, epsilon=1e-7,
                  monitor="val_loss", restore_best_weights=True,
                  smoke=bool(data.smoke), run_directory=str(out))
    write_json(out / "config.json", config)
    write_json(out / "status.json", {"status": "running"})
    try:
        data.preprocessor.save(out / "preprocess.npz")
        write_json(out / "split.json", {
            "split_id": data.split_id, "fit_ids": data.fit_ids.tolist(),
            "validation_ids": data.validation_ids.tolist(),
            "fit_subjects": sorted(set(data.fit_subjects.tolist())),
            "validation_subjects": sorted(set(data.validation_subjects.tolist())),
            "test_evaluated": False})
        loss = (keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                if task_type == "classification" else keras.losses.MeanSquaredError())
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate,
                      beta_1=0.9, beta_2=0.999, epsilon=1e-7), loss=loss,
                      metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
                      if task_type == "classification" else [])
        fit_targets = data.y_fit if task_type == "classification" else data.x_fit
        val_targets = data.y_validation if task_type == "classification" else data.x_validation
        probe_count = min(batch_size, len(data.x_fit))
        with tf.GradientTape() as tape:
            probe_output = model(data.x_fit[:probe_count], training=False)
            probe_loss = loss(fit_targets[:probe_count], probe_output)
        probe_gradients = tape.gradient(probe_loss, model.trainable_variables)
        if (not np.isfinite(float(probe_loss)) or
                any(g is None or not np.isfinite(g.numpy()).all() for g in probe_gradients)):
            raise FloatingPointError("Initial batch has missing/non-finite gradients")
        write_json(out / "initial_gradient_check.json", {
            "passed": True, "batch_size": probe_count, "loss": float(probe_loss),
            "gradient_global_norm": float(tf.linalg.global_norm(probe_gradients)),
            "trainable_tensor_count": len(probe_gradients), "training_mode": False})
        fit_ds = make_dataset(data.x_fit, fit_targets, batch_size, True, seed, threads)
        val_ds = make_dataset(data.x_validation, val_targets, batch_size, False, seed, threads)
        early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min",
                        patience=patience, min_delta=0.0, restore_best_weights=True)
        callbacks = [FiniteLoss()]
        if task_type == "classification":
            callbacks.append(ValidationF1(data.x_validation, data.y_validation))
        callbacks += [early_stop, keras.callbacks.CSVLogger(str(out / "history.csv"))]
        # Includes graph tracing, dataset cache construction, validation and callbacks;
        # excludes downloading/loading, preprocessing, saving and post-fit checks.
        start = time.perf_counter()
        history = model.fit(fit_ds, validation_data=val_ds, epochs=epochs,
                            callbacks=callbacks, shuffle=False, verbose=config.get("verbose", 2))
        elapsed = time.perf_counter() - start
        outputs = predict_batches(model, data.x_validation)
        model.save(out / "model.keras")
        restored = keras.models.load_model(out / "model.keras", compile=False)
        before = outputs[:30]
        after = predict_batches(restored, data.x_validation[:30])
        if task_type == "classification":
            before, after = softmax(before), softmax(after)
        np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)
        if task_type == "classification":
            np.testing.assert_array_equal(before.argmax(1), after.argmax(1))
        write_json(out / "reload_check.json", {
            "passed": True, "sample_count": len(before), "rtol": 1e-5, "atol": 1e-6,
            "max_absolute_difference": float(np.max(np.abs(before - after))),
            "compared": "probabilities" if task_type == "classification" else "reconstruction"})
        metrics = {"split": "validation", "task_type": task_type,
                   "fit_count": len(data.x_fit), "validation_count": len(data.x_validation),
                   "epochs_ran": len(history.history["loss"]),
                   "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
                   "best_val_loss": float(min(history.history["val_loss"])),
                   "train_seconds": elapsed, "parameters": int(model.count_params()),
                   "test_evaluated": False, "smoke": bool(data.smoke)}
        if task_type == "classification":
            metrics.update(classification_metrics(data.y_validation, outputs))
            _write_predictions(out / "validation_predictions.csv", data, outputs)
        else:
            metrics["reconstruction_mse"] = float(np.mean((outputs - data.x_validation) ** 2))
        write_json(out / "metrics.json", metrics)
        write_json(out / "metadata.json", {
            "schema_version": 1, "run_id": out.name, "experiment_id": config.get("experiment_id"),
            "task_type": task_type, "input_kind": data.preprocessor.input_kind,
            "model_input_shape": list(data.x_fit.shape[1:]), "output_shape": list(expected_output),
            "channels": CHANNELS, "classes": CLASSES, "split_id": data.split_id,
            "label_offset_from_original": -1, "sample_row_number_base": 1,
            "preprocessing": "reference_npz_v1", "model_sha256": file_sha256(out / "model.keras"),
            "preprocess_sha256": file_sha256(out / "preprocess.npz"),
            "optimizer_included": True,
            "checkpoint_purpose": "evaluation; optimizer state is not an exact best-epoch resume checkpoint",
            "timing_scope": "model.fit including tracing, cache, validation and callbacks",
            "environment": environment_info()})
        write_json(out / "status.json", {"status": "complete"})
        return out
    except Exception as error:
        write_json(out / "status.json", {"status": "failed", "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("HAR_DATA_ROOT", "data/raw/UCI HAR Dataset")))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed is not None:
        config["seed"] = args.seed
    if args.smoke:
        config["epochs"] = 2
    configure_runtime(config["seed"], config["threads"])
    samples, indices = load_dataset(args.data_root)
    data = prepare_data(samples, indices, config, smoke=args.smoke)
    model = build_mlp(data.x_fit.shape[1:], config["dropout"])
    suffix = "_smoke" if args.smoke else ""
    config["run_directory"] = str(args.runs_dir / f"{config['experiment_id']}_seed{config['seed']}{suffix}")
    print(f"{config['experiment_id']}: fit={len(data.x_fit)}, val={len(data.x_validation)}, shape={data.x_fit.shape[1:]}", flush=True)
    print(train_model(model, data, config), flush=True)


if __name__ == "__main__":
    main()
