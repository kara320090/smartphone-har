"""Tiny float64 LSTM: NumPy BPTT versus autodiff and central differences.

The intentionally omitted forget-gate sigmoid derivative is a negative control,
not a defect discovered in the production Keras implementation.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from .runtime import configure_runtime, write_json


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def forward_backward(x, y, weights, *, mutant=False):
    kernel, recurrent, bias, dense, output_bias = weights
    batch, timesteps, _ = x.shape
    units = recurrent.shape[0]
    h = np.zeros((batch, units))
    c = np.zeros_like(h)
    history = []
    for t in range(timesteps):
        z = x[:, t] @ kernel + h @ recurrent + bias
        zi, zf, zg, zo = np.split(z, 4, axis=1)
        i, f, g, o = sigmoid(zi), sigmoid(zf), np.tanh(zg), sigmoid(zo)
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        history.append((x[:, t], h, c, i, f, g, o, c_new))
        h, c = h_new, c_new
    logits = h @ dense + output_bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probability = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    loss = -np.log(probability[np.arange(batch), y]).mean()
    dl = probability.copy()
    dl[np.arange(batch), y] -= 1
    dl /= batch
    grad_dense, grad_output_bias = h.T @ dl, dl.sum(0)
    dh, dc = dl @ dense.T, np.zeros_like(c)
    grad_kernel, grad_recurrent, grad_bias = [np.zeros_like(w) for w in weights[:3]]
    for xt, hp, cp, i, f, g, o, ct in reversed(history):
        tanhc = np.tanh(ct)
        do = dh * tanhc
        dc = dc + dh * o * (1 - tanhc ** 2)
        df, di, dg = dc * cp, dc * g, dc * i
        dzf = df if mutant else df * f * (1 - f)
        dz = np.concatenate([di * i * (1 - i), dzf, dg * (1 - g * g), do * o * (1 - o)], axis=1)
        grad_kernel += xt.T @ dz
        grad_recurrent += hp.T @ dz
        grad_bias += dz.sum(0)
        dh, dc = dz @ recurrent.T, dc * f
    return loss, logits, [grad_kernel, grad_recurrent, grad_bias, grad_dense, grad_output_bias]


def check_lstm_gradients():
    configure_runtime(2026)
    rng = np.random.default_rng(2026)
    x = rng.normal(0, 0.4, (2, 3, 2)).astype(np.float64)
    y = np.array([1, 4], dtype=np.int64)
    shapes = [(2, 8), (2, 8), (8,), (2, 6), (6,)]
    weights = [rng.normal(0, 0.3, shape) for shape in shapes]
    model = keras.Sequential([
        keras.Input((3, 2), dtype="float64"),
        keras.layers.LSTM(2, dtype="float64", unit_forget_bias=False, use_cudnn=False),
        keras.layers.Dense(6, dtype="float64")])
    model.set_weights(weights)
    with tf.GradientTape() as tape:
        tf_logits = model(x, training=False)
        tf_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=tf_logits))
    auto = [g.numpy() for g in tape.gradient(tf_loss, model.trainable_variables)]
    loss, logits, manual = forward_backward(x, y, weights)
    _, _, wrong = forward_backward(x, y, weights, mutant=True)
    np.testing.assert_allclose(logits, tf_logits.numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(loss, float(tf_loss), rtol=1e-12, atol=1e-12)
    for a, b in zip(manual, auto):
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-10)
    rows = []
    for step in (1e-4, 1e-5, 1e-6):
        for wi, value in enumerate(weights):
            for index in np.ndindex(value.shape):
                original = value[index]
                value[index] = original + step
                plus = forward_backward(x, y, weights)[0]
                value[index] = original - step
                minus = forward_backward(x, y, weights)[0]
                value[index] = original
                numerical = (plus - minus) / (2 * step)
                analytic = manual[wi][index]
                np.testing.assert_allclose(analytic, numerical, rtol=1e-4, atol=1e-8)
                rows.append({"tensor": wi, "index": str(index), "step": step,
                             "manual": analytic, "autodiff": auto[wi][index], "numerical": numerical,
                             "absolute_error": abs(analytic - numerical)})
    mutation_error = max(float(abs(a-b).max()) for a, b in zip(wrong, auto))
    assert mutation_error > 1e-5
    return {"passed": True, "precision": "float64", "batch": 2, "timesteps": 3,
            "input_features": 2, "units": 2, "classes": 6,
            "parameter_count": sum(w.size for w in weights), "central_difference_checks": len(rows),
            "max_numerical_absolute_error": max(r["absolute_error"] for r in rows),
            "max_autodiff_absolute_error": max(float(abs(a-b).max()) for a, b in zip(manual, auto)),
            "negative_control": {"kind": "intentional mutation", "description": "omit forget gate sigmoid derivative",
                                 "detected": True, "max_autodiff_difference": mutation_error},
            "scope": "tiny educational example; production HAR training uses Keras autodiff"}, rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("reports/lstm"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, rows = check_lstm_gradients()
    write_json(args.output_dir / "math_verification.json", summary)
    with (args.output_dir / "gradient_checks.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(summary)
