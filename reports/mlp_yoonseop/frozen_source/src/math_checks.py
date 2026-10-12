"""Independent NumPy backprop, finite differences, and float64 autodiff checks."""
import argparse
import csv
from pathlib import Path
import numpy as np
import tensorflow as tf
from .runtime import write_json


def scalar_example():
    x, w, b, target = 2.0, 3.0, 1.0, 10.0
    prediction = x * w + b
    loss = 0.5 * (prediction - target) ** 2
    dw, db = (prediction - target) * x, prediction - target
    w_tf, b_tf = tf.Variable(w, dtype=tf.float64), tf.Variable(b, dtype=tf.float64)
    with tf.GradientTape() as tape:
        tf_loss = 0.5 * (x * w_tf + b_tf - target) ** 2
    gradients = [float(v.numpy()) for v in tape.gradient(tf_loss, [w_tf, b_tf])]
    np.testing.assert_allclose(gradients, [dw, db], rtol=0, atol=1e-12)
    updated = {str(lr): float(0.5 * (x * (w - lr * dw) + b - lr * db - target) ** 2)
               for lr in (0.1, 0.5)}
    composition_vars = [tf.Variable(v, dtype=tf.float64) for v in (-2.0, 5.0, 4.0)]
    with tf.GradientTape() as tape:
        composition = (composition_vars[0] + composition_vars[1]) * composition_vars[2]
    composition_gradients = [float(g) for g in tape.gradient(composition, composition_vars)]
    np.testing.assert_allclose(composition_gradients, [4, 4, 3], rtol=0, atol=1e-12)
    return {"prediction": prediction, "loss": loss, "gradient_w": dw, "gradient_b": db,
            "autodiff": gradients, "updated_loss_by_learning_rate": updated,
            "composition_f_x_y_z": {"input": [-2, 5, 4], "f": float(composition), "gradients": composition_gradients}}


def tiny_problem():
    rng = np.random.default_rng(2026)
    x = np.array([[0.2, -0.4], [0.7, 0.1], [-0.3, 0.8], [0.5, -0.6]], dtype=np.float64)
    y = np.array([0, 1, 1, 0])
    params = {"W1": rng.normal(0, 0.3, (2, 3)), "b1": rng.normal(0, 0.1, (3,)),
              "W2": rng.normal(0, 0.3, (3, 2)), "b2": rng.normal(0, 0.1, (2,))}
    return x, y, params


def numpy_loss_and_gradients(x, y, p):
    hidden = np.tanh(x @ p["W1"] + p["b1"])
    logits = hidden @ p["W2"] + p["b2"]
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -log_probs[np.arange(len(y)), y].mean()
    d_logits = np.exp(log_probs)
    d_logits[np.arange(len(y)), y] -= 1
    d_logits /= len(y)  # Divide once, not again in dW.
    d_hidden = (d_logits @ p["W2"].T) * (1 - hidden ** 2)
    gradients = {"W1": x.T @ d_hidden, "b1": d_hidden.sum(axis=0),
                 "W2": hidden.T @ d_logits, "b2": d_logits.sum(axis=0)}
    return float(loss), gradients


def check_gradients():
    x, y, params = tiny_problem()
    loss, analytical = numpy_loss_and_gradients(x, y, params)
    variables = {k: tf.Variable(v, dtype=tf.float64) for k, v in params.items()}
    with tf.GradientTape() as tape:
        hidden = tf.tanh(tf.constant(x) @ variables["W1"] + variables["b1"])
        logits = hidden @ variables["W2"] + variables["b2"]
        tf_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    autodiff = dict(zip(params, [g.numpy() for g in tape.gradient(tf_loss, list(variables.values()))]))
    rows = []
    for step in (1e-3, 1e-4, 1e-5):
        for name, array in params.items():
            for index in np.ndindex(array.shape):
                original = array[index]
                array[index] = original + step
                plus = numpy_loss_and_gradients(x, y, params)[0]
                array[index] = original - step
                minus = numpy_loss_and_gradients(x, y, params)[0]
                array[index] = original
                numerical = (plus - minus) / (2 * step)
                exact = analytical[name][index]
                automatic = autodiff[name][index]
                relative = abs(numerical - exact) / max(1e-8, abs(numerical) + abs(exact))
                rows.append({"h": step, "parameter": name, "index": str(index),
                             "analytical": float(exact), "finite_difference": float(numerical),
                             "autodiff": float(automatic), "relative_error": float(relative),
                             "autodiff_absolute_error": float(abs(automatic - exact))})
                np.testing.assert_allclose(automatic, exact, rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(numerical, exact, rtol=1e-5, atol=1e-8)
    summary = {"passed": True, "dtype": "float64", "architecture": "2 -> 3 tanh -> 2 logits",
               "parameter_count": sum(a.size for a in params.values()), "loss": loss,
               "max_relative_error_by_h": {
                   str(h): max(r["relative_error"] for r in rows if r["h"] == h)
                   for h in (1e-3, 1e-4, 1e-5)},
               "max_autodiff_absolute_error": max(r["autodiff_absolute_error"] for r in rows)}
    return summary, rows


def run_checks(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, rows = check_gradients()
    summary["scalar_example"] = scalar_example()
    write_json(output_dir / "gradient_checks.json", summary)
    with (output_dir / "gradient_details.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("reports/math"))
    print(run_checks(parser.parse_args().output_dir))
