"""Explicit error-injection controls; these are not claimed as historical accidents."""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/process_audit"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(41)
x = rng.normal(size=(8, 3))
y = np.arange(8) % 6
for attempt in range(10000):
    params = [rng.normal(0, .3, (3, 4)), rng.normal(0, .2, 4),
              rng.normal(0, .3, (4, 6)), rng.normal(0, .2, 6)]
    preactivation = x @ params[0] + params[1]
    if np.min(np.abs(preactivation)) > .03 and ((preactivation > 0).any(0) & (preactivation < 0).any(0)).all():
        break
else:
    raise RuntimeError("Could not construct a gradient fixture away from ReLU kinks")


def loss_and_gradient(p):
    w1, b1, w2, b2 = p
    a = x @ w1 + b1
    h = np.maximum(a, 0)
    z = h @ w2 + b2
    shifted = z - z.max(1, keepdims=True)
    log_prob = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
    loss = -log_prob[np.arange(len(y)), y].mean()
    dz = np.exp(log_prob)
    dz[np.arange(len(y)), y] -= 1
    dz /= len(y)
    da = (dz @ w2.T) * (a > 0)
    return loss, [x.T @ da, da.sum(0), h.T @ dz, dz.sum(0)]


loss, analytic = loss_and_gradient(params)
variables = [tf.Variable(p, dtype=tf.float64) for p in params]
with tf.GradientTape() as tape:
    hidden = tf.nn.relu(tf.constant(x) @ variables[0] + variables[1])
    logits = hidden @ variables[2] + variables[3]
    automatic_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
automatic = [g.numpy() for g in tape.gradient(automatic_loss, variables)]
auto_error = max(float(np.max(np.abs(a-b))) for a,b in zip(analytic, automatic))
np.testing.assert_allclose(float(automatic_loss), loss, rtol=1e-12)
for a,b in zip(analytic, automatic): np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-10)
relative = {}
for step in [1e-3, 1e-4, 1e-5]:
    errors = []
    for p, g in zip(params, analytic):
        for idx in np.ndindex(p.shape):
            old = p[idx]
            p[idx] = old + step; high = loss_and_gradient(params)[0]
            p[idx] = old - step; low = loss_and_gradient(params)[0]
            p[idx] = old
            num = (high-low)/(2*step)
            np.testing.assert_allclose(num,g[idx],atol=1e-8,rtol=1e-4)
            errors.append(abs(num-g[idx])/max(1e-8,abs(num)+abs(g[idx])))
    relative[str(step)] = float(max(errors))

# Deliberate mutation: forget / B in a gradient for a MEAN loss.
wrong = [g*len(y) for g in analytic]
detected = False
try:
    for a,b in zip(wrong, automatic): np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-10)
except AssertionError: detected = True
assert detected

# The central difference at ReLU(0) is 0.5, but TF chooses subgradient 0.
t = tf.Variable(0.0, dtype=tf.float64)
with tf.GradientTape() as tape: v = tf.nn.relu(t)
kink_auto = float(tape.gradient(v,t))
kink_numeric = (max(1e-4,0)-max(-1e-4,0))/(2e-4)

# Deliberate naive softmax overflow, followed by the stable equivalent.
large = np.array([[1000.,1001.,999.,-1000.,0.,1002.]])
with np.errstate(over='ignore',invalid='ignore'):
    naive = np.exp(large)/np.exp(large).sum(1,keepdims=True)
exp_shift = np.exp(large-large.max(1,keepdims=True))
stable = exp_shift/exp_shift.sum(1,keepdims=True)
assert np.isfinite(stable).all()
np.testing.assert_allclose(stable.sum(1),1)
result = {"timestamp_utc":datetime.now(timezone.utc).isoformat(),
 "control_type":"Deliberately constructed numerical and gradient controls, not HAR training incidents",
 "relu_network":{"architecture":"3 -> 4 ReLU -> 6 logits", "batch_size":8,
   "parameters":sum(p.size for p in params),"loss":float(loss),
   "minimum_absolute_hidden_preactivation":float(np.min(np.abs(preactivation))),
   "max_numpy_tf_absolute_error":auto_error,"max_relative_error_by_h":relative,
   "finite_difference_checks":sum(p.size for p in params)*3},
 "omitted_batch_mean":{"detected":detected,"gradient_norm_ratio":float(np.linalg.norm(np.concatenate([g.ravel() for g in wrong]))/np.linalg.norm(np.concatenate([g.ravel() for g in analytic]))),
   "qualification":"Wrong only because forward loss is a mean; a consistently summed loss has a different valid gradient."},
 "relu_kink":{"x":0,"tf_selected_subgradient":kink_auto,"central_difference":kink_numeric,
   "qualification":"No unique derivative exists at zero. This is a checker-design limitation, not a TF bug."},
 "softmax":{"naive_nonfinite_count":int((~np.isfinite(naive)).sum()),"stable_probabilities":stable.tolist(),
   "stable_sum":float(stable.sum()),"argmax":int(stable.argmax()),"all_finite":True}}
(OUT/'math_controls.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
print(json.dumps(result,indent=2))
