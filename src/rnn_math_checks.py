"""Central differences versus BPTT autodiff on a tiny tanh RNN, float64."""
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from .runtime import write_json


def run():
    rng = np.random.default_rng(2026)
    x = tf.constant(rng.normal(size=(2, 4, 2)), dtype=tf.float64)
    target = tf.constant([0, 2], dtype=tf.int64)
    arrays = [rng.normal(0, .15, s) for s in [(2,3),(3,3),(3,),(3,3),(3,)]]
    variables = [tf.Variable(a, dtype=tf.float64) for a in arrays]
    def loss():
        w,u,b,v,c = variables
        h = tf.zeros((2,3), tf.float64)
        for t in range(4):
            h = tf.tanh(x[:,t] @ w + h @ u + b)
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=h @ v + c))
    with tf.GradientTape() as tape:
        value = loss()
    analytical = tape.gradient(value, variables)
    errors = []
    relative = []
    step = 1e-5
    for variable, original, gradient in zip(variables,arrays,analytical):
        for index in np.ndindex(original.shape):
            up,down=original.copy(),original.copy()
            up[index]+=step
            down[index]-=step
            variable.assign(up)
            plus=float(loss())
            variable.assign(down)
            minus=float(loss())
            variable.assign(original)
            numerical=(plus-minus)/(2*step)
            exact=float(gradient[index])
            errors.append(abs(numerical-exact))
            relative.append(abs(numerical-exact)/max(1e-6,abs(numerical)+abs(exact)))
    result=dict(passed=max(errors)<1e-7, checked_parameters=len(errors), dtype="float64", timesteps=4,
                hidden_units=3, classes=3, finite_difference_step=step, loss=float(value),
                max_absolute_error=max(errors), max_relative_error=max(relative),
                scope="tiny explicit tanh RNN equation; not a long-memory test")
    if not result['passed']:
        raise AssertionError(result)
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path('reports/rnn/math_verification.json'))
    args=p.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    result=run()
    write_json(args.out,result)
    print(result)
