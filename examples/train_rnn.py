##############
#
# This example has a similar structure to `train_mlp.py`, except that we now train
# an RNN.
#
# In particular this demonstrates the use of Modules alongside jax.lax.scan. (They just
# work, no tricks required.)
#
#############

import functools as ft
import math

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import optax

# Borrow dataloader from other example
from train_mlp import dataloader

import equinox as eqx
from equinox.module import Module


# Toy data of spirals.
# We aim to classify clockwise versus anticlockwise.
def get_data(dataset_size, *, key):
    t = jnp.linspace(0, 2 * math.pi, 16)
    offset = jrandom.uniform(key, (dataset_size, 1), minval=0, maxval=2 * math.pi)
    x1 = jnp.sin(t + offset) / (1 + t)
    x2 = jnp.cos(t + offset) / (1 + t)
    y = jnp.ones((dataset_size, 1))

    half_dataset_size = dataset_size // 2
    x1 = x1.at[:half_dataset_size].multiply(-1)
    y = y.at[:half_dataset_size].set(0)
    x = jnp.stack([x1, x2], axis=-1)

    return x, y


class RNN(eqx.Module):
    _hidden_size: int
    cell: eqx.Module
    linear: eqx.nn.Linear

    def __init__(self, in_size, out_size, hidden_size, *, key):
        ckey, lkey = jrandom.split(key)
        self._hidden_size = hidden_size
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(hidden_size, out_size, key=lkey)

    def __call__(self, input):
        hidden = jnp.zeros((self._hidden_size,))

        def f(carry, inp):
            return self.cell(inp, carry), None

        out, _ = lax.scan(f, hidden, input)
        return jax.nn.sigmoid(self.linear(out))


def main(
    dataset_size=10000,
    batch_size=32,
    learning_rate=3e-3,
    steps=500,
    hidden_size=16,
    depth=1,
    seed=5678,
):
    data_key, loader_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 3)
    data = get_data(dataset_size, key=data_key)
    data = dataloader(data, batch_size=batch_size, key=loader_key)

    model = RNN(in_size=2, out_size=1, hidden_size=hidden_size, key=model_key)

    def loss(model, x, y):
        pred_y = jax.vmap(model)(x)
        # Trains with respect to binary cross-entropy
        return -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y))

    vag = eqx.value_and_grad_f(loss, filter_tree=model.parameters())

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)

    @ft.partial(eqx.jitf, filter_fn=lambda x: isinstance(x, jnp.DeviceArray))
    def update_fn(model: Module, opt_state, x, y):
        value, grads = vag(model, x, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return value, (model.remove_unjitable_fields(), opt_state)

    for step, (x, y) in zip(range(steps), data):
        value, (model_updates, opt_state) = update_fn(model, opt_state, x, y)
        model = model.update(model_updates)
        print(step, value)
    return value


if __name__ == "__main__":
    main()
