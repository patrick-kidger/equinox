import functools as ft
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax


def get_data(dataset_size, *, key):
    x = jrandom.normal(key, (dataset_size, 1))
    y = 5 * x - 2
    return x, y


def dataloader(arrays, key, batch_size):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        key, = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def main(dataset_size=10000, batch_size=256, learning_rate=3e-3, steps=1000, width_size=8, depth=1, seed=5678):
    data_key, loader_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 3)
    data = get_data(dataset_size, key=data_key)
    data = dataloader(data, batch_size=batch_size, key=loader_key)

    model = eqx.nn.MLP(in_size=1, out_size=1, width_size=depth, depth=depth, key=model_key)

    @ft.partial(eqx.jitf, filter_fn=eqx.is_inexact_array)
    @ft.partial(eqx.value_and_grad_f, filter_fn=eqx.is_inexact_array)
    def loss(model, x, y):
        pred_y = jax.vmap(model)(x)
        return jnp.mean((y - pred_y)**2)

    optim = optax.sgd(learning_rate)
    opt_state = optim.init(model)
    for step, (x, y) in zip(range(steps), data):
        value, grads = loss(model, x, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        print(step, value)


if __name__ == '__main__':
    main()
