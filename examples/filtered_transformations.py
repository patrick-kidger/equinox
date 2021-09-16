##############
#
# This example demonstrates how to use `equinox.filter_jit` and `equinox.filter_grad`.
#
# Here we'll use them to facilitate training a simple MLP: to automatically take gradients and jit with respect to
# all the jnp.arrays constituting the parameters. (But not with respect to anything else, like the choice of activation
# function -- as that isn't something we can differentiate/JIT anyway!)
#
#############

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

import equinox as eqx


# Toy data
def get_data(dataset_size, *, key):
    x = jrandom.normal(key, (dataset_size, 1))
    y = 5 * x - 2
    return x, y


# Simple dataloader
def dataloader(arrays, key, batch_size):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def main(
    dataset_size=10000,
    batch_size=256,
    learning_rate=3e-3,
    steps=1000,
    width_size=8,
    depth=1,
    seed=5678,
):
    data_key, loader_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 3)
    data = get_data(dataset_size, key=data_key)
    data = dataloader(data, batch_size=batch_size, key=loader_key)

    # We happen to be using an Equinox model here, but that *is not important*.
    # `equinox.filter_jit` and `equinox.filter_grad` will work just fine on any PyTree you like.
    # (Here, `model` is actually a PyTree -- have a look at the `build_model.py` example for more on that.)
    model = eqx.nn.MLP(
        in_size=1, out_size=1, width_size=width_size, depth=depth, key=model_key
    )

    # `filter_jit` and `filter_value_and_grad` are thin wrappers around the usual `jax` functions, that automatically
    # inspect the arguments of the function, JIT with respect to all JAX arrays, and differentiate with respect to all
    # floating point JAX arrays (i.e. the parameters of the model).
    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def loss(model, x, y):
        pred_y = jax.vmap(model)(x)
        return jnp.mean((y - pred_y) ** 2)

    optim = optax.sgd(learning_rate)
    opt_state = optim.init(model)
    for step, (x, y) in zip(range(steps), data):
        value, grads = loss(model, x, y)
        updates, opt_state = optim.update(grads, opt_state)
        # Essentially equivalent to optax.apply_updates, it just doesn't try to update anything with a gradient
        # of `None` (which is the gradient produced for anything we filtered out above).
        model = eqx.apply_updates(model, updates)
        print(step, value)
    return value  # Final loss


if __name__ == "__main__":
    main()
