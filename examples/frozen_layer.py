import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

# Steal these utilities from the previous example
from train_mlp import dataloader, get_data

import equinox as eqx


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

    model = eqx.nn.MLP(
        in_size=1, out_size=1, width_size=depth, depth=depth, key=model_key
    )
    # Let's train just the final layer of the MLP, and leave the others frozen.
    filter_tree = jax.tree_map(lambda _: False, model)
    filter_tree = eqx.tree_at(
        lambda tree: (tree.layers[-1].weight.value, tree.layers[-1].bias.value),
        filter_tree,
        replace=(True, True),
    )

    @ft.partial(
        eqx.jitf, filter_fn=eqx.is_inexact_array
    )  # We can still JIT with respect to the other parameters
    @ft.partial(eqx.value_and_grad_f, filter_tree=filter_tree)
    def loss(model, x, y):
        pred_y = jax.vmap(model)(x)
        return jnp.mean((y - pred_y) ** 2)

    original_model = model  # Keep around to compare to later.
    optim = optax.sgd(learning_rate)
    opt_state = optim.init(model)
    for step, (x, y) in zip(range(steps), data):
        value, grads = loss(model, x, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        print(step, value)
    print(
        f"Weights of first layer at initialisation: {jax.tree_flatten(original_model.layers[0])}"
    )
    print(
        f"Weights of first layer at end of training: {jax.tree_flatten(model.layers[0])}"
    )
    print(
        f"Weights of last layer at initialisation: {jax.tree_flatten(original_model.layers[-1])}"
    )
    print(
        f"Weights of last layer at end of training: {jax.tree_flatten(model.layers[-1])}"
    )
    return original_model, model


if __name__ == "__main__":
    main()
