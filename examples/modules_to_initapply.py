##############
#
# Existing JAX neural network libraries have sometimes followed the "init/apply"
# approach, in which the parameters of a network are initialised with `init`, and then
# the forward pass through a model is specified with `apply`. For example Stax follows
# this approach.
#
# As a corollary, the parameters returned from `init` are sometimes assumed to all be
# JIT-able or grad-able, e.g. by third-party libraries.
#
# In contrast Equinox is more general: it (a) doesn't assume that you necessarily
# want to take gradients with respect to all your parameters, and (b) doesn't even
# mandate that all your parameters are JAX arrays.
#
# If need be -- e.g. third party library compatibility -- then Equinox can be made to
# fit this style very easily, like so.
#
##############

import jax
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx


def make_mlp(in_size, out_size, width_size, depth, *, key):
    mlp = eqx.nn.MLP(in_size, out_size, width_size, depth, key=key)
    params, static = eqx.partition(mlp, eqx.is_inexact_array)

    def init_fn():
        return params

    def apply_fn(params, x):
        model = eqx.combine(params, static)
        return model(x)

    return init_fn, apply_fn


def main(in_size=2, seed=5678):
    key = jrandom.PRNGKey(seed)

    init_fn, apply_fn = make_mlp(
        in_size=in_size, out_size=1, width_size=8, depth=1, key=key
    )

    x = jnp.arange(in_size)  # sample data
    params = init_fn()
    y1 = apply_fn(params, x)
    params = jax.tree_map(lambda p: p + 1, params)  # "stochastic gradient descent"
    y2 = apply_fn(params, x)
    assert y1 != y2


if __name__ == "__main__":
    main()
