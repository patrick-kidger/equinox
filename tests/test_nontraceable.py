import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

import equinox as eqx
import equinox.internal as eqxi


def test_nontraceable(getkey):
    mlp = eqx.nn.MLP(2, 2, 2, 2, key=getkey())
    dynamic, static = eqx.partition(mlp, eqx.is_array)
    dynamic_batch = jtu.tree_map(lambda x: x[None], dynamic)
    dynamic_flat = jtu.tree_leaves(dynamic)
    dynamic_batch_flat = jtu.tree_leaves(dynamic)

    def run(dynamic, static):
        x = eqx.combine(dynamic, static)
        # Test passing static values through `nontraceable`.
        x = eqxi.nontraceable(x)
        x = [jnp.sum(2 * x) for x in jtu.tree_leaves(x) if eqx.is_array(x)]
        return sum(x)

    run(dynamic, static)
    jax.jit(run, static_argnums=1)(dynamic, static)

    with pytest.raises(RuntimeError):
        jax.grad(run)(dynamic, static)
    with pytest.raises(RuntimeError):
        jax.jit(jax.grad(run), static_argnums=1)(dynamic, static)
    with pytest.raises(RuntimeError):
        jax.vmap(run, in_axes=(0, None))(dynamic_batch, static)

    jaxpr = jax.make_jaxpr(run, static_argnums=1)(dynamic, static)
    run2 = jax.core.jaxpr_as_fun(jaxpr)

    run2(*dynamic_flat)
    jax.jit(run2)(*dynamic_flat)

    with pytest.raises(RuntimeError):
        jax.grad(run2)(*dynamic_flat)
    with pytest.raises(RuntimeError):
        jax.jit(jax.grad(run2))(*dynamic_flat)
    with pytest.raises(RuntimeError):
        jax.vmap(run2, in_axes=0)(*dynamic_batch_flat)
