from typing import cast

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.core
import jax.extend.core
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

from .helpers import tree_allclose


def test_nondiff():
    x = 1.0
    y = (jnp.array(1.0), object())
    assert tree_allclose(eqxi.nondifferentiable(x), x)
    assert tree_allclose(jax.jit(eqxi.nondifferentiable)(x), jnp.array(x))
    assert tree_allclose(eqxi.nondifferentiable(y), y)
    assert tree_allclose(eqx.filter_jit(eqxi.nondifferentiable, donate="none")(y), y)

    with pytest.raises(RuntimeError):
        jax.jvp(eqxi.nondifferentiable, (x,), (x,))
    with pytest.raises(RuntimeError):
        jax.jvp(jax.jit(eqxi.nondifferentiable), (x,), (x,))


def test_nondiff_back():
    x = 1.0
    y = (jnp.array(1.0), object())
    assert tree_allclose(eqxi.nondifferentiable_backward(x), x)
    assert tree_allclose(jax.jit(eqxi.nondifferentiable_backward)(x), jnp.array(x))
    assert tree_allclose(eqxi.nondifferentiable_backward(y), y)
    assert tree_allclose(
        eqx.filter_jit(eqxi.nondifferentiable_backward, donate="none")(y), y
    )

    x1, x2 = jax.jvp(eqxi.nondifferentiable_backward, (x,), (x,))
    x3, x4 = jax.jvp(jax.jit(eqxi.nondifferentiable_backward), (x,), (x,))
    assert tree_allclose(x1, x)
    assert tree_allclose(x2, x)
    assert tree_allclose(x3, jnp.array(x))
    assert tree_allclose(x4, jnp.array(x))

    with pytest.raises(RuntimeError):
        jax.grad(eqxi.nondifferentiable_backward)(x)
    with pytest.raises(RuntimeError):
        jax.jit(jax.grad(eqxi.nondifferentiable_backward))(x)


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
    jaxpr = cast(jax.extend.core.ClosedJaxpr, jaxpr)
    run2 = jax.extend.core.jaxpr_as_fun(jaxpr)

    run2(*dynamic_flat)  # pyright: ignore
    jax.jit(run2)(*dynamic_flat)  # pyright: ignore

    with pytest.raises(RuntimeError):
        jax.grad(run2)(*dynamic_flat)  # pyright: ignore
    with pytest.raises(RuntimeError):
        jax.jit(jax.grad(run2))(*dynamic_flat)  # pyright: ignore
    with pytest.raises(RuntimeError):
        jax.vmap(run2, in_axes=0)(*dynamic_batch_flat)  # pyright: ignore
