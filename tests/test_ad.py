import jax
import jax.numpy as jnp
import pytest

import equinox as eqx
import equinox.internal as eqxi

from .helpers import shaped_allclose


def test_nondiff():
    x = 1.0
    y = (jnp.array(1.0), object())
    assert shaped_allclose(eqxi.nondifferentiable(x), x)
    assert shaped_allclose(jax.jit(eqxi.nondifferentiable)(x), jnp.array(x))
    assert shaped_allclose(eqxi.nondifferentiable(y), y)
    assert shaped_allclose(eqx.filter_jit(eqxi.nondifferentiable)(y), y)

    with pytest.raises(RuntimeError):
        jax.jvp(eqxi.nondifferentiable, (x,), (x,))
    with pytest.raises(RuntimeError):
        jax.jvp(jax.jit(eqxi.nondifferentiable), (x,), (x,))


def test_nondiff_back():
    x = 1.0
    y = (jnp.array(1.0), object())
    assert shaped_allclose(eqxi.nondifferentiable_backward(x), x)
    assert shaped_allclose(jax.jit(eqxi.nondifferentiable_backward)(x), jnp.array(x))
    assert shaped_allclose(eqxi.nondifferentiable_backward(y), y)
    assert shaped_allclose(eqx.filter_jit(eqxi.nondifferentiable_backward)(y), y)

    x1, x2 = jax.jvp(eqxi.nondifferentiable_backward, (x,), (x,))
    x3, x4 = jax.jvp(jax.jit(eqxi.nondifferentiable_backward), (x,), (x,))
    assert shaped_allclose(x1, x)
    assert shaped_allclose(x2, x)
    assert shaped_allclose(x3, jnp.array(x))
    assert shaped_allclose(x4, jnp.array(x))

    with pytest.raises(RuntimeError):
        jax.grad(eqxi.nondifferentiable_backward)(x)
    with pytest.raises(RuntimeError):
        jax.jit(jax.grad(eqxi.nondifferentiable_backward))(x)
