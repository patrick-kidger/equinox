import jax
import jax.numpy as jnp
import pytest

import equinox as eqx
import equinox.internal as eqxi


def _f(x):
    x = eqx.error_if(x, x < 0, "x must be non-negative")
    return jax.nn.relu(x)


# Strangely, JAX raises different errors depending on context.
_error = pytest.raises((ValueError, RuntimeError))


def test_basic():
    jf = jax.jit(_f)
    _f(1.0)
    jf(1.0)
    with _error:
        _f(-1.0)
    with _error:
        jf(-1.0)


def test_vmap():
    vf = jax.vmap(_f)
    jvf = jax.jit(vf)
    good = jnp.array([1.0, 1.0])
    bad1 = jnp.array([1.0, -1.0])
    bad2 = jnp.array([-1.0, -1.0])

    vf(good)
    jvf(good)
    with _error:
        vf(bad1)
    with _error:
        vf(bad2)
    with _error:
        jvf(bad1)
    with _error:
        jvf(bad2)


def test_jvp():
    def g(p, t):
        return jax.jvp(_f, (p,), (t,))

    jg = jax.jit(g)

    for h in (g, jg):
        h(1.0, 1.0)
        h(1.0, -1.0)
        with _error:
            h(-1.0, 1.0)
        with _error:
            h(-1.0, -1.0)


def test_grad():
    g = jax.grad(_f)
    jg = jax.jit(g)

    for h in (g, jg):
        h(1.0)
        with _error:
            h(-1.0)


def test_grad2():
    @jax.jit
    @jax.grad
    def f(x, y, z):
        x = eqxi.nondifferentiable_backward(x)
        x, y = eqx.error_if((x, y), z, "oops")
        return y

    f(1.0, 1.0, True)


def test_tracetime():
    @jax.jit
    def f(x):
        return eqx.error_if(x, True, "hi")

    with pytest.raises(Exception):
        with pytest.warns(UserWarning):
            f(1.0)


def test_nan_tracetime():
    @jax.jit
    def f(x):
        return eqx.error_if(x, True, "hi", on_error="nan")

    with pytest.warns(UserWarning):
        y = f(1.0)
    assert jnp.isnan(y)


def test_nan():
    @jax.jit
    def f(x, pred):
        return eqx.error_if(x, pred, "hi", on_error="nan")

    y = f(1.0, True)
    assert jnp.isnan(y)
