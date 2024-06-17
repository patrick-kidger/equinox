import warnings

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import pytest


def _f_warn(x):
    x = eqx.warn_if(x, x < 0, "x must be non-negative")
    return jax.nn.relu(x)


_warn = pytest.warns(UserWarning, match="x must be non-negative")


def test_basic():
    jf = jax.jit(_f_warn)
    _f_warn(1.0)
    jf(1.0)
    with _warn:
        _f_warn(-1.0)
    with _warn:
        jf(-1.0)
    with _warn:
        jf(-1.0)


def test_vmap():
    vf = jax.vmap(_f_warn)
    jvf = jax.jit(vf)
    good = jnp.array([1.0, 1.0])
    bad1 = jnp.array([1.0, -1.0])
    bad2 = jnp.array([-1.0, -1.0])

    vf(good)
    jvf(good)
    with _warn:
        vf(bad1)
    with _warn:
        vf(bad2)
    with _warn:
        jvf(bad1)
    with _warn:
        jvf(bad2)


def test_jvp():
    def g(p, t):
        return jax.jvp(_f_warn, (p,), (t,))

    jg = jax.jit(g)

    for h in (g, jg):
        h(1.0, 1.0)
        h(1.0, -1.0)
        with _warn:
            h(-1.0, 1.0)
        with _warn:
            h(-1.0, -1.0)


def test_grad():
    g = jax.grad(_f_warn)
    jg = jax.jit(g)

    for h in (g, jg):
        h(1.0)
        with _warn:
            h(-1.0)


def test_grad2():
    @jax.jit
    @jax.grad
    def f(x, y, z):
        x = eqxi.nondifferentiable_backward(x)
        x, y = eqx.warn_if((x, y), z, "oops")
        return y

    with pytest.warns(UserWarning, match="oops"):
        f(1.0, 1.0, True)


def test_tracetime():
    @jax.jit
    def f(x):
        return eqx.warn_if(x, True, "hi")

    with pytest.warns(UserWarning):
        f(1.0)


def test_category():
    """Test warn_if with custom warning category."""

    def f_custom_warning(x):
        x = eqx.warn_if(x, x < 0, "custom warning", category=RuntimeWarning)
        return x

    with pytest.warns(RuntimeWarning, match="custom warning"):
        f_custom_warning(jnp.array(-1.0))


def test_category_jit():
    """Test warn_if with custom warning category under JIT."""

    @jax.jit
    def f_custom_warning(x):
        x = eqx.warn_if(x, x < 0, "custom warning", category=DeprecationWarning)
        return x

    with pytest.warns(DeprecationWarning, match="custom warning"):
        f_custom_warning(jnp.array(-1.0))


def test_multiple_messages():
    """Test warn_if with different messages based on condition."""

    def f(x):
        x = eqx.warn_if(x, x < 0, "x is negative")
        x = eqx.warn_if(x, x > 10, "x is too large")
        return x

    with pytest.warns(UserWarning, match="x is negative"):
        f(jnp.array(-1.0))

    with pytest.warns(UserWarning, match="x is too large"):
        f(jnp.array(11.0))


def test_unused_return():
    """Test warn_if when return value is unused."""

    @jax.jit
    def f(x):
        eqx.warn_if(x, x < 0, "unused warning")
        return x + 1

    with pytest.warns(UserWarning, match="unused warning"):
        result = f(jnp.array(-1.0))
    assert jnp.isclose(result, 0.0)


def test_used_return():
    """Test warn_if when return value is used."""

    @jax.jit
    def f(x):
        x = eqx.warn_if(x, x < 0, "used warning")
        return x + 1

    with pytest.warns(UserWarning, match="used warning"):
        result = f(jnp.array(-1.0))
    assert jnp.isclose(result, 0.0)


def test_pytree():
    """Test warn_if with PyTrees."""

    def f(pytree):
        x, y = pytree
        pytree = eqx.warn_if(pytree, (x**2 + y**2) > 1.0, "norm exceeded")
        return pytree

    # Should not warn
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        f((0.5, 0.5))
        assert len(record) == 0

    # Should warn
    with pytest.warns(UserWarning, match="norm exceeded"):
        f((0.8, 0.8))


def test_array_predicate():
    """Test warn_if with array predicates."""

    @jax.jit
    def f(x):
        x = eqx.warn_if(x, x < 0, "negative value")
        return jax.nn.relu(x)

    batched_f = jax.vmap(f)

    with pytest.warns(UserWarning, match="negative value"):
        batched_f(jnp.array([-1.0, 1.0, -2.0]))


def test_scalar_no_array():
    """Test warn_if with scalar conditions (no arrays)."""

    def f(x):
        x = eqx.warn_if(x, x < 0, "scalar negative")
        return x

    with pytest.warns(UserWarning, match="scalar negative"):
        f(-1)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        f(1)
        assert len(record) == 0
