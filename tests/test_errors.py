import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import pytest


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


def test_assert_dce():
    @jax.jit
    def f(x):
        x = x + 1
        eqxi.assert_dce(x, msg="oh no")
        return x

    f(1.0)

    @jax.jit
    def g(x):
        x = x + 1
        eqxi.assert_dce(x, msg="oh no")
        return x

    with jax.disable_jit():
        g(1.0)


def test_traceback_runtime_eqx(caplog):
    @eqx.filter_jit
    def f(x):
        return g(x)

    @eqx.filter_jit
    def g(x):
        return eqx.error_if(x, x > 0, "egads")

    try:
        f(jnp.array(1.0))
    except Exception as e:
        assert caplog.text == ""
        assert e.__cause__ is None
        msg = str(e).strip()
        assert msg.startswith("Above is the stack outside of JIT")
        assert "egads" in msg
        assert "EQX_ON_ERROR" in msg


def test_traceback_runtime_custom():
    class FooException(Exception):
        pass

    @eqx.filter_jit
    def f(x):
        return g(x)

    @eqx.filter_jit
    def g(x):
        def _raises():
            raise FooException("egads")

        return jax.pure_callback(_raises, x)  # pyright: ignore

    try:
        f(jnp.array(1.0))
    except Exception as e:
        # assert e.__cause__ is None  # varies by Python version and JAX version.
        assert "egads" in str(e)
        assert "EQX_ON_ERROR" not in str(e)
