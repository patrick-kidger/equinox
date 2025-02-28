import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import pytest
import warnings


def _f(x):
    x = eqx.warn_if(x, x < 0, "x must be non-negative")
    return jax.nn.relu(x)


# Strangely, JAX raises different errors depending on context.
_warn = pytest.warns(UserWarning)


def test_basic():
    jf = jax.jit(_f)
    _f(1.0)
    jf(1.0)
    with _warn:
        _f(-1.0)
    with _warn:
        jf(-1.0)
    with _warn:
        jf(-1.0)


def test_vmap():
    vf = jax.vmap(_f)
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
        return jax.jvp(_f, (p,), (t,))

    jg = jax.jit(g)

    for h in (g, jg):
        h(1.0, 1.0)
        h(1.0, -1.0)
        with _warn:
            h(-1.0, 1.0)
        with _warn:
            h(-1.0, -1.0)


def test_grad():
    g = jax.grad(_f)
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

    f(1.0, 1.0, True)


def test_tracetime():
    @jax.jit
    def f(x):
        return eqx.warn_if(x, True, "hi")

    with pytest.warns(UserWarning):
        f(1.0)


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


# def test_traceback_runtime_eqx():
#     @eqx.filter_jit
#     def f(x):
#         return g(x)

#     @eqx.filter_jit
#     def g(x):
#         return eqx.warn_if(x, x > 0, "egads")

#     f(jnp.array(1.0))
#     except Exception as e:
#         assert e.__cause__ is None
#         msg = str(e).strip()
#         assert msg.startswith("egads")
#         assert "EQX_ON_warn" in msg
#         assert msg.endswith("information.")
