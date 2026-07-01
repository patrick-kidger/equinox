import re

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import pytest


def _f(x):
    x = eqx.error_if(x, x < 0, "x must be non-negative")
    return jax.nn.relu(x)


# Strangely, JAX raises different errors depending on context.
_error = pytest.raises((ValueError, RuntimeError), match="x must be non-negative")


def test_basic():
    jf = jax.jit(_f)
    _f(1.0)
    jf(1.0)
    with _error as exc:
        jf(-1.0)
    assert "Batch index" not in str(exc.value)


def test_vmap():
    vf = jax.vmap(_f)
    jvf = jax.jit(vf)
    good = jnp.array([1.0, 1.0])
    bad1 = jnp.array([1.0, -1.0])
    bad2 = jnp.array([-1.0, -1.0])

    vf(good)
    jvf(good)
    with pytest.raises(
        (ValueError, RuntimeError),
        match=re.escape("Batch index (1,) had error:\nx must be non-negative"),
    ) as exc:
        vf(bad1)
    assert "Batch index (0,)" not in str(exc.value)
    with pytest.raises(
        (ValueError, RuntimeError),
        match=re.escape(
            "Batch index (0,) had error:\nx must be non-negative\n\n"
            "Batch index (1,) had error:\nx must be non-negative"
        ),
    ):
        vf(bad2)
    with pytest.raises(
        (ValueError, RuntimeError),
        match=re.escape("Batch index (1,) had error:\nx must be non-negative"),
    ) as exc:
        jvf(bad1)
    assert "Batch index (0,)" not in str(exc.value)
    with pytest.raises(
        (ValueError, RuntimeError),
        match=re.escape(
            "Batch index (0,) had error:\nx must be non-negative\n\n"
            "Batch index (1,) had error:\nx must be non-negative"
        ),
    ):
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

    with pytest.raises((ValueError, RuntimeError), match="hi"):
        f(1.0)


def test_nan():
    @jax.jit
    def f(x, pred):
        return eqx.error_if(x, pred, "hi", on_error="nan")

    y = f(1.0, True)
    assert jnp.isnan(y)


def test_off_tracetime():
    @jax.jit
    def f(x):
        return eqx.error_if(x, True, "hi", on_error="off") + 1

    assert jnp.isclose(f(1.0), 2.0)


def test_off():
    @jax.jit
    def f(x, pred):
        return eqx.error_if(x, pred, "hi", on_error="off")

    assert jnp.isclose(f(1.0, True), 1.0)


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
        assert "egads" in str(e)
        assert "EQX_ON_ERROR" not in str(e)


# https://github.com/patrick-kidger/equinox/issues/1232
def test_multi_device_eager():
    # Without the fix, the error cases below deadlock at an XLA collective and then
    # abort the whole process. `conftest.py` runs the suite with two CPU devices.
    mesh = jax.sharding.Mesh(jax.devices("cpu"), ("x",))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    x = jax.device_put(jnp.array(1.0), replicated)
    true_pred = jax.device_put(jnp.array(True), replicated)
    false_pred = jax.device_put(jnp.array(False), replicated)

    # (x sharded, pred sharded), (only x sharded), (only pred sharded).
    for x_i, pred_i in [(x, true_pred), (x, True), (jnp.array(1.0), true_pred)]:
        with pytest.raises(eqx.EquinoxRuntimeError, match="oops"):
            eqx.error_if(x_i, pred_i, "oops")

    # Non-replicated shardings as well.
    sharded = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x"))
    y = jax.device_put(jnp.arange(4.0), sharded)
    with pytest.raises(eqx.EquinoxRuntimeError, match="oops"):
        eqx.error_if(y, jnp.all(y >= 0), "oops")

    # No error: pass through unchanged, keeping the original sharding.
    out = eqx.error_if(x, false_pred, "oops")
    assert out.sharding == replicated
    assert jnp.array_equal(out, jnp.array(1.0))

    # Non-raising modes keep the original sharding too.
    out = eqx.error_if(x, true_pred, "oops", on_error="nan")
    assert out.sharding == replicated
    assert jnp.isnan(out).all()
    out = eqx.error_if(y, true_pred, "oops", on_error="nan")
    assert out.sharding == sharded
    assert jnp.isnan(out).all()
    out = eqx.error_if(x, true_pred, "oops", on_error="off")
    assert out.sharding == replicated
    assert jnp.array_equal(out, jnp.array(1.0))


def test_msg_callable():
    def _msg(x):
        return f"got bad value {x.item()}"

    @jax.jit
    def f(x):
        return eqx.error_if(x, x < 0, msg=_msg)

    f(jnp.array(1.0))
    with pytest.raises(
        (ValueError, RuntimeError), match=re.escape("got bad value -1.0")
    ):
        f(jnp.array(-1.0))


def test_msg_callable_vmap():
    def _msg(x):
        return f"got bad value {x.item()}"

    def f(x):
        return eqx.error_if(x, x < 0, msg=_msg)

    vf = jax.vmap(f)
    jvf = jax.jit(vf)
    good = jnp.array([1.0, 2.0])
    bad = jnp.array([1.0, -3.0])

    vf(good)
    jvf(good)
    with pytest.raises(
        (ValueError, RuntimeError),
        match=re.escape("Batch index (1,) had error:\ngot bad value -3.0"),
    ):
        vf(bad)
    with pytest.raises(
        (ValueError, RuntimeError),
        match=re.escape("Batch index (1,) had error:\ngot bad value -3.0"),
    ):
        jvf(bad)


def test_msg_callable_vmap_multiple_bad():
    def _msg(x):
        return f"got bad value {x.item()}"

    def f(x):
        return eqx.error_if(x, x < 0, msg=_msg)

    vf = jax.vmap(f)
    bad = jnp.array([-2.0, -3.0])

    with pytest.raises(
        (ValueError, RuntimeError),
        match=re.escape(
            "Batch index (0,) had error:\ngot bad value -2.0\n\n"
            "Batch index (1,) had error:\ngot bad value -3.0"
        ),
    ):
        vf(bad)


def test_msg_callable_non_arrays():
    def _msg(x):
        arr, label = x
        return f"{label}: got bad value {arr.item()}"

    @eqx.filter_jit
    def f(x):
        return eqx.error_if((x, "my_label"), x < 0, msg=_msg)

    val, label = f(jnp.array(1.0))
    assert jnp.isclose(val, 1.0)
    assert label == "my_label"
    with pytest.raises(Exception, match=re.escape("my_label: got bad value -1.0")):
        f(jnp.array(-1.0))


def test_branched_error_if():
    @eqx.filter_jit
    def f(x, pred, index):
        with pytest.warns(match="`equinox.branched_error_if` is deprecated"):
            return eqx.branched_error_if(
                x, pred, index, ["error zero", "error one", "error two"]
            )

    for idx in (0, jnp.array(0)):
        f(jnp.array(1.0), False, idx)

    for idx in (1, jnp.array(1)):
        with pytest.raises(Exception, match="error one"):
            f(jnp.array(1.0), True, idx)

    for idx in (2, jnp.array(2)):
        with pytest.raises(Exception, match="error two"):
            f(jnp.array(1.0), True, idx)


# https://github.com/patrick-kidger/equinox/issues/1156
def test_error_after_success():
    @eqx.filter_jit
    def foo(x):
        return eqx.error_if(x, x > 0.0, "foo")

    foo(jnp.array(-1.0))
    try:
        foo(jnp.array(1.0))
    except Exception as e:
        assert type(e) is eqx.EquinoxRuntimeError
