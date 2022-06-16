import functools as ft
from typing import Union

import jax
import jax.numpy as jnp
import pytest
from helpers import shaped_allclose as _shaped_allclose

import equinox as eqx


(cpu,) = jax.devices("cpu")
filter_pmap = ft.partial(eqx.filter_pmap, devices=[cpu])


def shaped_allclose(x, y, **kwargs):
    if isinstance(x, jnp.ndarray):
        x = jax.device_put(x)
    return _shaped_allclose(x, y, **kwargs)


def _zero_if_inexact_array_else_none(x):
    return 0 if eqx.is_inexact_array(x) else None


def test_args():
    @filter_pmap(args=(_zero_if_inexact_array_else_none, [{"a": None}], 0))
    def f(a, b, c, d):
        return a + b[0]["a"] + c + d

    out = f(jnp.array([1]), [{"a": jnp.array([2])}], jnp.array([3]), 4)
    assert shaped_allclose(out, jnp.array([[10]]))


def test_kwargs():
    @filter_pmap(kwargs=dict(a=_zero_if_inexact_array_else_none, b=[{"a": None}], c=0))
    def f(a, b, c, d):
        return a + b[0]["a"] + c + d

    out = f(jnp.array([1]), [{"a": jnp.array([2])}], jnp.array([3]), 4)
    assert shaped_allclose(out, jnp.array([[10]]))


def test_default():
    @filter_pmap(default=_zero_if_inexact_array_else_none)
    def f(a, b):
        return a + b

    assert shaped_allclose(f(jnp.array(3), jnp.array([3.0])), jnp.array([6.0]))

    with pytest.raises(ValueError):
        assert shaped_allclose(f(jnp.array(3.0), jnp.array([3.0])), jnp.array([6.0]))


def test_fn():
    class M(eqx.Module):
        increment: jnp.ndarray

        def __call__(self, x):
            return x + self.increment

    m = M(jnp.array([1]))
    o1 = filter_pmap(m, fn=0)(1)
    o2 = filter_pmap(m, fn=0)(jnp.array([3]))
    o3 = filter_pmap(m, default=None, fn=0)(jnp.array([3]))
    assert shaped_allclose(o1, jnp.array([2]))
    assert shaped_allclose(o2, jnp.array([4]))
    assert shaped_allclose(o3, jnp.array([[4]]))


def test_out():
    def f(x):
        return x

    o1 = filter_pmap(f, default=None, out=None, axis_size=1)(jnp.array([3, 4]))
    o2 = filter_pmap(f, out=0, axis_size=1)(1)
    o3 = filter_pmap(f, default=None, out=0, axis_size=1)(jnp.array([3, 4]))

    assert shaped_allclose(o1, jnp.array([3, 4]))
    assert shaped_allclose(o2, jnp.array([1]))
    assert shaped_allclose(o3, jnp.array([[3, 4]]))


def test_no_arrays():
    @filter_pmap(out=None, axis_size=1)
    def f(x):
        return x

    assert shaped_allclose(f(1), 1)


def test_bool():
    num_traces = 0

    @filter_pmap(args=(True, False), axis_size=1)
    def f(x, y):
        nonlocal num_traces
        num_traces += 1
        return x + y

    assert shaped_allclose(f(1, 2), jnp.array([3]))
    assert num_traces == 1
    assert shaped_allclose(f(3, 2), jnp.array([5]))
    assert num_traces == 1
    assert shaped_allclose(f(1, 3), jnp.array([4]))
    assert num_traces == 2
    assert shaped_allclose(f(3, 3), jnp.array([6]))
    assert num_traces == 2


@pytest.mark.parametrize("call", [False, True])
@pytest.mark.parametrize("outer", [False, True])
def test_methods(call, outer):
    num_traces = 0

    class M(eqx.Module):
        increment: Union[int, jnp.ndarray]

        if call:

            def __call__(self, x):
                nonlocal num_traces
                num_traces += 1
                return x + self.increment

            if not outer:
                __call__ = eqx.filter_pmap(__call__)
        else:

            def method(self, x):
                nonlocal num_traces
                num_traces += 1
                return x + self.increment

            if not outer:
                method = eqx.filter_pmap(method)

    y = jnp.array([1])

    def run(_m):
        if call:
            if outer:
                return eqx.filter_pmap(_m)(y)
            else:
                return _m(y)
        else:
            if outer:
                return eqx.filter_pmap(_m.method)(y)
            else:
                return _m.method(y)

    m = M(1)
    assert shaped_allclose(run(m), jnp.array([2]))
    assert shaped_allclose(run(m), jnp.array([2]))
    assert num_traces == 1
    n = M(2)
    assert shaped_allclose(run(n), jnp.array([3]))
    assert shaped_allclose(run(n), jnp.array([3]))
    assert num_traces == 2
    o = M(jnp.array([5]))
    p = M(jnp.array([6]))
    if outer:
        assert shaped_allclose(run(o), jnp.array([[6]]))
        assert shaped_allclose(run(p), jnp.array([[7]]))
    else:
        assert shaped_allclose(run(o), jnp.array([6]))
        assert shaped_allclose(run(p), jnp.array([7]))
    assert num_traces == 3


def test_pmap_grad():
    num_traces = 0

    def f(x):
        nonlocal num_traces
        num_traces += 1
        return x + 1

    grad = eqx.filter_pmap(eqx.filter_grad(f))(jnp.array([1.0]))
    assert shaped_allclose(grad, jnp.array([1.0]))
    grad = eqx.filter_pmap(eqx.filter_grad(f))(jnp.array([2.0]))
    assert shaped_allclose(grad, jnp.array([1.0]))
    assert num_traces == 1

    value, grad = eqx.filter_pmap(eqx.filter_value_and_grad(f))(jnp.array([1.0]))
    assert shaped_allclose(value, jnp.array([2.0]))
    assert shaped_allclose(grad, jnp.array([1.0]))
    value, grad = eqx.filter_pmap(eqx.filter_value_and_grad(f))(jnp.array([2.0]))
    assert shaped_allclose(value, jnp.array([3.0]))
    assert shaped_allclose(grad, jnp.array([1.0]))
    assert num_traces == 2


def test_pmap_vmap():
    num_traces = 0

    def f(x):
        nonlocal num_traces
        num_traces += 1
        return x + 1

    out = eqx.filter_pmap(eqx.filter_vmap(f))(jnp.array([[1, 2]]))
    assert shaped_allclose(out, jnp.array([[2, 3]]))
    out = eqx.filter_pmap(eqx.filter_vmap(f))(jnp.array([[2, 3]]))
    assert shaped_allclose(out, jnp.array([[3, 4]]))
    assert num_traces == 1


def test_args_kwargs():
    num_traces = 0

    @eqx.filter_pmap(kwargs=dict(x=True), axis_size=1)
    def f(*args, **kwargs):
        nonlocal num_traces
        num_traces += 1
        return kwargs["x"]

    assert f(x=2) == 2
    assert f(x=3) == 3
    assert num_traces == 1

    assert f(x=3, y=4) == 3  # check we can use other kwargs
    assert num_traces == 2

    @eqx.filter_pmap(default=eqx.is_array_like, axis_size=1)
    def g(*args, **kwargs):
        nonlocal num_traces
        num_traces += 1
        return kwargs["x"]

    assert g(x=1, y=1) == 1
    assert g(x=1, y=2) == 1
    assert num_traces == 3

    @eqx.filter_pmap(args=(eqx.is_array,), axis_size=1)
    def h(*args, **kwargs):
        nonlocal num_traces
        num_traces += 1
        return args[0]

    assert h(1, 2) == 1  # check we can use other args


def test_named_reduction():
    def f(x):
        y = x + 1
        return jax.lax.psum(y, axis_name="device")

    n = jax.local_device_count()
    output = eqx.filter_pmap(f, axis_name="device")(jnp.zeros(n))

    assert shaped_allclose(output, n * jnp.ones(n))


def test_map_non_jax():
    devices = jax.local_devices()

    # this contains a non-jax value for the `activation` field
    # and will therefore break filter_pmap if not filtered out
    # at input and output
    pytree = eqx.nn.MLP(
        2,
        2,
        2,
        2,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(42),
    )

    def maybe_replicate(value):
        if eqx.is_array(value):
            return jax.device_put_replicated(value, devices)
        else:
            return value

    pytree_sharded = jax.tree_map(maybe_replicate, pytree)

    def identity(x):
        """will return a pytree with non-jax fields, which could break filter_pmap"""
        return x

    _ = eqx.filter_pmap(
        identity,
        out=jax.tree_map(
            lambda value: 0 if eqx.is_array(value) else None,
            pytree_sharded,
        ),
    )(pytree_sharded)
