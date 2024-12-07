import functools as ft
from typing import Any, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest

from .helpers import tree_allclose as _shaped_allclose


(cpu,) = jax.devices("cpu")
filter_pmap: Any = ft.partial(eqx.filter_pmap, devices=[cpu])  # pyright: ignore


def shaped_allclose(x, y, **kwargs):
    if isinstance(x, jax.Array):
        x = jax.device_put(x)
    return _shaped_allclose(x, y, **kwargs)


def _zero_if_inexact_array_else_none(x):
    return 0 if eqx.is_inexact_array(x) else None


def test_args():
    @filter_pmap(
        in_axes=(_zero_if_inexact_array_else_none, [{"a": None}], 0, eqx.if_array(0))
    )
    def f(a, b, c, d):
        return a + b[0]["a"] + c + d

    out = f(jnp.array([1]), [{"a": jnp.array([2])}], jnp.array([3]), 4)
    assert shaped_allclose(out, jnp.array([[10]]))


def test_default():
    @filter_pmap(in_axes=_zero_if_inexact_array_else_none)
    def f(a, b):
        with jax.numpy_dtype_promotion("standard"):
            return a + b

    assert shaped_allclose(f(jnp.array(3), jnp.array([3.0])), jnp.array([6.0]))

    with pytest.raises(ValueError):
        f(jnp.array(3.0), jnp.array([3.0]))


def test_out():
    def f(x):
        return x

    o1 = filter_pmap(f, in_axes=None, out_axes=None, axis_size=1)(jnp.array([3, 4]))
    o2 = filter_pmap(f, out_axes=0, axis_size=1)(1)
    o3 = filter_pmap(f, in_axes=None, out_axes=0, axis_size=1)(jnp.array([3, 4]))

    assert shaped_allclose(o1, jnp.array([3, 4]))
    assert shaped_allclose(o2, jnp.array([1]))
    assert shaped_allclose(o3, jnp.array([[3, 4]]))


def test_no_arrays():
    @filter_pmap(out_axes=None, axis_size=1)
    def f(x):
        return x

    assert shaped_allclose(f(1), 1)


def test_num_traces():
    num_traces = 0

    @filter_pmap(in_axes=None, axis_size=1)
    def f(x, y):
        nonlocal num_traces
        num_traces += 1
        return x + y

    assert shaped_allclose(f(jnp.array(1), 2), jnp.array([3]))
    assert num_traces == 2  # eval_shape + pmap
    assert shaped_allclose(f(jnp.array(3), 2), jnp.array([5]))
    assert num_traces == 2
    assert shaped_allclose(f(jnp.array(1), 3), jnp.array([4]))
    assert num_traces == 4
    assert shaped_allclose(f(jnp.array([3]), 3), jnp.array([[6]]))
    assert num_traces == 6
    assert shaped_allclose(f(jnp.array([4]), 3), jnp.array([[7]]))
    assert num_traces == 6


@pytest.mark.parametrize("call", [False, True])
@pytest.mark.parametrize("outer", [False, True])
def test_methods(call, outer):
    num_traces = 0

    class M(eqx.Module):
        increment: Union[int, jax.Array]

        if call:

            def __call__(self, x):
                nonlocal num_traces
                num_traces += 1
                return x + self.increment

            if not outer:
                __call__ = filter_pmap(__call__)
        else:

            def method(self, x):
                nonlocal num_traces
                num_traces += 1
                return x + self.increment

            if not outer:
                method = filter_pmap(method)

    y = jnp.array([1])

    def run(_m):
        if call:
            if outer:
                return filter_pmap(_m)(y)
            else:
                return _m(y)
        else:
            if outer:
                return filter_pmap(_m.method)(y)
            else:
                return _m.method(y)

    m = M(1)
    assert shaped_allclose(run(m), jnp.array([2]))
    assert num_traces == 2
    assert shaped_allclose(run(m), jnp.array([2]))
    assert num_traces == 2
    n = M(2)
    assert shaped_allclose(run(n), jnp.array([3]))
    assert num_traces == 4
    assert shaped_allclose(run(n), jnp.array([3]))
    assert num_traces == 4
    o = M(jnp.array([5]))
    p = M(jnp.array([6]))
    if outer:
        assert shaped_allclose(run(o), jnp.array([[6]]))
        assert num_traces == 6
        assert shaped_allclose(run(p), jnp.array([[7]]))
    else:
        assert shaped_allclose(run(o), jnp.array([6]))
        assert num_traces == 6
        assert shaped_allclose(run(p), jnp.array([7]))
    assert num_traces == 6


def test_pmap_grad():
    num_traces = 0

    def f(x):
        nonlocal num_traces
        num_traces += 1
        return x + 1

    grad = filter_pmap(eqx.filter_grad(f))(jnp.array([1.0]))
    assert shaped_allclose(grad, jnp.array([1.0]))
    assert num_traces == 2  # eval_shape + pmap

    grad = filter_pmap(eqx.filter_grad(f))(jnp.array([2.0]))
    assert shaped_allclose(grad, jnp.array([1.0]))
    assert num_traces == 2  # (eval_shape cached, pmap cached)

    value, grad = filter_pmap(eqx.filter_value_and_grad(f))(jnp.array([1.0]))
    assert shaped_allclose(value, jnp.array([2.0]))
    assert shaped_allclose(grad, jnp.array([1.0]))
    assert num_traces == 4  # eval_shape + pmap

    value, grad = filter_pmap(eqx.filter_value_and_grad(f))(jnp.array([2.0]))
    assert shaped_allclose(value, jnp.array([3.0]))
    assert shaped_allclose(grad, jnp.array([1.0]))
    assert num_traces == 4  # (eval_shape cached, pmap cached)


def test_pmap_vmap():
    num_traces = 0

    def f(x):
        nonlocal num_traces
        num_traces += 1
        return x + 1

    out = filter_pmap(eqx.filter_vmap(f))(jnp.array([[1, 2]]))
    assert shaped_allclose(out, jnp.array([[2, 3]]))
    assert num_traces == 2  # eval_shape, pmap

    out = filter_pmap(eqx.filter_vmap(f))(jnp.array([[2, 3]]))
    assert shaped_allclose(out, jnp.array([[3, 4]]))
    assert num_traces == 2  # both cached


def test_named_reduction():
    def f(x):
        y = x + 1
        return jax.lax.psum(y, axis_name="device")

    n = jax.local_device_count()
    output = filter_pmap(f, axis_name="device")(jnp.zeros(n))

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

    pytree_sharded = jtu.tree_map(maybe_replicate, pytree)

    def identity(x):
        """will return a pytree with non-jax fields, which could break filter_pmap"""
        return x

    filter_pmap(identity)(pytree_sharded)


def test_keyword_in_axes(getkey):
    x = jr.normal(getkey(), (1, 4))
    y = jr.normal(getkey(), (1, 1))
    out = filter_pmap(lambda x, y: x + y, in_axes=dict(y=1))(x, y)
    true_out = x + y.T
    assert shaped_allclose(out, true_out)


def test_keyword_default(getkey):
    x = jr.normal(getkey(), (1, 4))
    out = filter_pmap(lambda x, y=1: x + y, in_axes=dict(x=0))(x)
    true_out = x + 1
    assert shaped_allclose(out, true_out)

    with pytest.raises(ValueError):
        filter_pmap(lambda x, y=1: x, in_axes=dict(y=0))(x)


# Issue 325


@pytest.mark.parametrize("donate", ("all", "none"))
def test_aot_compilation(donate):
    def f(x, y):
        return 2 * x + y

    x, y = jnp.array([3]), 4
    lowered = filter_pmap(f, donate=donate).lower(x, y)
    lowered.as_text()
    compiled = lowered.compile()
    compiled(x, y)


# https://github.com/patrick-kidger/equinox/issues/900
# Unlike the vmap case we only test nonnegative integers, as pmap does not support
# negative indexing for `in_axes` or `out_axes`.
@pytest.mark.parametrize("out_axes", (0, 1, 2))
def test_out_axes_with_at_least_three_dimensions(out_axes):
    def foo(x):
        return x * 2

    x = jnp.arange(24).reshape((1, 2, 3, 4))
    y = jax.pmap(foo, out_axes=out_axes)(x)
    z = filter_pmap(foo, out_axes=out_axes)(x)
    assert y.shape == z.shape
    assert (y == z).all()
