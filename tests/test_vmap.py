from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from .helpers import tree_allclose


def _zero_if_inexact_array_else_none(x):
    return 0 if eqx.is_inexact_array(x) else None


def test_args():
    @eqx.filter_vmap(
        in_axes=(_zero_if_inexact_array_else_none, [{"a": None}], 0, eqx.if_array(0))
    )
    def f(a, b, c, d):
        return a + b[0]["a"] + c + d

    out = f(jnp.array([1]), [{"a": jnp.array([2])}], jnp.array([3]), 4)
    assert tree_allclose(out, jnp.array([[10]]))


def test_default():
    @eqx.filter_vmap(in_axes=_zero_if_inexact_array_else_none)
    def f(a, b):
        return a + b

    assert tree_allclose(f(jnp.array(3), jnp.array([3.0])), jnp.array([6.0]))

    with pytest.raises(ValueError):
        f(jnp.array(3.0), jnp.array([3.0]))


def test_out():
    def f(x):
        return x

    o1 = eqx.filter_vmap(f, out_axes=None, axis_size=5)(1)
    o2 = eqx.filter_vmap(f, in_axes=None, out_axes=None, axis_size=5)(jnp.array([3, 4]))
    o3 = eqx.filter_vmap(f, out_axes=0, axis_size=5)(1)
    o4 = eqx.filter_vmap(f, in_axes=None, out_axes=0, axis_size=5)(jnp.array([3, 4]))

    assert tree_allclose(o1, 1)
    assert tree_allclose(o2, jnp.array([3, 4]))
    assert tree_allclose(o3, jnp.array([1, 1, 1, 1, 1]))
    assert tree_allclose(o4, jnp.array([[3, 4], [3, 4], [3, 4], [3, 4], [3, 4]]))


def test_no_arrays():
    @eqx.filter_vmap(out_axes=_zero_if_inexact_array_else_none, axis_size=5)
    def f(x):
        return x

    assert tree_allclose(f(1), 1)


def test_ensemble(getkey):
    def make(key):
        return eqx.nn.MLP(5, 4, 3, 2, key=getkey())

    keys = jr.split(getkey(), 7)
    models = eqx.filter_vmap(make)(keys)

    def call(model, x):
        return model(x)

    xs1 = jr.normal(getkey(), (7, 5))
    assert eqx.filter_vmap(call)(models, xs1).shape == (7, 4)

    xs2 = jr.normal(getkey(), (5,))
    assert eqx.filter_vmap(call, in_axes=(eqx.if_array(0), None))(
        models, xs2
    ).shape == (7, 4)


@pytest.mark.parametrize("call", [False, True])
@pytest.mark.parametrize("outer", [False, True])
def test_methods(call, outer):
    class M(eqx.Module):
        increment: Union[int, jax.Array]

        if call:

            def __call__(self, x):
                return x + self.increment

            if not outer:
                __call__ = eqx.filter_vmap(__call__)
        else:

            def method(self, x):
                return x + self.increment

            if not outer:
                method = eqx.filter_vmap(method)

    m = M(5)
    y = jnp.array([1.0])

    if call:
        if outer:
            assert eqx.filter_vmap(m)(y) == 6
        else:
            assert m(y) == 6
    else:
        if outer:
            assert eqx.filter_vmap(m.method)(y) == 6
        else:
            assert m.method(y) == 6


def test_named_reduction():
    def f(x):
        y = x + 1
        return jax.lax.psum(y, axis_name="device")

    output = eqx.filter_vmap(f, axis_name="device")(jnp.zeros(2))
    assert tree_allclose(output, jnp.array([2.0, 2.0]))


def test_map_non_jax():
    # this contains a non-jax value for the `activation` field
    # and will therefore break filter_vmap if not filtered out
    # at input and output
    pytree = eqx.nn.MLP(
        2,
        2,
        2,
        2,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(42),
    )

    def identity(x):
        """will return a pytree with non-jax fields, which could break filter_vmap"""
        return x

    _ = eqx.filter_vmap(identity)(pytree)


def test_keyword_in_axes(getkey):
    x = jr.normal(getkey(), (3, 4))
    y = jr.normal(getkey(), (1, 3))
    out = eqx.filter_vmap(lambda x, y: x + y, in_axes=dict(y=1))(x, y)
    true_out = x + y.T
    assert tree_allclose(out, true_out)


def test_keyword_default(getkey):
    x = jr.normal(getkey(), (3, 4))
    out = eqx.filter_vmap(lambda x, y=1: x + y, in_axes=dict(x=0))(x)
    true_out = x + 1
    assert tree_allclose(out, true_out)

    with pytest.raises(ValueError):
        eqx.filter_vmap(lambda x, y=1: x, in_axes=dict(y=0))(x)


# https://github.com/patrick-kidger/equinox/issues/900
@pytest.mark.parametrize("out_axes", (0, 1, 2, -1, -2, -3))
def test_out_axes_with_at_least_three_dimensions(out_axes):
    def foo(x):
        return x * 2

    x = jnp.arange(24).reshape((2, 3, 4))
    y = jax.vmap(foo, out_axes=out_axes)(x)
    z = eqx.filter_vmap(foo, out_axes=out_axes)(x)
    assert y.shape == z.shape
    assert (y == z).all()
