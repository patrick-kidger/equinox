from typing import Union

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from helpers import shaped_allclose

import equinox as eqx


def _zero_if_inexact_array_else_none(x):
    return 0 if eqx.is_inexact_array(x) else None


def _zero_if_array_else_none(x):
    return 0 if eqx.is_array(x) else None


def test_args():
    @eqx.filter_vmap(args=(_zero_if_inexact_array_else_none, [{"a": None}], 0))
    def f(a, b, c, d):
        return a + b[0]["a"] + c + d

    out = f(jnp.array([1]), [{"a": jnp.array([2])}], jnp.array([3]), 4)
    assert shaped_allclose(out, jnp.array([[10]]))


def test_kwargs():
    @eqx.filter_vmap(
        kwargs=dict(a=_zero_if_inexact_array_else_none, b=[{"a": None}], c=0)
    )
    def f(a, b, c, d):
        return a + b[0]["a"] + c + d

    out = f(jnp.array([1]), [{"a": jnp.array([2])}], jnp.array([3]), 4)
    assert shaped_allclose(out, jnp.array([[10]]))


def test_default():
    @eqx.filter_vmap(default=_zero_if_inexact_array_else_none)
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

    m = M(jnp.array([1, 2]))
    o1 = eqx.filter_vmap(m, fn=0)(1)
    o2 = eqx.filter_vmap(m, fn=0)(jnp.array([3, 4]))
    o3 = eqx.filter_vmap(m, default=None, fn=0)(jnp.array([3, 4]))
    assert shaped_allclose(o1, jnp.array([2, 3]))
    assert shaped_allclose(o2, jnp.array([4, 6]))
    assert shaped_allclose(o3, jnp.array([[4, 5], [5, 6]]))


def test_out():
    def f(x):
        return x

    o1 = eqx.filter_vmap(f, out=None, axis_size=5)(1)
    o2 = eqx.filter_vmap(f, default=None, out=None, axis_size=5)(jnp.array([3, 4]))
    o3 = eqx.filter_vmap(f, out=0, axis_size=5)(1)
    o4 = eqx.filter_vmap(f, default=None, out=0, axis_size=5)(jnp.array([3, 4]))

    assert shaped_allclose(o1, 1)
    assert shaped_allclose(o2, jnp.array([3, 4]))
    assert shaped_allclose(o3, jnp.array([1, 1, 1, 1, 1]))
    assert shaped_allclose(o4, jnp.array([[3, 4], [3, 4], [3, 4], [3, 4], [3, 4]]))


def test_no_arrays():
    @eqx.filter_vmap(out=_zero_if_inexact_array_else_none, axis_size=5)
    def f(x):
        return x

    assert shaped_allclose(f(1), 1)


def test_ensemble(getkey):
    def make(key):
        return eqx.nn.MLP(5, 4, 3, 2, key=getkey())

    keys = jr.split(getkey(), 7)
    models = eqx.filter_vmap(make, out=lambda x: 0 if eqx.is_array(x) else None)(keys)

    def call(model, x):
        return model(x)

    xs1 = jr.normal(getkey(), (7, 5))
    assert eqx.filter_vmap(call)(models, xs1).shape == (7, 4)
    assert eqx.filter_vmap(models, fn=_zero_if_array_else_none)(xs1).shape == (7, 4)

    xs2 = jr.normal(getkey(), (5,))
    assert eqx.filter_vmap(call, args=(_zero_if_array_else_none, None))(
        models, xs2
    ).shape == (7, 4)
    assert eqx.filter_vmap(models, default=None, fn=_zero_if_array_else_none,)(
        xs2
    ).shape == (7, 4)


@pytest.mark.parametrize("call", [False, True])
@pytest.mark.parametrize("outer", [False, True])
def test_methods(call, outer):
    class M(eqx.Module):
        increment: Union[int, jnp.ndarray]

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


def test_args_kwargs():
    @eqx.filter_vmap(kwargs=dict(x=0))
    def f(*args, **kwargs):
        return kwargs["x"]

    # check we can use other kwargs
    assert shaped_allclose(f(x=jnp.array([3]), y=4), jnp.array([3]))
    assert shaped_allclose(f(x=jnp.array([3]), y=jnp.array([4])), jnp.array([3]))

    with pytest.raises(ValueError):
        f(x=jnp.array([3]), y=jnp.array(4))

    with pytest.raises(ValueError):
        f(x=jnp.array(3))

    @eqx.filter_vmap(args=(_zero_if_array_else_none,))
    def h(*args, **kwargs):
        return args[0]

    # check we can use other args
    assert h(1, jnp.array([2])) == 1
    assert shaped_allclose(h(jnp.array([2]), 3), jnp.array([2]))


def test_named_reduction():
    def f(x):
        y = x + 1
        return jax.lax.psum(y, axis_name="device")

    n = 2
    output = eqx.filter_vmap(f, axis_name="device")(jnp.zeros(n))

    assert shaped_allclose(output, n * jnp.ones(n))


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

    _ = eqx.filter_vmap(
        identity,
        out=jax.tree_map(
            lambda value: 0 if eqx.is_array(value) else None,
            pytree,
        ),
    )(pytree)
