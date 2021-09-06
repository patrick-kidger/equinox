import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

import equinox as eqx


def test_filter_grad1(getkey):
    a = jrandom.normal(getkey(), (2, 3))

    @ft.partial(eqx.filter_grad, filter_spec=lambda _: True)
    def f(x):
        return jnp.sum(x)

    grad_f = f(a)
    assert jnp.all(grad_f == 1)


def test_filter_grad2(getkey):
    a = jrandom.normal(getkey(), (2, 3))
    b = jrandom.normal(getkey(), (2, 3))

    @ft.partial(eqx.filter_grad, filter_spec=eqx.is_inexact_array)
    def f(x):
        sum = 0.0
        for arg in jax.tree_leaves(x):
            if eqx.is_array_like(arg):
                sum = sum + jnp.sum(arg)
        return sum

    ga, gb = f([a, b])
    assert jnp.all(ga == 1)
    assert jnp.all(gb == 1)

    gtrue, ghi, gobject, ga = f([True, "hi", object(), a])
    assert gtrue is None
    assert ghi is None
    assert gobject is None
    assert jnp.all(ga == 1)

    gtrue, gdict, (g5, g1), gnp = f(
        [
            True,
            {"hi": eqx.nn.Linear(1, 1, key=getkey())},
            (5, 1.0),
            np.array([2.0, 3.0]),
        ]
    )
    assert gtrue is None
    assert list(gdict.keys()) == ["hi"]
    assert isinstance(gdict["hi"], eqx.nn.Linear)
    assert jnp.all(gdict["hi"].weight == 1)
    assert jnp.all(gdict["hi"].bias == 1)
    assert g5 is None
    assert g1 is None
    assert gnp is None


def test_filter_grad3(getkey):
    a = jrandom.normal(getkey(), (2, 3))
    b = jrandom.normal(getkey(), (1, 2))
    c = jrandom.normal(getkey(), ())

    @ft.partial(eqx.filter_grad, filter_spec=[True, False])
    def f(x):
        return jnp.sum(x[0]) + jnp.sum(x[1])

    ga, gb = f([a, b])
    assert jnp.all(ga == 1)
    assert gb is None

    @ft.partial(eqx.filter_grad, filter_spec={"a": True, "b": False})
    def h(x, y):
        return jnp.sum(x["a"]) * jnp.sum(x["b"]) * y

    grad = h({"a": a, "b": b}, c)
    assert jnp.allclose(grad["a"], jnp.sum(b) * c)
    assert grad["b"] is None

    with pytest.raises(ValueError):
        grad = h(c, {"a": a, "b": b})


# TODO: more comprehensive tests on this.
def test_filter_value_and_grad_(getkey):
    a = jrandom.normal(getkey(), (2, 3))

    @ft.partial(eqx.filter_value_and_grad, filter_spec=eqx.is_inexact_array)
    def f(x):
        return jnp.sum(x)

    val, grad = f(a)
    assert val == jnp.sum(a)
    assert jnp.all(grad == 1)


def test_aux(getkey):
    a = jrandom.normal(getkey(), (2, 3))

    @ft.partial(eqx.filter_grad, has_aux=True, filter_spec=eqx.is_inexact_array)
    def f(x):
        return jnp.sum(x), "hi"

    aux, grad = f(a)
    assert aux == "hi"
    assert jnp.all(grad == 1)

    @ft.partial(
        eqx.filter_value_and_grad, has_aux=True, filter_spec=eqx.is_inexact_array
    )
    def f(x):
        return jnp.sum(x), "hi"

    (value, aux), grad = f(a)
    assert value == jnp.sum(a)
    assert aux == "hi"
    assert jnp.all(grad == 1)
