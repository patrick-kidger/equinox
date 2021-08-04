import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

import equinox as eqx


def test_gradf_filter_fn(getkey):
    a = jrandom.normal(getkey(), (2, 3))
    b = jrandom.normal(getkey(), (2, 3))

    @ft.partial(eqx.gradf, filter_fn=lambda _: True)
    def f(x):
        return jnp.sum(x)

    grad_f = f(a)
    assert jnp.all(grad_f == 1)

    @ft.partial(eqx.gradf, argnums=(0, 1), filter_fn=lambda _: True)
    def g1(x, y):
        return jnp.sum(x)

    grad_g1a, grad_g1b = g1(a, b)
    assert jnp.all(grad_g1a == 1)
    assert jnp.all(grad_g1b == 0)

    @ft.partial(eqx.gradf, argnums=(0, 1), filter_fn=lambda _: False)
    def g2(x, y):
        return jnp.sum(x)

    grad_g2a, grad_g2b = g2(a, b)
    assert grad_g2a is None
    assert grad_g2b is None

    @ft.partial(eqx.gradf, argnums=1, filter_fn=lambda _: True)
    def h(x, y):
        return jnp.sum(x + y)

    grad_h1b = h(a, b)
    assert jnp.all(grad_h1b == 1)

    @ft.partial(eqx.gradf, argnums=(0, 1), filter_fn=lambda _: True)
    def i(x, y):
        return jnp.sum(x(y))

    with pytest.raises(Exception):
        i(a, b)  # there's no way to take a gradient wrt a

    with pytest.raises(Exception):
        i(lambda v: v, b)  # there's no way to take a gradient wrt the lambda

    @ft.partial(eqx.gradf, filter_fn=eqx.is_inexact_array)
    def j(x):
        sum = 0.0
        for arg in jax.tree_leaves(x):
            if eqx.is_array_like(arg):
                sum = sum + jnp.sum(arg)
        return sum

    ga, gb = j([a, b])
    assert jnp.all(ga == 1)
    assert jnp.all(gb == 1)

    gtrue, ghi, gobject, ga = j([True, "hi", object(), a])
    assert gtrue is None
    assert ghi is None
    assert gobject is None
    assert jnp.all(ga == 1)

    gtrue, gdict, (g5, g1), gnp = j(
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

    @ft.partial(eqx.gradf, filter_fn=eqx.is_array_like)
    def k(x):
        sum = 0.0
        for arg in jax.tree_leaves(x):
            if eqx.is_array_like(arg):
                sum = sum + jnp.sum(arg)
        return sum

    gx, gy = k([a, b])
    assert jnp.all(ga == 1)
    assert jnp.all(gb == 1)

    ghi, gobject, ga = k(["hi", object(), a])
    assert ghi is None
    assert gobject is None
    assert jnp.all(ga == 1)

    gdict, (g1,), gnp = k(
        [{"hi": eqx.nn.Linear(1, 1, key=getkey())}, (1.0,), np.array([2.0, 3.0])]
    )
    assert list(gdict.keys()) == ["hi"]
    assert isinstance(gdict["hi"], eqx.nn.Linear)
    assert jnp.all(gdict["hi"].weight == 1)
    assert jnp.all(gdict["hi"].bias == 1)
    assert g1 == 1
    assert gnp.shape == (2,)
    assert np.all(gnp == 1)


def test_gradf_filter_tree(getkey):
    a = jrandom.normal(getkey(), (2, 3))
    b = jrandom.normal(getkey(), (1, 2))
    c = jrandom.normal(getkey(), ())

    @ft.partial(eqx.gradf, filter_tree=[True, False])
    def f(x):
        return jnp.sum(x[0]) + jnp.sum(x[1])

    ga, gb = f([a, b])
    assert jnp.all(ga == 1)
    assert gb is None

    @ft.partial(eqx.gradf, argnums=(0, 1), filter_tree=[True, False])
    def g(x, y):
        return jnp.sum(x) + jnp.sum(y)

    ga, gb = g(a, b)
    assert jnp.all(ga == 1)
    assert gb is None

    @ft.partial(eqx.gradf, argnums=0, filter_tree={"a": True, "b": False})
    def h1(x, y):
        return jnp.sum(x["a"]) * jnp.sum(x["b"]) * y

    @ft.partial(eqx.gradf, argnums=1, filter_tree={"a": True, "b": False})
    def h2(x, y):
        return jnp.sum(y["a"]) * jnp.sum(y["b"]) * x

    grad = h1({"a": a, "b": b}, c)
    assert jnp.allclose(grad["a"], jnp.sum(b) * c)
    assert grad["b"] is None

    grad = h2(c, {"a": a, "b": b})
    assert jnp.allclose(grad["a"], jnp.sum(b) * c)
    assert grad["b"] is None

    with pytest.raises(ValueError):
        grad = h1(c, {"a": a, "b": b})
    with pytest.raises(ValueError):
        grad = h2({"a": a, "b": b}, c)

    @ft.partial(eqx.gradf, argnums=(2, 0), filter_tree=(True,))
    def i(x, y, z):
        return jnp.sum(x) * jnp.sum(y) * jnp.sum(z)

    with pytest.raises(IndexError):
        i(a, b, c)

    @ft.partial(eqx.gradf, argnums=(2, 0), filter_tree=(True, {"a": True, "b": False}))
    def j(x, y, z):
        return jnp.sum(x["a"]) * jnp.sum(x["b"]) * jnp.sum(y) * jnp.sum(z)

    gradc, graddict = j({"a": a, "b": b}, 2.0, c)
    assert jnp.allclose(gradc, jnp.sum(a) * jnp.sum(b) * 2)
    assert jnp.allclose(graddict["a"], jnp.sum(b) * jnp.sum(c) * 2)
    assert graddict["b"] is None


def test_both_filter():
    with pytest.raises(ValueError):

        @ft.partial(eqx.gradf, filter_tree=True, filter_fn=lambda _: True)
        def f(x):
            return x


def test_no_filter():
    with pytest.raises(ValueError):

        @eqx.gradf
        def f(x):
            return x


# TODO: more comprehensive tests on this.
def test_value_and_grad_f(getkey):
    a = jrandom.normal(getkey(), (2, 3))

    @ft.partial(eqx.value_and_grad_f, filter_fn=eqx.is_inexact_array)
    def f(x):
        return jnp.sum(x)

    val, grad = f(a)
    assert val == jnp.sum(a)
    assert jnp.all(grad == 1)


def test_aux(getkey):
    a = jrandom.normal(getkey(), (2, 3))

    @ft.partial(eqx.gradf, has_aux=True, filter_fn=eqx.is_inexact_array)
    def f(x):
        return jnp.sum(x), "hi"

    aux, grad = f(a)
    assert aux == "hi"
    assert jnp.all(grad == 1)

    @ft.partial(eqx.value_and_grad_f, has_aux=True, filter_fn=eqx.is_inexact_array)
    def f(x):
        return jnp.sum(x), "hi"

    (value, aux), grad = f(a)
    assert value == jnp.sum(a)
    assert aux == "hi"
    assert jnp.all(grad == 1)
