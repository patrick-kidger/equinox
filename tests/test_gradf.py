import equinox as eqx
import functools as ft
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest


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
    assert jnp.all(grad_g2a == 0)
    assert jnp.all(grad_g2b == 0)

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
    assert gtrue == 0
    assert ghi == 0
    assert gobject == 0
    assert jnp.all(ga == 1)

    gtrue, gdict, (g5, g1), gnp = j([True, {"hi": eqx.nn.Linear(1, 1, key=getkey())}, (5, 1.), np.array([2., 3.])])
    assert gtrue == 0
    assert list(gdict.keys()) == ["hi"]
    assert isinstance(gdict["hi"], eqx.nn.Linear)
    assert jnp.all(gdict["hi"].weight == 1)
    assert jnp.all(gdict["hi"].bias == 1)
    assert g5 == 0
    assert g1 == 0
    assert gnp == 0

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
    assert ghi == 0
    assert gobject == 0
    assert jnp.all(ga == 1)

    gdict, (g1,), gnp = k([{"hi": eqx.nn.Linear(1, 1, key=getkey())}, (1.,), np.array([2., 3.])])
    assert list(gdict.keys()) == ["hi"]
    assert isinstance(gdict["hi"], eqx.nn.Linear)
    assert jnp.all(gdict["hi"].weight == 1)
    assert jnp.all(gdict["hi"].bias == 1)
    assert g1 == 1
    assert gnp.shape == (2,)
    assert np.all(gnp == 1)


def test_gradf_filter_tree(getkey):
    jrandom.normal(getkey(), (2, 3))
