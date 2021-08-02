import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

import equinox as eqx


def _eq(a, b):
    return (type(a) is type(b)) and (a == b)


def test_jitf_filter_fn(getkey):
    a = jrandom.normal(getkey(), (2, 3))
    b = jrandom.normal(getkey(), (3,))
    c = jrandom.normal(getkey(), (1, 4))
    general_tree = [
        1,
        True,
        object(),
        {"a": a, "tuple": (2.0, b)},
        c,
        eqx.nn.MLP(2, 2, 2, 2, key=getkey()),
    ]
    array_tree = [{"a": a, "b": b}, (c,)]
    _mlp = jax.tree_map(lambda u: u if eqx.is_array_like(u) else None, general_tree[-1])

    @ft.partial(eqx.jitf, filter_fn=lambda _: True)
    def f(x):
        return x

    assert jnp.all(a == f(a))
    f1 = f(array_tree)
    assert jnp.all(f1[0]["a"] == a)
    assert jnp.all(f1[0]["b"] == b)
    assert jnp.all(f1[1][0] == c)

    with pytest.raises(TypeError):
        f(general_tree)

    @ft.partial(eqx.jitf, filter_fn=eqx.is_inexact_array)
    def g(x):
        return jax.tree_map(lambda u: u if eqx.is_array_like(u) else None, x)

    assert jnp.all(a == g(a))
    g1 = g(array_tree)
    assert jnp.all(g1[0]["a"] == a)
    assert jnp.all(g1[0]["b"] == b)
    assert jnp.all(g1[1][0] == c)
    g2 = g(general_tree)
    assert _eq(g2[0], jnp.array(1))
    assert _eq(g2[1], jnp.array(True))
    assert _eq(g2[2], None)
    assert jnp.all(g2[3]["a"] == a)
    assert _eq(g2[3]["tuple"][0], jnp.array(2.0))
    assert jnp.all(g2[3]["tuple"][1] == b)
    assert jnp.all(g2[4] == c)
    assert _eq(g2[5], _mlp)

    @ft.partial(eqx.jitf, filter_fn=eqx.is_array_like)
    def h(x):
        return jax.tree_map(lambda u: u if eqx.is_array_like(u) else None, x)

    assert jnp.all(a == h(a))
    h1 = h(array_tree)
    assert jnp.all(h1[0]["a"] == a)
    assert jnp.all(h1[0]["b"] == b)
    assert jnp.all(h1[1][0] == c)
    h2 = h(general_tree)
    assert _eq(h2[0], jnp.array(1))
    assert _eq(h2[1], jnp.array(True))
    assert _eq(h2[2], None)
    assert jnp.all(h2[3]["a"] == a)
    assert _eq(g2[3]["tuple"][0], jnp.array(2.0))
    assert jnp.all(g2[3]["tuple"][1] == b)
    assert jnp.all(g2[4] == c)
    assert _eq(g2[5], _mlp)

    @ft.partial(eqx.jitf, static_argnums=1, filter_fn=eqx.is_array_like)
    def i(x, y, z):
        return x * y * z

    assert i(1, 1, 1) == 1

    @ft.partial(eqx.jitf, static_argnums=(1, 2), filter_fn=eqx.is_array_like)
    def j(x, y, z):
        return x * y * z

    assert j(1, 1, 1) == 1


def test_jitf_filter_tree(getkey):
    a = jrandom.normal(getkey(), (2, 3))
    b = jrandom.normal(getkey(), (3,))
    c = jrandom.normal(getkey(), (1, 4))
    general_tree = [
        1,
        True,
        object(),
        {"a": a, "tuple": (2.0, b)},
        c,
        eqx.nn.MLP(2, 2, 2, 2, key=getkey()),
    ]
    _mlp = jax.tree_map(lambda u: u if eqx.is_array_like(u) else None, general_tree[-1])
    _filter_mlp = jax.tree_map(eqx.is_inexact_array, general_tree[-1])

    @ft.partial(
        eqx.jitf,
        filter_tree=[
            True,
            True,
            False,
            {"a": True, "tuple": (False, True)},
            True,
            _filter_mlp,
        ],
    )
    def f(x):
        return jax.tree_map(lambda u: u if eqx.is_array_like(u) else None, x)

    f1 = f(general_tree)
    assert _eq(f1[0], jnp.array(1))
    assert _eq(f1[1], jnp.array(True))
    assert _eq(f1[2], None)
    assert jnp.all(f1[3]["a"] == a)
    assert _eq(f1[3]["tuple"][0], jnp.array(2.0))
    assert jnp.all(f1[3]["tuple"][1] == b)
    assert jnp.all(f1[4] == c)
    assert _eq(f1[5], _mlp)

    @ft.partial(eqx.jitf, static_argnums=1, filter_tree=True)
    def g(x, y):
        return x * y

    assert g(1, 1) == 1

    @ft.partial(eqx.jitf, static_argnums=1, filter_tree=[True])
    def g2(x, y):
        return x * y

    _g = g2([1], 1)
    assert isinstance(_g, list)
    assert len(_g) == 1
    assert _g[0] == 1
    with pytest.raises(ValueError):
        g2(1, 1)  # filter tree doesn't match up

    @ft.partial(eqx.jitf, static_argnums=1, filter_tree=[True, True])
    def h(x, y, z):
        return x * y * z

    assert h(1, 1, 1) == 1
    with pytest.raises(ValueError):
        h([1], 1, 1)  # filter tree doesn't match up


def test_num_traces():
    num_traces = 0

    @ft.partial(eqx.jitf, filter_fn=lambda _: True)
    def f(x):
        nonlocal num_traces
        num_traces += 1

    f(jnp.zeros(2))
    f(jnp.zeros(2))
    assert num_traces == 1

    f(jnp.zeros(3))
    f(jnp.zeros(3))
    assert num_traces == 2

    f([jnp.zeros(2)])
    f([jnp.zeros(2), jnp.zeros(3)])
    f([jnp.zeros(2), True])
    assert num_traces == 5

    num_traces = 0

    @ft.partial(eqx.jitf, static_argnums=1, filter_fn=eqx.is_array_like)
    def g(x, y):
        nonlocal num_traces
        num_traces += 1

    g(jnp.zeros(2), True)
    g(jnp.zeros(2), False)
    assert num_traces == 2

    num_traces = 0

    @ft.partial(
        eqx.jitf, static_argnums=(0, 2), filter_tree=[{"a": True, "b": False}, False]
    )
    def h(x, y, z, w):
        nonlocal num_traces
        num_traces += 1

    h(True, {"a": 1, "b": 1}, True, True)
    h(False, {"a": 1, "b": 1}, True, True)
    h(True, {"a": 1, "b": 0}, True, True)
    h(True, {"a": 1, "b": 1}, True, 2)
    h(True, {"a": 1, "b": 1}, 5, True)
    assert num_traces == 5
    h(True, {"a": 2, "b": 1}, True, True)
    assert num_traces == 5
