from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest


def test_is_array(getkey):
    objs = [
        1,
        2.0,
        [2.0],
        True,
        object(),
        jnp.array([1]),
        jnp.array(1.0),
        np.array(1.0),
        np.array(1),
        eqx.nn.Linear(1, 1, key=getkey()),
    ]
    results = [False, False, False, False, False, True, True, True, True, False]
    for o, r in zip(objs, results):
        assert eqx.is_array(o) == r


def test_is_array_like(getkey):
    objs = [
        1,
        2.0,
        [2.0],
        True,
        object(),
        jnp.array([1]),
        jnp.array(1.0),
        np.array(1.0),
        np.array(1),
        eqx.nn.Linear(1, 1, key=getkey()),
    ]
    results = [True, True, False, True, False, True, True, True, True, False]
    for o, r in zip(objs, results):
        assert eqx.is_array_like(o) == r


def test_is_inexact_array(getkey):
    objs = [
        1,
        2.0,
        [2.0],
        True,
        object(),
        jnp.array([1]),
        jnp.array(1.0),
        np.array(1.0),
        np.array(1),
        eqx.nn.Linear(1, 1, key=getkey()),
    ]
    results = [False, False, False, False, False, False, True, True, False, False]
    for o, r in zip(objs, results):
        assert eqx.is_inexact_array(o) == r


def test_is_inexact_array_like(getkey):
    objs = [
        1,
        2.0,
        [2.0],
        True,
        object(),
        jnp.array([1]),
        jnp.array(1.0),
        np.array(1.0),
        np.array(1),
        eqx.nn.Linear(1, 1, key=getkey()),
    ]
    results = [False, True, False, False, False, False, True, True, False, False]
    for o, r in zip(objs, results):
        assert eqx.is_inexact_array_like(o) == r


def test_filter(getkey):
    filter_fn = lambda x: isinstance(x, int)
    for pytree in (
        [
            1,
            2,
            [
                3,
                "hi",
                {"a": jnp.array(1), "b": 4, "c": eqx.nn.MLP(2, 2, 2, 2, key=getkey())},
            ],
        ],
        [1, 1, 1, 1, "hi"],
    ):
        filtered = eqx.filter(pytree, filter_spec=filter_fn)
        for arg in jtu.tree_leaves(filtered):
            assert isinstance(arg, int)
        num_int_leaves = sum(
            1 for leaf in jtu.tree_leaves(filtered) if isinstance(leaf, int)
        )
        assert len(jtu.tree_leaves(filtered)) == num_int_leaves

    filter_spec = [False, True, [filter_fn, True]]
    sentinel = object()
    pytree = [
        eqx.nn.Linear(1, 1, key=getkey()),
        eqx.nn.Linear(1, 1, key=getkey()),
        [eqx.nn.Linear(1, 1, key=getkey()), sentinel],
    ]
    filtered = eqx.filter(pytree, filter_spec=filter_spec)
    none_linear = jtu.tree_map(lambda _: None, eqx.nn.Linear(1, 1, key=getkey()))
    assert filtered[0] == none_linear
    assert filtered[1] == pytree[1]
    assert filtered[2][0] == none_linear
    assert filtered[2][1] is sentinel

    with pytest.raises(ValueError):
        eqx.filter(pytree, filter_spec=filter_spec[1:])


def test_partition_and_combine(getkey):
    filter_fn = lambda x: isinstance(x, int)
    for pytree in (
        [
            1,
            2,
            [
                3,
                "hi",
                {"a": jnp.array(1), "b": 4, "c": eqx.nn.MLP(2, 2, 2, 2, key=getkey())},
            ],
        ],
        [1, 1, 1, 1, "hi"],
    ):
        filtered, unfiltered = eqx.partition(pytree, filter_spec=filter_fn)
        for arg in jtu.tree_leaves(filtered):
            assert isinstance(arg, int)
        for arg in jtu.tree_leaves(unfiltered):
            assert not isinstance(arg, int)
        assert eqx.combine(filtered, unfiltered) == pytree
        assert eqx.combine(unfiltered, filtered) == pytree


def test_partition_subtree():
    a, b = eqx.partition([(1,), 2], [True, False])
    eqx.combine(a, b)


def test_is_leaf():
    class M(eqx.Module):
        value: Any

    def is_m(x):
        return isinstance(x, M)

    def filter_spec(x):
        if is_m(x):
            return x.value == 1
        return True

    pytree = [M(1), M(2), 3]
    out = eqx.filter(pytree, filter_spec, is_leaf=is_m)
    assert out == [M(1), None, 3]
    out1, out2 = eqx.partition(pytree, filter_spec, is_leaf=is_m)
    assert out1 == [M(1), None, 3]
    assert out2 == [None, M(2), None]
