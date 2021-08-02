import jax
import jax.numpy as jnp
import pytest

import equinox as eqx


def test_is_inexact_array(getkey):
    objs = [
        1,
        2.0,
        [2.0],
        True,
        object(),
        jnp.array([1]),
        jnp.array(1.0),
        eqx.nn.Linear(1, 1, key=getkey()),
    ]
    results = [False, False, False, False, False, False, True, False]
    for o, r in zip(objs, results):
        assert eqx.is_inexact_array(o) == r


def test_is_array_like(getkey):
    objs = [
        1,
        2.0,
        [2.0],
        True,
        object(),
        jnp.array([1]),
        jnp.array(1.0),
        eqx.nn.Linear(1, 1, key=getkey()),
    ]
    results = [True, True, True, True, False, True, True, False]
    for o, r in zip(objs, results):
        assert eqx.is_array_like(o) == r


def test_split_and_merge(getkey):
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
    ):  # has repeated elements
        int_args, notint_args, which, treedef = eqx.filters.split(pytree, filter_fn)
        for arg in int_args:
            assert isinstance(arg, int)
        for arg in notint_args:
            assert not isinstance(arg, int)
        assert sum(which) == 4
        re_pytree = eqx.filters.merge(int_args, notint_args, which, treedef)
        assert re_pytree == pytree


def test_splittree_and_merge(getkey):
    linear = eqx.nn.Linear(1, 1, key=getkey())
    linear_tree = jax.tree_map(lambda _: True, linear)
    filter_tree = [
        True,
        False,
        [False, False, {"a": True, "b": False, "c": linear_tree}],
    ]
    for i, pytree in enumerate(
        (
            [1, 2, [3, True, {"a": jnp.array(1), "b": 4, "c": linear}]],
            [1, 1, [1, 1, {"a": 1, "b": 1, "c": linear}]],
        )
    ):  # has repeated elements
        keep_args, notkeep_args, which, treedef = eqx.filters.split_tree(
            pytree, filter_tree
        )
        if i == 0:
            assert set(notkeep_args) == {2, 3, True, 4}
        else:
            assert notkeep_args == [1, 1, 1, 1]
        assert sum(which) == 4

        re_pytree = eqx.filters.merge(keep_args, notkeep_args, which, treedef)
        assert re_pytree == pytree

    filter_tree = [True, [False, False]]
    pytree = [True, None]
    with pytest.raises(ValueError):
        eqx.filters.split_tree(pytree, filter_tree)
