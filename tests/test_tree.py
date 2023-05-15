from typing import Callable

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import pytest

import equinox as eqx


def test_tree_at_replace(getkey):
    key = getkey()
    key1, key2 = jrandom.split(key, 2)
    pytree = [1, 2, {"a": jnp.array([1.0, 2.0])}, eqx.nn.Linear(1, 2, key=key1)]
    true_pytree1 = [1, 2, {"a": "hi"}, eqx.nn.Linear(1, 2, key=key1)]
    true_pytree2 = [1, 2, {"a": jnp.array([1.0, 2.0])}, eqx.nn.Linear(1, 2, key=key2)]
    where1 = lambda tree: tree[2]["a"]
    where2 = lambda tree: (tree[3].weight, tree[3].bias)
    weight2 = true_pytree2[3].weight
    bias2 = true_pytree2[3].bias
    pytree1 = eqx.tree_at(where1, pytree, replace="hi")
    pytree2 = eqx.tree_at(where2, pytree, replace=(weight2, bias2))

    assert pytree1[:-2] == true_pytree1[:-2]
    assert pytree2[:-2] == true_pytree2[:-2]
    assert jnp.all(pytree1[-2]["a"] == true_pytree1[-2]["a"])
    assert jnp.all(pytree2[-2]["a"] == true_pytree2[-2]["a"])
    assert jnp.all(pytree1[-1].weight == true_pytree1[-1].weight)
    assert jnp.all(pytree1[-1].bias == true_pytree1[-1].bias)
    assert jnp.all(pytree2[-1].weight == true_pytree2[-1].weight)
    assert jnp.all(pytree2[-1].bias == true_pytree2[-1].bias)

    true_pytree3 = ["hi", 2, {"a": 4}, eqx.nn.Linear(1, 2, key=key1)]
    where3 = lambda tree: (tree[0], tree[2]["a"])
    pytree3 = eqx.tree_at(where3, pytree, replace=("hi", 4))
    assert pytree3[:-1] == true_pytree3[:-1]
    assert jnp.all(pytree3[-1].weight == true_pytree3[-1].weight)
    assert jnp.all(pytree3[-1].bias == true_pytree3[-1].bias)

    with pytest.raises(TypeError):
        eqx.tree_at(where3, pytree, replace=4)
    with pytest.raises(ValueError):
        eqx.tree_at(where3, pytree, replace=(3, 4, 5))


def test_tree_at_replace_fn(getkey):
    key = getkey()
    pytree = [1, 2, 3, {"a": jnp.array([1.0, 2.0])}, eqx.nn.Linear(1, 2, key=key)]

    def replace_fn(x):
        if isinstance(x, int):
            return "found an int"
        else:
            return x

    true_pytree1 = [
        "found an int",
        "found an int",
        3,
        {"a": jnp.array([1.0, 2.0])},
        eqx.nn.Linear(1, 2, key=key),
    ]
    where = lambda tree: (tree[0], tree[1])
    pytree1 = eqx.tree_at(where, pytree, replace_fn=replace_fn)

    assert pytree1[:3] == true_pytree1[:3]
    assert jnp.all(pytree1[3]["a"] == true_pytree1[3]["a"])
    assert jnp.all(pytree1[-1].weight == true_pytree1[-1].weight)
    assert jnp.all(pytree1[-1].bias == true_pytree1[-1].bias)

    with pytest.raises(ValueError):
        eqx.tree_at(where, pytree, replace=(0, 1), replace_fn=replace_fn)


def test_tree_at_subtree(getkey):
    class L(eqx.Module):
        def __call__(self, x):
            return x

    mlp = eqx.nn.MLP(2, 2, 2, 2, key=getkey())

    # m.layers is a node in the PyTree
    newmlp1 = eqx.tree_at(
        lambda m: m.layers, mlp, [L() for _ in range(len(mlp.layers))]
    )

    # tuple(m.layers) is a sequence of nodes in the PyTree.
    newmlp2 = eqx.tree_at(
        lambda m: tuple(m.layers), mlp, [L() for _ in range(len(mlp.layers))]
    )

    x = jrandom.normal(getkey(), (2,))
    assert (jnn.relu(x) == newmlp1(x)).all()
    assert (jnn.relu(x) == newmlp2(x)).all()


def test_tree_at_dependent_where(getkey):
    mlp = eqx.nn.MLP(2, 2, 2, 2, key=getkey())

    def where(m):
        return jtu.tree_leaves(eqx.filter(m, eqx.is_array))

    with pytest.raises(ValueError):
        eqx.tree_at(where, mlp, where(mlp))


def test_tree_at_none_leaf():
    with pytest.raises(ValueError):
        eqx.tree_at(lambda y: y[0], (None, None, 0), True)
    x = eqx.tree_at(lambda y: y[0], (None, None, 0), True, is_leaf=lambda y: y is None)
    assert x == (True, None, 0)


def test_tree_equal():
    key1 = jrandom.PRNGKey(0)
    key2 = jrandom.PRNGKey(1)
    # Not using getkey as ever-so-in-principle two random keys could produce the same
    # weights (like that's ever going to happen).
    pytree1 = [1, 2, 3, {"a": jnp.array([1.0, 2.0])}, eqx.nn.Linear(1, 2, key=key1)]
    pytree2 = [1, 2, 3, {"a": jnp.array([1.0, 2.0])}, eqx.nn.Linear(1, 2, key=key1)]
    pytree3 = [1, 2, 3, {"a": jnp.array([1.0, 2.0])}, eqx.nn.Linear(1, 2, key=key2)]
    pytree4 = [1, 2, 3, {"a": jnp.array([1.0, 4.0])}, eqx.nn.Linear(1, 2, key=key1)]
    pytree5 = [1, 2, 4, {"a": jnp.array([1.0, 2.0])}, eqx.nn.Linear(1, 2, key=key1)]

    assert eqx.tree_equal(pytree1, pytree1, pytree1)
    assert eqx.tree_equal(pytree1, pytree2)
    assert not eqx.tree_equal(pytree1, pytree3)
    assert not eqx.tree_equal(pytree1, pytree4)
    assert not eqx.tree_equal(pytree1, pytree5)


def test_tree_equal_jit():
    a = jnp.array(0)
    b = jnp.array(0)

    @jax.jit
    def run1():
        assert not eqx.tree_equal(a, 0)

    run1()

    @jax.jit
    def run2():
        return eqx.tree_equal(a, b)

    assert run2()

    @jax.jit
    def run3(x, y):
        return eqx.tree_equal(x, y)

    assert run3(a, b)
    assert not run3(a, 1)


def test_tree_inference(getkey):
    attention = eqx.nn.MultiheadAttention(2, 4, key=getkey())
    assert attention.dropout.inference is False
    attention2 = eqx.tree_inference(attention, True)
    assert attention.dropout.inference is False
    assert attention2.dropout.inference is True


def test_tree_flatten_one_level():
    x = {"a": 3, "b": (1, 2)}
    leaves, treedef = eqx.tree_flatten_one_level(x)
    assert leaves == ([3, (1, 2)])
    assert treedef == jtu.tree_structure({"a": 0, "b": 0})

    y = 4
    leaves, treedef = eqx.tree_flatten_one_level(y)
    assert leaves == [4]
    assert treedef == jtu.tree_structure(0)

    x = []
    y = []
    x.append(y)
    y.append(x)
    leaves, treedef = eqx.tree_flatten_one_level(x)
    assert leaves == [y]
    assert treedef == jtu.tree_structure([0])

    x = []
    x.append(x)
    with pytest.raises(ValueError):
        eqx.tree_flatten_one_level(x)


def test_tree_check():
    x = []
    y = []
    x.append(y)
    y.append(x)
    with pytest.raises(ValueError):
        eqx.tree_check(x)

    x = []
    x.append(x)
    with pytest.raises(ValueError):
        eqx.tree_check(x)

    # From https://github.com/patrick-kidger/equinox/issues/327
    class Component(eqx.Module):
        transform: Callable[[float], float]
        validator: Callable[[Callable], Callable]

        def __init__(self, transform=lambda x: x, validator=lambda f: f) -> None:
            self.transform = transform
            self.validator = validator

        def __call__(self, x):
            return self.validator(self.transform)(x)

    class SubComponent(Component):
        test: Callable[[float], float]

        def __init__(self, test=lambda x: 2 * x) -> None:
            self.test = test
            super().__init__(self._transform)

        def _transform(self, x):
            return self.test(x)

    a = SubComponent()
    with pytest.raises(ValueError):
        eqx.tree_check(a)
