import collections as co

import equinox as eqx
import jax
import jax.core
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import numpy as np
import pytest


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


def test_tree_at_empty_tuple():
    # Tuples are singletons, so we have a specific test for the wrapper
    a = ()
    x1 = [a]
    x2 = [a, a]
    x3 = [(), ()]

    b = (1,)
    x4 = [b]
    x5 = [b, b]
    x6 = [(1,), (1,)]

    Empty = co.namedtuple("Empty", [])
    empty = Empty()
    x7 = [empty]
    x8 = [empty, empty]
    x9 = [Empty(), Empty()]

    for x in (x1, x2, x3, x4, x5, x6, x7, x8, x9):
        # Test with replace
        new_x = eqx.tree_at(lambda xi: xi[0], x, "hello")
        assert new_x[0] == "hello"
        if len(new_x) != 1:
            assert new_x[1] != "hello"

        # Test with replace fn
        expected = len(x[0])
        new_x = eqx.tree_at(lambda xi: xi[0], x, replace_fn=len)
        assert new_x[0] == expected


def test_tree_at_empty_namedtuple():
    Empty = co.namedtuple("Empty", [])
    pytree = [Empty(), 5]
    out = eqx.tree_at(lambda x: x[1], pytree, 4)
    assert isinstance(out[0], Empty)

    # Test with replace fn
    expected = str(pytree[0])
    out = eqx.tree_at(lambda x: x[0], pytree, replace_fn=str)
    assert out[0] == expected


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


def _typeequal(x, y):
    return (type(x) == type(y)) and (x == y)


def test_tree_equal():
    key1 = jrandom.PRNGKey(0)
    key2 = jrandom.PRNGKey(1)
    # Not using `getkey` as ever-so-in-principle two random keys could produce the same
    # weights (like that's ever going to happen).
    pytree1 = [1, 2, 3, {"a": jnp.array([1.0, 2.0])}, eqx.nn.Linear(1, 2, key=key1)]
    pytree2 = [1, 2, 3, {"a": jnp.array([1.0, 2.0])}, eqx.nn.Linear(1, 2, key=key1)]
    pytree3 = [1, 2, 3, {"a": jnp.array([1.0, 2.0])}, eqx.nn.Linear(1, 2, key=key2)]
    pytree4 = [1, 2, 3, {"a": jnp.array([1.0, 4.0])}, eqx.nn.Linear(1, 2, key=key1)]
    pytree5 = [1, 2, 4, {"a": jnp.array([1.0, 2.0])}, eqx.nn.Linear(1, 2, key=key1)]

    assert _typeequal(eqx.tree_equal(pytree1, pytree1, pytree1), jnp.array(True))
    assert _typeequal(eqx.tree_equal(pytree1, pytree2), jnp.array(True))
    assert _typeequal(eqx.tree_equal(pytree1, pytree3), jnp.array(False))
    assert _typeequal(eqx.tree_equal(pytree1, pytree4), jnp.array(False))
    assert _typeequal(eqx.tree_equal(pytree1, pytree5), False)


def test_tree_equal_jit():
    a = jnp.array(0)
    b = jnp.array(0)

    @jax.jit
    def run1():
        assert _typeequal(eqx.tree_equal(a, 0), False)

    run1()

    @jax.jit
    def run2():
        return eqx.tree_equal(a, b)

    assert _typeequal(run2(), jnp.array(True))

    @jax.jit
    def run3(x, y):
        return eqx.tree_equal(x, y)

    assert _typeequal(run3(a, b), jnp.array(True))
    assert _typeequal(run3(a, 1), jnp.array(False))


def test_tree_equal_numpy():
    x = np.array([1, 2], dtype=np.float32)
    x2 = np.array([1, 2], dtype=np.float32)
    y = jnp.array([1, 2], dtype=jnp.float32)
    z = jnp.array([1, 2], dtype=jnp.float16)
    assert _typeequal(eqx.tree_equal(x, x), True)
    assert _typeequal(eqx.tree_equal(x, x2), True)
    assert _typeequal(eqx.tree_equal(x, x2, typematch=True), True)
    assert _typeequal(eqx.tree_equal(x, y), jnp.array(True))
    assert _typeequal(eqx.tree_equal(x, y, typematch=True), False)
    assert _typeequal(eqx.tree_equal(y, y), jnp.array(True))
    assert _typeequal(eqx.tree_equal(y, y, typematch=True), jnp.array(True))
    assert _typeequal(eqx.tree_equal(x, z), False)
    assert _typeequal(eqx.tree_equal(y, z), False)

    @jax.jit
    def f():
        assert _typeequal(eqx.tree_equal(x, x), True)
        assert _typeequal(eqx.tree_equal(x, y, typematch=True), False)
        out = eqx.tree_equal(x, y)
        assert isinstance(out, jax.core.Tracer)
        return out

    assert _typeequal(f(), jnp.array(True))


def test_tree_equal_scalars():
    x = np.float32(1)
    y = np.array(1, dtype=np.float32)
    z = np.array(1, dtype=np.float16)
    # scalar-ness does not matter
    assert _typeequal(eqx.tree_equal(x, y), True)
    # dtype does matter
    assert _typeequal(eqx.tree_equal(x, z), False)

    z = jax.dtypes.bfloat16(1)
    z2 = jax.dtypes.bfloat16(1)
    w = jax.dtypes.bfloat16(2)
    assert _typeequal(eqx.tree_equal(z, z2), True)
    assert _typeequal(eqx.tree_equal(z, w), False)


def test_tree_allclose():
    x = np.array(1.0, dtype=np.float32)
    y = np.array(1.00001, dtype=np.float32)
    z = jnp.array(1.00001, dtype=np.float32)
    assert _typeequal(eqx.tree_equal(x, y), False)
    assert _typeequal(eqx.tree_equal(x, y, atol=1e-3), True)
    assert _typeequal(eqx.tree_equal(x, z, atol=1e-3), jnp.array(True))
    assert _typeequal(eqx.tree_equal(x, z, typematch=True, atol=1e-3), False)


def test_inference_mode(getkey):
    attention = eqx.nn.MultiheadAttention(2, 4, key=getkey())
    assert attention.dropout.inference is False
    attention2 = eqx.nn.inference_mode(attention)
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


# This matches the behaviour of `jax._src.tree_util.flatten_one_level`
def test_tree_flatten_one_level_special():
    x = [None, None, eqx.Module(), 1, 2]
    leaves, treedef = eqx.tree_flatten_one_level(x)
    assert leaves == [None, None, eqx.Module(), 1, 2]
    assert treedef == jtu.tree_structure([0, 0, 0, 0, 0])


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


def test_tree_check_none():
    eqx.tree_check([None, None])


def test_tree_check_integer():
    eqx.tree_check([0, 0])


def test_tree_check_module():
    a = eqx.Module()  # same `id(...)` for both entries passed to `tree_check`.
    eqx.tree_check([a, a])
