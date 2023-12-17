import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.dtypes import bfloat16


def _example_trees():
    jax_array1 = jnp.array(1)
    jax_array2 = jnp.array([1.0, 2.0])
    numpy_array1 = np.array(1)
    numpy_array2 = np.array([1.0, 2.0])
    scalars = (True, 1, 1.0, 1 + 1j)
    func = lambda x: x
    obj = object()
    tree = (
        jax_array1,
        jax_array2,
        numpy_array1,
        numpy_array2,
        scalars,
        func,
        obj,
    )

    like_jax_array1 = jnp.array(5)
    like_jax_array2 = jnp.array([6.0, 7.0])
    like_numpy_array1 = np.array(5)
    like_numpy_array2 = np.array([6.0, 7.0])
    like_scalars = (False, 6, 6.0, 6 + 6j)
    like_func = lambda x: x
    like_obj = object()
    like = (
        like_jax_array1,
        like_jax_array2,
        like_numpy_array1,
        like_numpy_array2,
        like_scalars,
        like_func,
        like_obj,
    )

    return tree, like, like_func, like_obj


def test_leaf_serialisation_path(getkey, tmp_path):
    tree, like, like_func, like_obj = _example_trees()

    eqx.tree_serialise_leaves(tmp_path, tree)

    tree_loaded = eqx.tree_deserialise_leaves(tmp_path, like)

    tree_serialisable = tree[:-2]
    tree_loaded_serialisable = tree_loaded[:-2]
    tree_loaded_func, tree_loaded_obj = tree_loaded[-2:]

    assert eqx.tree_equal(tree_serialisable, tree_loaded_serialisable)
    assert tree_loaded_func is like_func
    assert tree_loaded_obj is like_obj


def test_leaf_serialisation_file(getkey, tmp_path):
    tree, like, like_func, like_obj = _example_trees()

    with open(os.path.join(tmp_path, "test.eqx"), "wb") as tmp_file:
        eqx.tree_serialise_leaves(tmp_file, tree)

    with open(os.path.join(tmp_path, "test.eqx"), "rb") as tmp_file:
        tree_loaded = eqx.tree_deserialise_leaves(tmp_file, like)

    tree_serialisable = tree[:-2]
    tree_loaded_serialisable = tree_loaded[:-2]
    tree_loaded_func, tree_loaded_obj = tree_loaded[-2:]

    assert eqx.tree_equal(tree_serialisable, tree_loaded_serialisable)
    assert tree_loaded_func is like_func
    assert tree_loaded_obj is like_obj


def test_helpful_errors(getkey, tmp_path):
    # Test that we get a helpful error message when the loading itself fails
    tree = jnp.array(1), {}
    eqx.tree_serialise_leaves(tmp_path, tree)
    bad_like_tree = (
        jnp.array(2),
        {
            "a": jnp.array(2),
            "b": jnp.array(2),
        },
    )
    with pytest.raises(
        RuntimeError,
        match=r"Error at leaf with path \(SequenceKey\(idx=1\), DictKey\(key='a'\)\)",
    ):
        _ = eqx.tree_deserialise_leaves(tmp_path, bad_like_tree)

    # Test that we get a helpful error message when the types don't match
    tree = (
        jnp.array(1),
        {
            "a": jnp.array(1),
            "b": jnp.array(1),
        },
    )
    eqx.tree_serialise_leaves(tmp_path, tree)
    bad_like_tree = (
        jnp.array(2),
        {
            "a": jnp.array(2, dtype=jnp.float32),
            "b": jnp.array(2),
        },
    )
    with pytest.raises(
        RuntimeError,
        match=r"Deserialised leaf at path \(SequenceKey\(idx=1\), DictKey\(key='a'\)\)",
    ):
        _ = eqx.tree_deserialise_leaves(tmp_path, bad_like_tree)


def test_generic_dtype_serialisation(getkey, tmp_path):
    # Ensure we can round trip when we start with an array
    jax_array = jnp.array(bfloat16(1))
    eqx.tree_serialise_leaves(tmp_path, jax_array)
    like_jax_array = jnp.array(bfloat16(2))
    loaded_jax_array = eqx.tree_deserialise_leaves(tmp_path, like_jax_array)
    assert jax_array.item() == loaded_jax_array.item()

    tree = (
        jnp.array(1e-8),
        bfloat16(1e-8),
        np.float32(1e-8),
        jnp.array(1e-8),
        np.float64(1e-8),
    )
    like_tree = (
        jnp.array(2.0),
        bfloat16(2),
        np.float32(2),
        jnp.array(2.0),
        np.float64(2.0),
    )

    # Ensure we can round trip when we start with a scalar
    eqx.tree_serialise_leaves(tmp_path, tree)
    loaded_tree = eqx.tree_deserialise_leaves(tmp_path, like_tree)
    assert len(loaded_tree) == len(tree)
    for a, b in zip(loaded_tree, tree):
        assert type(a) is type(b)
        assert a.item() == b.item()

    # Ensure we can round trip when we start with a scalar that we've JAX JITed
    # `[:-1]` to skip float64, as JIT turns it into float32
    eqx.tree_serialise_leaves(tmp_path, jax.jit(lambda x: x)(tree[:-1]))
    loaded_tree = eqx.tree_deserialise_leaves(tmp_path, like_tree[:-1])
    assert len(loaded_tree) == len(tree[:-1])
    for a, b in zip(loaded_tree, tree):
        assert type(a) is type(b)
        assert a.item() == b.item()


def test_python_scalar(tmp_path):
    eqx.tree_serialise_leaves(tmp_path, 1e-8)
    out = eqx.tree_deserialise_leaves(tmp_path, 0.0)
    assert out == 1e-8


def test_custom_leaf_serialisation(getkey, tmp_path):
    jax_array1 = jnp.array(1)
    jax_array2 = jnp.array([1.0, 2.0])
    numpy_array1 = np.array(1)
    numpy_array2 = np.array([1.0, 2.0])
    scalars = (True, 1, 1.0, 1 + 1j)
    func = lambda x: x
    obj = object()
    tree = (
        jax_array1,
        jax_array2,
        numpy_array1,
        numpy_array2,
        scalars,
        func,
        obj,
    )

    def ser_filter_spec(f, x):
        if isinstance(x, jax.Array):
            pass
        else:
            return eqx.default_serialise_filter_spec(f, x)

    eqx.tree_serialise_leaves(tmp_path, tree, filter_spec=ser_filter_spec)

    like_numpy_array1 = np.array(5)
    like_numpy_array2 = np.array([6.0, 7.0])
    like_scalars = (False, 6, 6.0, 6 + 6j)
    like_func = lambda x: x
    like_obj = object()
    like = (
        like_numpy_array1,
        like_numpy_array2,
        like_scalars,
        like_func,
        like_obj,
    )

    unlike_func = lambda x: -x

    def deser_filter_spec(f, x):
        if callable(x):
            return unlike_func
        else:
            return eqx.default_deserialise_filter_spec(f, x)

    tree_loaded = eqx.tree_deserialise_leaves(
        tmp_path, like, filter_spec=deser_filter_spec
    )
    tree_loaded_func, tree_loaded_obj = tree_loaded[-2:]
    tree_ser_func, tree_ser_obj = tree[-2:]
    tree_loaded_same = tree_loaded[:3]
    tree_ser_same = tree[2:5]

    assert eqx.tree_equal(tree_loaded_same, tree_ser_same)
    assert tree_loaded_func is unlike_func
    assert tree_loaded_obj is like_obj
    assert tree_loaded_obj is not tree_ser_obj


# Deserialisation and seralisation work despite both `model` and `model2` having
# different objects as the markers in their `StateIndex`s.
# This is because we iterate over `model` and `model2` in the same order when
# creating `State(model)` and `State(model2)`. Thus the entries in `state` and `state2`
# are stored in the same order (as Python dictionaries are ordered). So standard
# de/serialisation just matches things up automatically.
def test_stateful(tmp_path):
    class Model(eqx.Module):
        norm1: eqx.nn.BatchNorm
        norm2: eqx.nn.BatchNorm

    model = Model(eqx.nn.BatchNorm(3, "hi"), eqx.nn.BatchNorm(4, "bye"))
    state = eqx.nn.State(model)

    eqx.tree_serialise_leaves(tmp_path, (model, state))

    model2 = Model(eqx.nn.BatchNorm(3, "hi"), eqx.nn.BatchNorm(4, "bye"))
    state2 = eqx.nn.State(model2)

    eqx.tree_deserialise_leaves(tmp_path, (model2, state2))


def test_eval_shape(getkey, tmp_path):
    model = eqx.nn.MLP(2, 2, 2, 2, key=getkey())
    eqx.tree_serialise_leaves(tmp_path, model)

    model2 = eqx.filter_eval_shape(eqx.nn.MLP, 2, 2, 2, 2, key=getkey())
    model3 = eqx.tree_deserialise_leaves(tmp_path, model2)

    assert eqx.tree_equal(model, model3, typematch=True)
