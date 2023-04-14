import os

import jax
import jax.numpy as jnp
import numpy as np

import equinox as eqx


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
