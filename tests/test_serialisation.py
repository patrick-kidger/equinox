import jax.numpy as jnp
import numpy as np

import equinox as eqx
import equinox.serialisation


def test_leaf_serialisation(getkey, tmp_path):
    jax_array1 = jnp.array(1)
    jax_array2 = jnp.array([1.0, 2.0])
    numpy_array1 = np.array(1)
    numpy_array2 = np.array([1.0, 2.0])
    scalars = (True, 1, 1.0, 1 + 1j)
    index = eqx.experimental.StateIndex()
    index_value = jnp.array(9)
    eqx.experimental.set_state(index, index_value)
    func = lambda x: x
    obj = object()
    tree = (
        jax_array1,
        jax_array2,
        numpy_array1,
        numpy_array2,
        scalars,
        index,
        func,
        obj,
    )

    eqx.tree_serialise_leaves(tmp_path, tree)

    like_jax_array1 = jnp.array(5)
    like_jax_array2 = jnp.array([6.0, 7.0])
    like_numpy_array1 = np.array(5)
    like_numpy_array2 = np.array([6.0, 7.0])
    like_scalars = (False, 6, 6.0, 6 + 6j)
    like_index = eqx.experimental.StateIndex()
    eqx.experimental.set_state(index, jnp.array(6))
    like_func = lambda x: x
    like_obj = object()
    like = (
        like_jax_array1,
        like_jax_array2,
        like_numpy_array1,
        like_numpy_array2,
        like_scalars,
        like_index,
        like_func,
        like_obj,
    )

    tree_loaded = eqx.tree_deserialise_leaves(tmp_path, like)

    tree_serialisable = tree[:-3]
    tree_loaded_serialisable = tree_loaded[:-3]
    tree_loaded_index, tree_loaded_func, tree_loaded_obj = tree_loaded[-3:]
    import jax

    print(jax.tree_leaves(tree_loaded))
    print(jax.tree_leaves(tree_serialisable))
    print("*" * 50)
    assert eqx.tree_equal(tree_serialisable, tree_loaded_serialisable)
    assert tree_loaded_index is like_index
    assert jnp.array_equal(
        eqx.experimental.get_state(like_index, jnp.array(4)), index_value
    )
    assert tree_loaded_func is like_func
    assert tree_loaded_obj is like_obj


def test_custom_leaf_serialisation(getkey, tmp_path):
    jax_array1 = jnp.array(1)
    jax_array2 = jnp.array([1.0, 2.0])
    numpy_array1 = np.array(1)
    numpy_array2 = np.array([1.0, 2.0])
    scalars = (True, 1, 1.0, 1 + 1j)
    index = eqx.experimental.StateIndex()
    index_value = jnp.array(9)
    eqx.experimental.set_state(index, index_value)
    func = lambda x: x
    obj = object()
    tree = (
        jax_array1,
        jax_array2,
        numpy_array1,
        numpy_array2,
        scalars,
        index,
        func,
        obj,
    )

    def ser_filter_spec(f, x):
        if isinstance(x, jnp.ndarray):
            pass
        else:
            return equinox.serialisation.default_serialise_filter_spec(f, x)

    eqx.tree_serialise_leaves(tmp_path, tree, filter_spec=ser_filter_spec)

    like_numpy_array1 = np.array(5)
    like_numpy_array2 = np.array([6.0, 7.0])
    like_scalars = (False, 6, 6.0, 6 + 6j)
    like_index = eqx.experimental.StateIndex()
    eqx.experimental.set_state(index, jnp.array(6))
    like_func = lambda x: x
    like_obj = object()
    like = (
        like_numpy_array1,
        like_numpy_array2,
        like_scalars,
        like_index,
        like_func,
        like_obj,
    )

    unlike_func = lambda x: -x

    def deser_filter_spec(f, x):
        if callable(x):
            return unlike_func
        else:
            return equinox.serialisation.default_deserialise_filter_spec(f, x)

    tree_loaded = eqx.tree_deserialise_leaves(
        tmp_path, like, filter_spec=deser_filter_spec
    )
    tree_loaded_index, tree_loaded_func, tree_loaded_obj = tree_loaded[-3:]
    tree_ser_exp, tree_ser_func, tree_ser_obj = tree[-3:]
    tree_loaded_same = tree_loaded[:3]
    tree_ser_same = tree[2:5]

    assert eqx.tree_equal(tree_loaded_same, tree_ser_same)
    assert jnp.array_equal(
        eqx.experimental.get_state(tree_loaded_index, jnp.array(4)), index_value
    )
    assert tree_loaded_func is unlike_func
    assert tree_loaded_obj is like_obj
    assert tree_loaded_obj is not tree_ser_obj
