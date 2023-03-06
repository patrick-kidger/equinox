import functools as ft
import operator

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np


def random_pytree(key, treedef):
    keys = jr.split(key, treedef.num_leaves)
    leaves = []
    for key in keys:
        dimkey, sizekey, valuekey = jr.split(key, 3)
        num_dims = jr.randint(dimkey, (), 0, 5).item()
        dim_sizes = jr.randint(sizekey, (num_dims,), 0, 5).tolist()
        value = jr.normal(valuekey, dim_sizes)
        leaves.append(value)
    return jtu.tree_unflatten(treedef, leaves)


treedefs = [
    jtu.tree_structure(x)
    for x in (
        0,
        None,
        {"a": [0, 0], "b": 0},
    )
]


def _shaped_allclose(x, y, *, match_weak, **kwargs):
    if match_weak:
        x = jnp.asarray(x)
        y = jnp.asarray(y)
    if type(x) is not type(y):
        return False
    if isinstance(x, jax.Array):
        if jnp.issubdtype(x.dtype, jnp.inexact):
            return (
                x.shape == y.shape
                and x.dtype == y.dtype
                and jnp.allclose(x, y, **kwargs)
            )
        else:
            return x.shape == y.shape and x.dtype == y.dtype and jnp.all(x == y)
    elif isinstance(x, np.ndarray):
        if np.issubdtype(x.dtype, np.inexact):
            return (
                x.shape == y.shape
                and x.dtype == y.dtype
                and np.allclose(x, y, **kwargs)
            )
        else:
            return x.shape == y.shape and x.dtype == y.dtype and np.all(x == y)
    elif isinstance(x, jax.ShapeDtypeStruct):
        assert x.shape == y.shape and x.dtype == y.dtype
    else:
        return x == y


def shaped_allclose(x, y, *, match_weak=False, **kwargs):
    """As `jnp.allclose`, except:
    - It also supports PyTree arguments.
    - It mandates that shapes match as well (no broadcasting)
    """
    same_structure = jtu.tree_structure(x) == jtu.tree_structure(y)
    allclose = ft.partial(_shaped_allclose, match_weak=match_weak, **kwargs)
    return same_structure and jtu.tree_reduce(
        operator.and_, jtu.tree_map(allclose, x, y), True
    )
