import functools as ft
import operator

import jax
import jax.numpy as jnp
import numpy as np


def _shaped_allclose(x, y, **kwargs):
    if type(x) is not type(y):
        return False
    if isinstance(x, jnp.ndarray):
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
    else:
        return x == y


def shaped_allclose(x, y, **kwargs):
    """As `jnp.allclose`, except:
    - It also supports PyTree arguments.
    - It mandates that shapes match as well (no broadcasting)
    """
    same_structure = jax.tree_structure(x) == jax.tree_structure(y)
    allclose = ft.partial(_shaped_allclose, **kwargs)
    return same_structure and jax.tree_util.tree_reduce(
        operator.and_, jax.tree_map(allclose, x, y), True
    )
