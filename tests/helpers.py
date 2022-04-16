import functools as ft
import operator

import jax
import jax.numpy as jnp


def _shaped_allclose(x, y, **kwargs):
    return jnp.shape(x) == jnp.shape(y) and jnp.allclose(x, y, **kwargs)


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
