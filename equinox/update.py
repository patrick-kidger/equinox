import jax
import jax.numpy as jnp

from .custom_types import PyTree


def _apply_update(p, u):
    u = jnp.asarray(u)
    if jnp.count_nonzero(u) == 0:
        return p
    else:
        p = jnp.asarray(p)
        u = u.astype(p.dtype)
        return p + u


def apply_updates(model: PyTree, updates: PyTree) -> PyTree:
    return jax.tree_map(_apply_update, model, updates)
