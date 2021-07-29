import jax
import jax.numpy as jnp

from .annotations import get_annotation, set_annotation
from .custom_types import PyTree


def _apply_update(p, u):
    u = jnp.asarray(u)
    if jnp.count_nonzero(u) == 0:
        return p
    else:
        p = jnp.asarray(p)
        u = u.astype(p.dtype)
        out = p + u
        # Propagate annotations for convenience
        try:
            annotation = get_annotation(p)
        except KeyError:
            pass
        else:
            set_annotation(out, annotation)
        return out


def apply_updates(params: PyTree, updates: PyTree) -> PyTree:
    return jax.tree_map(_apply_update, params, updates)
