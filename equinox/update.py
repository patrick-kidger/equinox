import jax

from .custom_types import PyTree


def _apply_update(p, u):
    if u is None:
        return p
    else:
        return p + u


def apply_updates(model: PyTree, updates: PyTree) -> PyTree:
    return jax.tree_map(_apply_update, model, updates)
