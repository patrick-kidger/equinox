import jax

from .custom_types import PyTree


def _apply_update(u, p):
    if u is None:
        return p
    else:
        return p + u


def _is_none(x):
    return x is None


def apply_updates(model: PyTree, updates: PyTree) -> PyTree:
    # Assumes that updates is a prefix of model
    return jax.tree_map(_apply_update, updates, model, is_leaf=_is_none)
