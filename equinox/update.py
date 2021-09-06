import jax

from .custom_types import PyTree


_sentinel = object()


def _apply_update(u, p):
    if u is _sentinel:
        return p
    else:
        return p + u


def apply_updates(model: PyTree, updates: PyTree) -> PyTree:
    # Assumes that updates is a prefix of model
    updates = jax._src.tree_util._replace_nones(_sentinel, updates)
    return jax.tree_map(_apply_update, updates, model)
