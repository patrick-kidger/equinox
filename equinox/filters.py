import jax
import jax.numpy as jnp
import jaxlib
from typing import Any, Callable, List, Tuple

from .custom_types import PyTree, TreeDef


# TODO: not sure if this is the best way to do this? In light of:
# https://github.com/google/jax/commit/258ae44303b1539eff6253263694ec768b8803f0#diff-de759f969102e9d64b54a299d11d5f0e75cfe3052dc17ffbcd2d43b250719fb0
def is_inexact_array(element: Any) -> bool:
    return isinstance(element, (jax.core.Tracer,
                                jaxlib.xla_extension.DeviceArray)) and jnp.issubdtype(element.dtype, jnp.inexact)


def is_array_like(element: Any) -> bool:
    try:
        jnp.asarray(element)
    except Exception:
        return False
    else:
        return True


def split(pytree: PyTree, filter_fn: Callable[[Any], bool]) -> Tuple[List[Any], List[Any], List[bool], TreeDef]:
    flat, treedef = jax.tree_flatten(pytree)
    flat_true = []
    flat_false = []
    which = []
    for f in flat:
        if filter_fn(f):
            flat_true.append(f)
            which.append(True)
        else:
            flat_false.append(f)
            which.append(False)
    return flat_true, flat_false, which, treedef


def split_tree(pytree: PyTree, filter_tree: PyTree):
    flat, treedef = jax.tree_flatten(pytree)
    which, treedef_filter = jax.tree_flatten(filter_tree)
    if treedef != treedef_filter:
        raise ValueError("filter_tree must have the same tree structure as the PyTree being split.")
    flat_true = []
    flat_false = []
    for f, w in zip(flat, which):
        if w:
            flat_true.append(f)
        else:
            flat_false.append(f)
    return flat_true, flat_false, which, treedef


def merge(flat_true: List[Any], flat_false: List[Any], which: List[bool], treedef: TreeDef):
    flat = []
    flat_true = iter(flat_true)
    flat_false = iter(flat_false)
    for element in which:
        if element:
            flat.append(next(flat_true))
        else:
            flat.append(next(flat_false))
    return jax.tree_unflatten(treedef, flat)


def validate_filters(fn_name, filter_fn, filter_tree):
    if (filter_fn is None and filter_tree is None) or (filter_fn is not None and filter_tree is not None):
        raise ValueError(f"Precisely one of `filter_fn` and `filter_tree` should be passed to {fn_name}")
