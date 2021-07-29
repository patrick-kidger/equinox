import jax
import jax.numpy as jnp
from typing import Any, Callable, List, Tuple

from .custom_types import PyTree, TreeDef


# TODO: not sure if this is the best way to do this? In light of:
# https://github.com/google/jax/commit/258ae44303b1539eff6253263694ec768b8803f0#diff-de759f969102e9d64b54a299d11d5f0e75cfe3052dc17ffbcd2d43b250719fb0
def is_inexact_array(element: Any) -> bool:
    try:
        element = jnp.asarray(element)
    except Exception:
        return False
    else:
        return jnp.issubdtype(element.dtype, jnp.inexact)


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
