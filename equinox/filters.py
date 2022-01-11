from typing import Any, Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from .custom_types import PyTree, TreeDef
from .deprecated import deprecated


#
# Filter functions
#


def is_array(element: Any) -> bool:
    return isinstance(element, jnp.ndarray)


# Does _not_ do a try/except on jnp.asarray(element) because that's very slow.
# Chosen to match
# https://github.com/google/jax/blob/4a17c78605e7fc69a69a999e2f6298db79d3837a/jax/_src/numpy/lax_numpy.py#L542  # noqa: E501
def is_array_like(element: Any) -> bool:
    return isinstance(
        element, (jnp.ndarray, np.ndarray, float, complex, bool, int)
    ) or hasattr(element, "__jax_array__")


def is_inexact_array(element: Any) -> bool:
    return is_array(element) and jnp.issubdtype(element.dtype, jnp.inexact)


def is_inexact_array_like(element: Any) -> bool:
    if hasattr(element, "__jax_array__"):
        element = element.__jax_array__()
    return (
        isinstance(element, (jnp.ndarray, np.ndarray))
        and jnp.issubdtype(element.dtype, jnp.inexact)
    ) or isinstance(element, (float, complex))


#
# Filtering/combining
#


def _make_filter_tree(mask: Union[bool, Callable[[Any], bool]], arg: Any) -> bool:
    if isinstance(mask, bool):
        return mask
    elif callable(mask):
        return jax.tree_map(mask, arg)
    else:
        raise ValueError("`filter_spec` must consist of booleans and callables only.")


def filter(
    pytree: PyTree, filter_spec: PyTree, inverse: bool = False, replace: Any = None
) -> PyTree:

    inverse = bool(inverse)  # just in case, to make the != trick below work reliably
    filter_tree = jax.tree_map(_make_filter_tree, filter_spec, pytree)
    return jax.tree_map(
        lambda mask, x: x if bool(mask) != inverse else replace, filter_tree, pytree
    )


def partition(pytree: PyTree, filter_spec: PyTree, replace: Any = None) -> PyTree:

    filter_tree = jax.tree_map(_make_filter_tree, filter_spec, pytree)
    left = jax.tree_map(lambda mask, x: x if mask else replace, filter_tree, pytree)
    right = jax.tree_map(lambda mask, x: replace if mask else x, filter_tree, pytree)
    return left, right


def _combine(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


def _is_none(x):
    return x is None


def combine(*pytrees: PyTree):
    return jax.tree_map(_combine, *pytrees, is_leaf=_is_none)


#
# Deprecated
#


@deprecated(in_favour_of=filter)
def split(
    pytree: PyTree,
    filter_fn: Optional[Callable[[Any], bool]] = None,
    filter_tree: Optional[PyTree] = None,
) -> Tuple[List[Any], List[Any], List[bool], TreeDef]:
    validate_filters("split", filter_fn, filter_tree)
    flat, treedef = jax.tree_flatten(pytree)
    flat_true = []
    flat_false = []

    if filter_fn is None:
        which, treedef_filter = jax.tree_flatten(filter_tree)
        if treedef != treedef_filter:
            raise ValueError(
                "filter_tree must have the same tree structure as the PyTree being split."
            )
        for f, w in zip(flat, which):
            if w:
                flat_true.append(f)
            else:
                flat_false.append(f)
    else:
        which = []
        for f in flat:
            if filter_fn(f):
                flat_true.append(f)
                which.append(True)
            else:
                flat_false.append(f)
                which.append(False)

    return flat_true, flat_false, which, treedef


@deprecated(in_favour_of=combine)
def merge(
    flat_true: List[Any], flat_false: List[Any], which: List[bool], treedef: TreeDef
):
    flat = []
    flat_true = iter(flat_true)
    flat_false = iter(flat_false)
    for element in which:
        if element:
            flat.append(next(flat_true))
        else:
            flat.append(next(flat_false))
    return jax.tree_unflatten(treedef, flat)


# Internal and only used by deprecated functions
def validate_filters(fn_name, filter_fn, filter_tree):
    if (filter_fn is None and filter_tree is None) or (
        filter_fn is not None and filter_tree is not None
    ):
        raise ValueError(
            f"Precisely one of `filter_fn` and `filter_tree` should be passed to {fn_name}"
        )
