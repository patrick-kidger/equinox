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
    """Returns `True` if `element` is a JAX array (but not a NumPy array)."""
    return isinstance(element, jnp.ndarray)


# Does _not_ do a try/except on jnp.asarray(element) because that's very slow.
# Chosen to match
# https://github.com/google/jax/blob/4a17c78605e7fc69a69a999e2f6298db79d3837a/jax/_src/numpy/lax_numpy.py#L542  # noqa: E501
def is_array_like(element: Any) -> bool:
    """Returns `True` if `element` is a JAX array, a NumPy array, or a Python
    `float`/`complex`/`bool`/`int`.
    """
    return isinstance(
        element, (jnp.ndarray, np.ndarray, float, complex, bool, int)
    ) or hasattr(element, "__jax_array__")


def is_inexact_array(element: Any) -> bool:
    """Returns `True` if `element` is an inexact (i.e. floating point) JAX array."""
    return is_array(element) and jnp.issubdtype(element.dtype, jnp.inexact)


def is_inexact_array_like(element: Any) -> bool:
    """Returns `True` if `element` is an inexact JAX array, an inexact NumPy array, or
    a Python `float` or `complex`.
    """
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
    """
    Filters out the leaves of a PyTree not satisfying a condition. Those not satisfying
    the condition are replaced with `replace`.

    **Arguments:**

    - `pytree` is any PyTree.
    - `filter_spec` is a PyTree whose structure should be a prefix of the structure of
        `pytree`. Each of its leaves should either be:
        - `True`, in which case the leaf or subtree is kept;
        - `False`, in which case the leaf or subtree is replaced with `replace`;
        - a callable `Leaf -> bool`, in which case this is evaluted on the leaf or
            mapped over the subtree, and the leaf kept or replaced as appropriate.
    - `inverse` switches the truthy/falsey behaviour: falsey results are kept and
        truthy results are replaced.
    - `replace` is what to replace any falsey leaves with. Defaults to `None`.

    **Returns:**

    A PyTree of the same structure as `pytree`.

    !!! info

        A common special case is `equinox.filter(pytree, equinox.is_array)`. Then
        `equinox.is_array` is evaluted on all of `pytree`'s leaves, and each leaf then
        kept or replaced.

    !!! info

        See also [`equinox.combine`][] to reconstitute the PyTree again.
    """

    inverse = bool(inverse)  # just in case, to make the != trick below work reliably
    filter_tree = jax.tree_map(_make_filter_tree, filter_spec, pytree)
    return jax.tree_map(
        lambda mask, x: x if bool(mask) != inverse else replace, filter_tree, pytree
    )


def partition(pytree: PyTree, filter_spec: PyTree, replace: Any = None) -> PyTree:
    """Equivalent to `filter(...), filter(..., inverse=True)`, but slightly more
    efficient.
    """

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


def combine(*pytrees: PyTree) -> PyTree:
    """Combines multiple PyTrees into one PyTree, by replacing `None` leaves.

    !!! example

        ```python
        pytree1 = [None, 1, 2]
        pytree2 = [0, None, None]
        equinox.combine(pytree1, pytree2)  # [0, 1, 2]
        ```

    !!! tip

        The idea is that `equinox.combine` should be used to undo a call to
        [`equinox.filter`][] or [`equinox.partition`][].

    **Arguments:**

    - `*pytrees`: a sequence of PyTrees all with the same structure.

    **Returns:**

    A PyTree with the same structure as its inputs. Each leaf will be the first
    non-`None` leaf found in the corresponding leaves of `pytrees` as they are
    iterated over.
    """

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
