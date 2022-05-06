from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .custom_types import BoolAxisSpec, PyTree, ResolvedBoolAxisSpec


#
# Filter functions
#


def is_array(element: Any) -> bool:
    """Returns `True` if `element` is a JAX array (but not a NumPy array)."""
    return isinstance(element, jnp.ndarray)


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


def _make_filter_tree(is_leaf):
    def _filter_tree(mask: BoolAxisSpec, arg: Any) -> ResolvedBoolAxisSpec:
        if isinstance(mask, bool):
            return jax.tree_map(lambda _: mask, arg, is_leaf=is_leaf)
        elif callable(mask):
            return jax.tree_map(mask, arg, is_leaf=is_leaf)
        else:
            raise ValueError(
                "`filter_spec` must consist of booleans and callables only."
            )

    return _filter_tree


def filter(
    pytree: PyTree,
    filter_spec: PyTree[BoolAxisSpec],
    inverse: bool = False,
    replace: Any = None,
    is_leaf: Optional[Callable[[Any], bool]] = None,
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
    - `is_leaf`: Optional function called at each node of the PyTree. It should return
        a boolean. `True` indicates that the whole subtree should be treated as leaf;
        `False` indicates that the subtree should be traversed as a PyTree. This is
        mostly useful for evaluating a callable `filter_spec` on a node instead of a
        leaf.

    **Returns:**

    A PyTree of the same structure as `pytree`.

    !!! info

        A common special case is `equinox.filter(pytree, equinox.is_array)`. Then
        `equinox.is_array` is evaluted on all of `pytree`'s leaves, and each leaf then
        kept or replaced.
    """

    inverse = bool(inverse)  # just in case, to make the != trick below work reliably
    filter_tree = jax.tree_map(_make_filter_tree(is_leaf), filter_spec, pytree)
    return jax.tree_map(
        lambda mask, x: x if bool(mask) != inverse else replace, filter_tree, pytree
    )


def partition(
    pytree: PyTree,
    filter_spec: PyTree[BoolAxisSpec],
    replace: Any = None,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> PyTree:
    """Equivalent to `filter(...), filter(..., inverse=True)`, but slightly more
    efficient.

    !!! info

        See also [`equinox.combine`][] to reconstitute the PyTree again.
    """

    filter_tree = jax.tree_map(_make_filter_tree(is_leaf), filter_spec, pytree)
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
