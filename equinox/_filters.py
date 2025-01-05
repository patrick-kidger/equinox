from collections.abc import Callable
from typing import Any, Optional, overload, TypeVar, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PyTree


AxisSpec = Union[bool, Callable[[Any], bool]]


#
# Filter functions
#


def is_array(element: Any) -> bool:
    """Returns `True` if `element` is a JAX array or NumPy array."""
    return isinstance(element, (np.ndarray, np.generic, jax.Array))


# Chosen to match
# https://github.com/google/jax/blob/4a17c78605e7fc69a69a999e2f6298db79d3837a/jax/_src/numpy/lax_numpy.py#L542  # noqa: E501
def is_array_like(element: Any) -> bool:
    """Returns `True` if `element` is a JAX array, a NumPy array, or a Python
    `float`/`complex`/`bool`/`int`.
    """
    return isinstance(
        element, (jax.Array, np.ndarray, np.generic, float, complex, bool, int)
    ) or hasattr(element, "__jax_array__")


def is_inexact_array(element: Any) -> bool:
    """Returns `True` if `element` is an inexact (i.e. floating or complex) JAX/NumPy
    array.
    """
    if isinstance(element, (np.ndarray, np.generic)):
        return bool(np.issubdtype(element.dtype, np.inexact))
    elif isinstance(element, jax.Array):
        return jnp.issubdtype(element.dtype, jnp.inexact)
    else:
        return False


def is_inexact_array_like(element: Any) -> bool:
    """Returns `True` if `element` is an inexact JAX array, an inexact NumPy array, or
    a Python `float` or `complex`.
    """
    if hasattr(element, "__jax_array__"):
        element = element.__jax_array__()
    if isinstance(element, (np.ndarray, np.generic)):
        return bool(np.issubdtype(element.dtype, np.inexact))
    elif isinstance(element, jax.Array):
        return jnp.issubdtype(element.dtype, jnp.inexact)
    else:
        return isinstance(element, (float, complex))


#
# Filtering/combining
#


def _make_filter_tree(is_leaf):
    def _filter_tree(mask: AxisSpec, arg: Any) -> PyTree[bool]:
        if isinstance(mask, bool):
            return jtu.tree_map(lambda _: mask, arg, is_leaf=is_leaf)
        elif callable(mask):
            return jtu.tree_map(mask, arg, is_leaf=is_leaf)
        else:
            raise ValueError(
                "`filter_spec` must consist of booleans and callables only."
            )

    return _filter_tree


def filter(
    pytree: PyTree,
    filter_spec: PyTree[AxisSpec],
    inverse: bool = False,
    replace: Any = None,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> PyTree:
    """
    Filters out the leaves of a PyTree not satisfying a condition. Those not satisfying
    the condition are replaced with `replace`.

    !!! Example

        ```python
        pytree = [(jnp.array(0), 1), object()]
        result = eqx.filter(pytree, eqx.is_array)
        # [(jnp.array(0), None), None]
        ```

    !!! Example

        ```python
        pytree = [(jnp.array(0), 1), object()]
        result = eqx.filter(pytree, [(False, False), True])
        # [(None, None), object()]
        ```

    **Arguments:**

    - `pytree` is any PyTree.
    - `filter_spec` is a PyTree whose structure should be a prefix of the structure of
        `pytree`. Each of its leaves should either be:
        - `True`, in which case the leaf or subtree is kept;
        - `False`, in which case the leaf or subtree is replaced with `replace`;
        - a callable `Leaf -> bool`, in which case this is evaluated on the leaf or
            mapped over the subtree, and the leaf kept or replaced as appropriate.
    - `inverse` switches the truthy/falsey behaviour: falsey results are kept and
        truthy results are replaced.
    - `replace` is what to replace any falsey leaves with. Defaults to `None`.
    - `is_leaf`: Optional function called at each node of the PyTree. It should return
        a boolean. `True` indicates that the whole subtree should be treated as leaf;
        `False` indicates that the subtree should be traversed as a PyTree.

    **Returns:**

    A PyTree of the same structure as `pytree`.
    """

    inverse = bool(inverse)  # just in case, to make the != trick below work reliably
    filter_tree = jtu.tree_map(_make_filter_tree(is_leaf), filter_spec, pytree)
    return jtu.tree_map(
        lambda mask, x: x if bool(mask) != inverse else replace, filter_tree, pytree
    )


def partition(
    pytree: PyTree,
    filter_spec: PyTree[AxisSpec],
    replace: Any = None,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> tuple[PyTree, PyTree]:
    """Splits a PyTree into two pieces. Equivalent to
    `filter(...), filter(..., inverse=True)`, but slightly more efficient.

    !!! info

        See also [`equinox.combine`][] to reconstitute the PyTree again.
    """

    filter_tree = jtu.tree_map(_make_filter_tree(is_leaf), filter_spec, pytree)
    left = jtu.tree_map(lambda mask, x: x if mask else replace, filter_tree, pytree)
    right = jtu.tree_map(lambda mask, x: replace if mask else x, filter_tree, pytree)
    return left, right


def _combine(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


def _is_none(x):
    return x is None


_T = TypeVar("_T", bound=PyTree)


@overload
def combine(*pytrees: _T, is_leaf: Optional[Callable[[Any], bool]] = None) -> _T: ...
@overload
def combine(
    *pytrees: PyTree, is_leaf: Optional[Callable[[Any], bool]] = None
) -> PyTree: ...
def combine(
    *pytrees: PyTree, is_leaf: Optional[Callable[[Any], bool]] = None
) -> PyTree:
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
    - `is_leaf`: As [`equinox.partition`][].

    **Returns:**

    A PyTree with the same structure as its inputs. Each leaf will be the first
    non-`None` leaf found in the corresponding leaves of `pytrees` as they are
    iterated over.
    """
    if is_leaf is None:
        _is_leaf = _is_none
    else:
        _is_leaf = lambda x: _is_none(x) or is_leaf(x)

    return jtu.tree_map(_combine, *pytrees, is_leaf=_is_leaf)
