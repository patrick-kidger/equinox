import functools as ft
import inspect
import pprint
from typing import Any, Callable, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from .custom_types import PyTree
from .filters import is_array


_sentinel = object()

_Leaf = Any


def tree_at(
    where: Callable[[PyTree], Union[_Leaf, Sequence[_Leaf]]],
    pytree: PyTree,
    replace: Union[_Leaf, Sequence[_Leaf]] = _sentinel,
    replace_fn: Callable[[_Leaf], _Leaf] = _sentinel,
    is_leaf: Callable[[_Leaf], bool] = None,
) -> PyTree:
    """Updates a PyTree out-of-place; a bit like using `.at[].set()` on a JAX array.

    **Arguments:**

    - `where`: A callable `PyTree -> Leaf` or `PyTree -> Sequence[Leaf]`. It should
        consume a PyTree with the same structure as `pytree`, and return the leaf or
        leaves that should be replaced. For example
        `where = lambda mlp: mlp.layers[-1].linear.weight`.
    - `pytree`: The PyTree to modify.
    - `replace`: Either a single element, or a sequence of the same length as returned
        by `where`. This specifies the replacements to make at the locations specified
        by `where`. Mutually exclusive with `replace_fn`.
    - `replace_fn`: A function `Leaf -> Any`. It will be called on every leaf specified
        by `where`. The return value from `replace_fn` will be used in its place.
        Mutually exclusive with `replace`.
    - `is_leaf`: As `jax.tree_flatten`; used to determine what should be treated as a
        leaf.

    **Returns:**

    A copy of the input PyTree, with the appropriate modifications.

    !!! example

        This can be used to help specify the weights of a model to train or not to
        train:

        ```python
        model = ...
        trainable = jax.tree_map(lambda _: False, model)
        trainable = equinox.tree_at(lambda mlp: mlp.layers[-1].linear.weight, model, replace=True)
        equinox.filter_grad(..., filter_spec=trainable)
        ```

    !!! example

        Sub-PyTrees can be replaced by flattening them to leaves first:

        ```python
        equinox.tree_at(lambda t: jax.tree_leaves(t.subtree), pytree,
                        jax.tree_leaves(new_subtree))
        ```
    """

    if (replace is _sentinel and replace_fn is _sentinel) or (
        replace is not _sentinel and replace_fn is not _sentinel
    ):
        raise ValueError(
            "Precisely one of `replace` and `replace_fn` must be specified."
        )
    elif replace is _sentinel:
        replace_passed = False
        replacer = lambda j, i: replace_fn(flat[i])
    else:
        replace_passed = True
        replacer = lambda j, i: replace[j]

    # TODO: is there a neater way of accomplishing this?
    flat, treedef = jax.tree_flatten(pytree, is_leaf=is_leaf)
    flat_indices = list(range(len(flat)))
    index_pytree = jax.tree_unflatten(treedef, flat_indices)
    index = where(index_pytree)
    # where can return either a single entry, or a sequence
    if isinstance(index, int):
        index = (index,)
        replace = (replace,)
    elif isinstance(index, Sequence):
        for i in index:
            if not isinstance(i, int):
                raise ValueError(
                    r"""`where` must return a sequence of only leaves; not some subtree.

                    If you want to replace all of a subtree, you can do so by replacing
                    >>> eqx.tree_at(lambda t: t.subtree, tree, new_subtree)  # buggy
                    with
                    >>> eqx.tree_at(lambda t: jax.tree_leaves(t.subtree), tree, 
                    ...             jax.tree_leaves(new_subtree))  # fixed
                    """
                )

    if replace_passed and len(index) != len(replace):
        raise ValueError(
            "`where` must return a sequence of leaves of the same length as `replace`."
        )
    for j, i in enumerate(index):
        flat[i] = replacer(j, i)

    return jax.tree_unflatten(treedef, flat)


def tree_equal(*pytrees: PyTree) -> bool:
    """Returns `True` if all input PyTrees are equal. Every PyTree must have the same
    structure. Any JAX or NumPy arrays (as leaves) must have the same shape, dtype, and
    values to be considered equal. JAX arrays and NumPy arrays are not considered equal
    to each other.

    **Arguments:**

    - `*pytrees`: Any number of PyTrees each with any structure.

    **Returns:**

    A boolean.
    """
    flat, treedef = jax.tree_flatten(pytrees[0])
    array_types = (jnp.ndarray, np.ndarray)
    for pytree in pytrees[1:]:
        flat_, treedef_ = jax.tree_flatten(pytree)
        if treedef_ != treedef:
            return False
        for elem, elem_ in zip(flat, flat_):
            if isinstance(elem, array_types):
                if isinstance(elem_, array_types):
                    if (
                        (type(elem) != type(elem_))
                        or (elem.shape != elem_.shape)
                        or (elem.dtype != elem_.dtype)
                        or (elem != elem_).any()
                    ):
                        return False
                else:
                    return False
            else:
                if isinstance(elem_, array_types):
                    return False
                else:
                    if elem != elem_:
                        return False
    return True


# From:
# https://github.com/google/jax/blob/ee6749608a1588f1f458e0e8ad8c9ecc8942aa83/jax/core.py#L1080  # noqa: E501
def _short_dtype_name(dtype):
    return (
        dtype.name.replace("float", "f")
        .replace("uint", "u")
        .replace("int", "i")
        .replace("complex", "c")
    )


class _WithRepr:
    def __init__(self, val):
        self.val = val

    def __repr__(self):
        return self.val


def _convert(leaf):
    wrapped = False
    wrapper_types = (jax.custom_vjp, jax.custom_jvp, ft.partial)
    while isinstance(leaf, wrapper_types):
        if isinstance(leaf, (jax.custom_jvp, jax.custom_vjp)):
            leaf = leaf.__wrapped__
            # Not always thought of as a wrapper so we don't set wrapped=True
        if isinstance(leaf, ft.partial):
            leaf = leaf.func
            wrapped = True

    if is_array(leaf):
        dt_str = _short_dtype_name(leaf.dtype)
        shape_str = ",".join(map(str, leaf.shape))
        return _WithRepr(f"{dt_str}[{shape_str}]")
    elif inspect.isfunction(leaf):
        if wrapped:
            fn_str = "wrapped function"
        else:
            fn_str = "function"
        return _WithRepr(f"<{fn_str} {leaf.__name__}>")
    else:
        return leaf


def tree_pformat(pytree: PyTree, **kwargs) -> str:
    """Pretty-formats a PyTree as a string, whilst abbreviating JAX arrays.

    All JAX arrays in the PyTree are condensed down to a short string representation
    of their dtype and shape.

    (This is the function used in `__str__` of [`equinox.Module`][].)

    !!! example

        A 32-bit floating-point JAX array of shape `(3, 4)` is printed as `f32[3,4]`.

    **Arguments:**

    - `pytree`: The PyTree to pretty-format.
    - `**kwargs`: Any keyword arguments for `pprint.pformat`.

    **Returns:**

    A string.

    !!! info

        This is best used with Python 3.10 or above, for which the standard library
        `pprint` supports dataclasses and will add line breaks appropriately.
        (`tree_pformat` uses `pprint`; [`equinox.Module`][] uses dataclasses.)
    """
    return pprint.pformat(jax.tree_map(_convert, pytree), **kwargs)
