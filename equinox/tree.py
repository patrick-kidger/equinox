from typing import Any, Callable, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from .custom_types import PyTree


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
