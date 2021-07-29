import jax
from typing import Any, Callable, Optional, Tuple, Union

from .custom_types import PyTree


def tree_at(
    pytree: PyTree,
    where: Callable[[PyTree], Union[Any, Tuple[Any]]],
    replace: Optional[Union[Any, Tuple[Any]]] = None,
    replace_fn: Optional[Callable[[Any], Any]] = None
) -> PyTree:

    if replace is None and replace_fn is None:
        raise ValueError("At least one of `replace` and `replace_fn` must be specified.")
    elif replace is None:
        replacer = lambda i, r: replace_fn(r)
    else:
        # replace_fn is None
        replacer = lambda i, r: replace[i]

    # TODO: is there a neater way of accomplishing this?
    flat, treedef = jax.tree_flatten(pytree)
    flat_indices = list(range(len(flat)))
    index_pytree = jax.tree_unflatten(treedef, flat_indices)
    index = where(index_pytree)

    if isinstance(index, int):
        index = (index,)
        replace = (replace,)
    if len(index) != len(replace):
        raise ValueError("`where` must return a tuple of the same length as `replace`.")

    for i, r in zip(index, replace):
        flat[i] = replacer(i, r)

    return jax.tree_unflatten(treedef, flat)
