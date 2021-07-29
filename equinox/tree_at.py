import jax
from typing import Any, Callable, Optional, Tuple, Union

from .custom_types import PyTree


_sentinel = object()


def tree_at(
    pytree: PyTree,
    where: Callable[[PyTree], Union[Any, Tuple[Any]]],
    replace: Optional[Union[Any, Tuple[Any]]] = _sentinel,
    replace_fn: Optional[Callable[[Any], Any]] = _sentinel
) -> PyTree:

    if (replace is _sentinel and replace_fn is _sentinel) or (replace is not _sentinel and replace_fn is not _sentinel):
        raise ValueError("Precisely one of `replace` and `replace_fn` must be specified.")
    elif replace is _sentinel:
        replace_passed = False
        replacer = lambda j, i: replace_fn(flat[i])
    else:
        replace_passed = True
        replacer = lambda j, i: replace[j]

    # TODO: is there a neater way of accomplishing this?
    flat, treedef = jax.tree_flatten(pytree)
    flat_indices = list(range(len(flat)))
    index_pytree = jax.tree_unflatten(treedef, flat_indices)
    index = where(index_pytree)
    # where can return either a single entry, or a tuple.
    if isinstance(index, int):
        index = (index,)
        replace = (replace,)

    if replace_passed and len(index) != len(replace):
        raise ValueError("`where` must return a tuple of the same length as `replace`.")
    for j, i in enumerate(index):
        flat[i] = replacer(j, i)

    return jax.tree_unflatten(treedef, flat)
