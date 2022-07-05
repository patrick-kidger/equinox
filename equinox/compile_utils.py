import functools as ft
from typing import Any

import jax

from .filters import combine, partition
from .module import Module, static_field


def hashable_partition(pytree, filter_spec):
    dynamic, static = partition(pytree, filter_spec)
    static_leaves, static_treedef = jax.tree_flatten(static)
    static_leaves = tuple(static_leaves)
    return dynamic, static_leaves, static_treedef


def hashable_combine(dynamic, static_leaves, static_treedef):
    static = jax.tree_unflatten(static_treedef, static_leaves)
    return combine(dynamic, static)


class Static(Module):
    value: Any = static_field()


def _strip_wrapped_partial(fun):
    if hasattr(fun, "__wrapped__"):  # ft.wraps
        return _strip_wrapped_partial(fun.__wrapped__)
    if isinstance(fun, ft.partial):
        return _strip_wrapped_partial(fun.func)
    return fun


def get_fun_names(fun):
    fun = _strip_wrapped_partial(fun)
    try:
        return fun.__name__, fun.__qualname__
    except AttributeError:
        return type(fun).__name__, type(fun).__qualname__


def compile_cache(fun):
    @ft.lru_cache(maxsize=None)
    def _cache(leaves, treedef):
        args, kwargs = jax.tree_unflatten(treedef, leaves)
        return fun(*args, **kwargs)

    @ft.wraps(fun)
    def _fun(*args, **kwargs):
        leaves, treedef = jax.tree_flatten((args, kwargs))
        leaves = tuple(leaves)
        return _cache(leaves, treedef)

    return _fun
