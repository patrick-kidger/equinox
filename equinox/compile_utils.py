import functools as ft
import weakref
from typing import Any

import jax.tree_util as jtu

from .filters import combine, partition
from .module import Module, static_field


def hashable_partition(pytree, filter_spec):
    dynamic, static = partition(pytree, filter_spec)
    static_leaves, static_treedef = jtu.tree_flatten(static)
    static_leaves = tuple(static_leaves)
    return dynamic, static_leaves, static_treedef


def hashable_combine(dynamic, static_leaves, static_treedef):
    static = jtu.tree_unflatten(static_treedef, static_leaves)
    return combine(dynamic, static)


class Static(Module):
    value: Any = static_field()


def _strip_wrapped_partial(fun):
    if hasattr(fun, "__wrapped__"):  # ft.wraps
        return _strip_wrapped_partial(fun.__wrapped__)
    if isinstance(fun, ft.partial):
        return _strip_wrapped_partial(fun.func)
    return fun


def _get_fun_names(fun):
    fun = _strip_wrapped_partial(fun)
    try:
        return fun.__name__, fun.__qualname__
    except AttributeError:
        return type(fun).__name__, type(fun).__qualname__


class _WeakRefAble:
    __slots__ = ("__weakref__",)


_fallback_weakrefable = _WeakRefAble()


def compile_cache(fun):
    fun_cache = weakref.WeakKeyDictionary()

    @ft.wraps(fun)
    def _fun(wrapped_fun, *args, **kwargs):
        wrapped_fun_name = _get_fun_names(wrapped_fun)
        try:
            weakref.ref(wrapped_fun)
        except TypeError:
            wrapped_fun = _fallback_weakrefable
        try:
            arg_cache = fun_cache[wrapped_fun]
        except KeyError:
            arg_cache = fun_cache[wrapped_fun] = {}
        leaves, treedef = jtu.tree_flatten((args, kwargs))
        leaves = tuple(leaves)
        try:
            result = arg_cache[leaves, treedef]
        except KeyError:
            result = arg_cache[leaves, treedef] = fun(wrapped_fun_name, *args, **kwargs)
        return result

    return _fun
