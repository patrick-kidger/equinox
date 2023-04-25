import functools as ft
import types
from typing import Callable

import jax
import jax.tree_util as jtu
from jaxtyping import PyTree

from ._module import Module


def hashable_filter(pytree: PyTree, filter_fn: Callable):
    leaves = jtu.tree_leaves(pytree)
    dynamic_leaves = tuple(x if filter_fn(x) else None for x in leaves)
    return dynamic_leaves


def hashable_partition(pytree: PyTree, filter_fn: Callable):
    leaves, treedef = jtu.tree_flatten(pytree)
    dynamic_leaves = tuple(x if filter_fn(x) else None for x in leaves)
    static_leaves = tuple(None if filter_fn(x) else x for x in leaves)
    return dynamic_leaves, (static_leaves, treedef)


def hashable_combine(dynamic_leaves, static) -> PyTree:
    static_leaves, treedef = static
    leaves = [d if s is None else s for d, s in zip(dynamic_leaves, static_leaves)]
    return jtu.tree_unflatten(treedef, leaves)


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
        args, kwargs = jtu.tree_unflatten(treedef, leaves)
        return fun(*args, **kwargs)

    @ft.wraps(fun)
    def _fun(*args, **kwargs):
        leaves, treedef = jtu.tree_flatten((args, kwargs))
        leaves = tuple(leaves)
        return _cache(leaves, treedef)

    return _fun


class Lowered(Module):
    lowered: jax.stages.Lowered
    info: PyTree
    preprocess: types.FunctionType
    postprocess: types.FunctionType

    def as_text(self):
        return self.lowered.as_text()

    def compile(self):
        return Compiled(
            self.lowered.compile(),
            self.info,
            self.preprocess,  # pyright: ignore
            self.postprocess,  # pyright: ignore
        )


class Compiled(Module):
    compiled: jax.stages.Compiled
    info: PyTree
    preprocess: types.FunctionType
    postprocess: types.FunctionType

    def __call__(self, *args, **kwargs):
        dynamic = self.preprocess(self.info, args, kwargs)
        out = self.compiled(dynamic)
        return self.postprocess(out)
