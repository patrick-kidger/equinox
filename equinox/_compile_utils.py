import functools as ft
import types
import warnings
import weakref
from collections.abc import Callable

import jax
import jax.tree_util as jtu
from jaxtyping import PyTree

from ._caches import cache_clears
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


# A sentinel weakref-able value.
def _default_cache_key():
    assert False


def get_fn_names(user_fn):
    user_fn = _strip_wrapped_partial(user_fn)
    try:
        return user_fn.__name__, user_fn.__qualname__
    except AttributeError:
        return type(user_fn).__name__, type(user_fn).__qualname__


def compile_cache(cacheable_fn):
    cache = weakref.WeakKeyDictionary()
    cache_clears.append(cache.clear)

    def cached_fn_impl(leaves, treedef):
        user_fn_names, args, kwargs = jtu.tree_unflatten(treedef, leaves)
        return cacheable_fn(user_fn_names, *args, **kwargs)

    @ft.wraps(cacheable_fn)
    def wrapped_cacheable_fn(user_fn, *args, **kwargs):
        user_fn_names = get_fn_names(user_fn)
        leaves, treedef = jtu.tree_flatten((user_fn_names, args, kwargs))
        leaves = tuple(leaves)

        # Best-effort attempt to clear the cache of old and unused entries.
        if type(user_fn) is types.FunctionType:
            cache_key = user_fn
        else:
            cache_key = _default_cache_key

        try:
            cached_fn = cache[cache_key]
        except KeyError:
            cached_fn = cache[cache_key] = ft.lru_cache(maxsize=None)(cached_fn_impl)
        return cached_fn(leaves, treedef)

    def delete(user_fn):
        user_fn = _strip_wrapped_partial(user_fn)
        if type(user_fn) is types.FunctionType:
            try:
                del cache[user_fn]
            except KeyError:
                warnings.warn(
                    f"Could not delete cache for function {user_fn}. Has it already "
                    "been deleted?"
                )
        else:
            warnings.warn("Could not delete non-function from cache.")

    wrapped_cacheable_fn.delete = delete  # pyright: ignore
    return wrapped_cacheable_fn


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
        out = self.compiled(*dynamic)
        return self.postprocess(out)
