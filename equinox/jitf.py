from dataclasses import dataclass
import functools as ft
import jax
from typing import Any

from .filters import validate_filters


@ft.lru_cache(maxsize=4096)
def _jitf_cache(f, args_treedef, **jitkwargs):
    @ft.partial(jax.jit, **jitkwargs)
    def f_wrapped(*args):
        args = jax.tree_unflatten(args_treedef, args)
        return f(*args)

    return f_wrapped


@dataclass(frozen=True)
class _UnPyTreeAble:
    value: Any

    def __bool__(self):
        return False


_marker_sentinel = object()


def jitf(
    fun, *, filter_fn=None, filter_tree=None, static_argnums=None, static_argnames=None, donate_argnums=(), **jitkwargs
):
    """
    A jax.jit that automatically sets whether arguments are static or not, according to either `filter_fn` or
    `filter_tree`. The `static_argnums` argument can still be used to additionally specify any extra static arguments.

    The above applies recursively inside PyTrees, e.g. some parts of the PyTree can be static and some can be traced.
    """
    if isinstance(static_argnums, int):
        static_argnums = (static_argnums,)
    if static_argnames is not None:
        raise NotImplementedError("jitf does not yet support `static_argnames`. use static_argnums instead.")
    if donate_argnums != ():
        raise NotImplementedError("jitf does not ye support `donate_argnums`.")
    validate_filters("jitf", filter_fn, filter_tree)

    @ft.wraps(fun)
    def f_wrapper(*args, **kwargs):
        if len(kwargs):
            raise NotImplementedError("jitf does not yet support keyword arguments. Use positional arguments instead.")

        if len(args) - len(static_argnums) == 1 and filter_tree is not None:
            new_filter_tree = (filter_tree,)
        else:
            new_filter_tree = filter_tree

        # Mark the arguments that have been explicitly declared static via `static_argnums`
        if static_argnums is not None:
            args = list(args)
            for index in static_argnums:
                args[index] = _UnPyTreeAble(args[index])
            if filter_tree is not None:
                new_filter_tree = list(new_filter_tree)
                for index in static_argnums:
                    new_filter_tree.insert(index, _UnPyTreeAble(None))

        # Flatten everything else
        args_flat, args_treedef = jax.tree_flatten(args)
        if filter_tree is not None:
            filter_flat, flat_treedef = jax.tree_flatten(new_filter_tree)
            if flat_treedef != args_treedef:
                raise ValueError("The tree stucture for the filters and the arguments must be the same.")

        # Figure out static argnums with respect to this new flattened structure.
        new_static_argnums = []
        if filter_tree is None:
            # implies filter_fn is not None
            for i, arg in enumerate(args_flat):
                if isinstance(arg, _UnPyTreeAble) or not filter_fn(arg):
                    new_static_argnums.append(i)
        else:
            for i, (arg, filter) in enumerate(zip(args_flat, filter_flat)):
                if not filter:
                    new_static_argnums.append(i)
        new_static_argnums = tuple(new_static_argnums)
        if static_argnums is not None:
            args_flat = [arg.value if isinstance(arg, _UnPyTreeAble) else arg]

        f_jitted = _jitf_cache(fun, args_treedef, static_argnums=new_static_argnums, **jitkwargs)
        return f_jitted(*args_flat)

    return f_wrapper
