from dataclasses import dataclass
import functools as ft
import jax
from typing import Any


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


_marker_sentinel = object()


def jitf(fun, *, filter_fn, static_argnums=None, static_argnames=None, donate_argnums=(), **jitkwargs):
    """
    A jax.jit that automatically sets whether arguments are static or not, according to `filter_fn`. By default all
    floating-point arrays will be traced, and all other argments will be static. The static_argnums can still be used
    to additionally specify any extra static arguments.

    The above applies recursively inside PyTrees, e.g. some parts of the PyTree can be static and some can be traced.
    """
    if isinstance(static_argnums, int):
        static_argnums = (static_argnums,)
    if static_argnames is not None:
        raise NotImplementedError("static_argnames is not yet supported; use static_argnums instead.")
    if isinstance(donate_argnums, int):
        donate_argnums = (donate_argnums,)

    @ft.wraps(fun)
    def f_wrapper(*args):
        if donate_argnums != ():
            marker_args = list(args)
            for i in donate_argnums:
                marker_args[i] = jax.tree_map(lambda _: _marker_sentinel, marker_args[i])
            marker_args_flat, _ = jax.tree_flatten(marker_args)
            new_donate_argnums = tuple(i for i, arg in enumerate(marker_args_flat) if arg is _marker_sentinel)
        else:
            new_donate_argnums = ()
        if static_argnums is not None:
            args = list(args)
            for index in static_argnums:
                args[index] = _UnPyTreeAble(args[index])
        args_flat, args_treedef = jax.tree_flatten(args)
        new_static_argnums = []
        for i, arg in enumerate(args_flat):
            if not filter_fn(arg):
                new_static_argnums.append(i)
        new_static_argnums = tuple(new_static_argnums)
        if static_argnums is not None:
            args_flat = [arg.value if isinstance(arg, _UnPyTreeAble) else arg]
        f_jitted = _jitf_cache(
            fun, args_treedef, static_argnums=new_static_argnums, donate_argnums=new_donate_argnums, **jitkwargs
        )
        return f_jitted(*args_flat)

    return f_wrapper
