import functools as ft
from dataclasses import dataclass
from typing import Any

import jax

from .deprecated import deprecated
from .filters import combine, is_array, partition, validate_filters
from .module import Module, static_field


class _Static(Module):
    value: Any = static_field()


@ft.lru_cache(maxsize=4096)
def _filter_jit_cache(f, **jitkwargs):
    @ft.partial(jax.jit, static_argnums=(0, 1, 4), **jitkwargs)
    def f_wrapped(
        static_leaves, static_treedef, dynamic_args, dynamic_kwargs, filter_spec_return
    ):
        static_args, static_kwargs = jax.tree_unflatten(static_treedef, static_leaves)
        args = combine(dynamic_args, static_args)
        kwargs = combine(dynamic_kwargs, static_kwargs)
        out = f(*args, **kwargs)
        dynamic_out, static_out = partition(out, filter_spec_return)
        return dynamic_out, _Static(static_out)

    return f_wrapped


def filter_jit(fun, *, filter_spec=is_array, filter_spec_return=is_array, **jitkwargs):
    """Wraps together [`equinox.partition`][] and `jax.jit`.

    **Arguments:**

    - `fun` is a pure function to JIT compile.
    - `filter_spec` is a PyTree whose structure should be a prefix of the structure of
        the inputs to `fun`. It behaves as the `filter_spec` argument to
        [`equinox.filter`][]. Truthy values will be traced; falsey values will be held
        static.
    - `filter_spec_return` is a PyTree whose structure should be a prefix of the
        structure of the outputs of `fun`. It behaves as the `filter_spec` argument to
        [`equinox.filter`][]. Truthy values should be tracers; falsely values are any
        (non-tracer) auxiliary information to return.
    - `**jitkwargs` are any other keyword arguments to `jax.jit`.

        !!! info

            Specifically, if calling `fun(*args, **kwargs)`, then `filter_spec` must
            have a structure which is a prefix for `(args, kwrgs)`.

    **Returns:**

    The JIT'd version of `fun`.

    !!! info

        A very important special case is to trace all JAX arrays and treat all other
        objects as static.

        This is accomplished with `filter_spec=equinox.is_array`,
        `filter_spec_return=equinox.is_array` -- which are the defaults. (It is
        unusual to need different behaviour to this.)
    """

    if any(
        x in jitkwargs for x in ("static_argnums", "static_argnames", "donate_argnums")
    ):
        raise ValueError(
            "`jitkwargs` cannot contain 'static_argnums', 'static_argnames' or "
            "'donate_argnums'."
        )

    # We choose not to make a distinction between ([arg, ..., arg], kwargs) and ((arg, ..., arg), kwargs)
    if (
        isinstance(filter_spec, tuple)
        and len(filter_spec) == 2
        and isinstance(filter_spec[0], list)
    ):
        filter_spec = (tuple(filter_spec[0]), filter_spec[1])

    @ft.wraps(fun)
    def fun_wrapper(*args, **kwargs):
        (dynamic_args, dynamic_kwargs), (static_args, static_kwargs) = partition(
            (args, kwargs), filter_spec
        )
        static_leaves, static_treedef = jax.tree_flatten((static_args, static_kwargs))
        static_leaves = tuple(static_leaves)
        dynamic_out, static_out = _filter_jit_cache(fun, **jitkwargs)(
            static_leaves,
            static_treedef,
            dynamic_args,
            dynamic_kwargs,
            filter_spec_return,
        )
        return combine(dynamic_out, static_out.value)

    return fun_wrapper


#
# Deprecated
#


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


@deprecated(in_favour_of=filter_jit)
def jitf(
    fun,
    *,
    filter_fn=None,
    filter_tree=None,
    static_argnums=None,
    static_argnames=None,
    donate_argnums=(),
    **jitkwargs
):
    if isinstance(static_argnums, int):
        static_argnums = (static_argnums,)
    if static_argnames is not None:
        raise NotImplementedError(
            "jitf does not yet support `static_argnames`. use static_argnums instead."
        )
    if donate_argnums != ():
        raise NotImplementedError("jitf does not ye support `donate_argnums`.")
    validate_filters("jitf", filter_fn, filter_tree)

    if static_argnums is None:
        len_static_argnums = 0
    else:
        len_static_argnums = len(static_argnums)

    @ft.wraps(fun)
    def f_wrapper(*args, **kwargs):
        if len(kwargs):
            raise NotImplementedError(
                "jitf does not yet support keyword arguments. Use positional arguments instead."
            )

        if filter_tree is not None:
            if len(args) - len_static_argnums == 1:
                new_filter_tree = (filter_tree,)
            else:
                new_filter_tree = tuple(filter_tree)

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
                raise ValueError(
                    "The tree stucture for the filters and the arguments must be the same."
                )

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
            args_flat = [
                arg.value if isinstance(arg, _UnPyTreeAble) else arg
                for arg in args_flat
            ]

        f_jitted = _jitf_cache(
            fun, args_treedef, static_argnums=new_static_argnums, **jitkwargs
        )
        return f_jitted(*args_flat)

    return f_wrapper
