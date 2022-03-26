import functools as ft
from typing import Any

import jax

from .filters import combine, is_array, partition
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
