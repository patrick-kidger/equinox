import functools as ft
import inspect
import warnings
from typing import Callable

import jax

from .compile_utils import (
    hashable_combine,
    hashable_partition,
    Static,
    strip_wrapped_partial,
)
from .custom_types import BoolAxisSpec, PyTree, sentinel
from .doc_utils import doc_fn, doc_strip_annotations
from .filters import combine, is_array, partition


@ft.lru_cache(maxsize=None)
def _filter_jit_cache(unwrapped_fun_treedef, unwrapped_fun_leaves, **jitkwargs):
    unwrapped_fun = jax.tree_unflatten(unwrapped_fun_treedef, unwrapped_fun_leaves)

    @ft.partial(jax.jit, static_argnums=1, **jitkwargs)
    @ft.wraps(unwrapped_fun)
    def fun_wrapped(dynamic, static):
        dynamic_fun, dynamic_spec = dynamic
        (
            static_fun_treedef,
            static_fun_leaves,
            static_spec_treedef,
            static_spec_leaves,
            filter_out,
        ) = static
        fun = hashable_combine(dynamic_fun, static_fun_leaves, static_fun_treedef)
        args, kwargs = hashable_combine(
            dynamic_spec, static_spec_leaves, static_spec_treedef
        )
        out = fun(*args, **kwargs)
        dynamic_out, static_out = partition(out, filter_out)
        return dynamic_out, Static(static_out)

    return fun_wrapped


@doc_strip_annotations
def filter_jit(
    fun: Callable = sentinel,
    *,
    default: BoolAxisSpec = doc_fn(is_array),
    fn: PyTree[BoolAxisSpec] = doc_fn(is_array),
    args: PyTree[BoolAxisSpec] = (),
    kwargs: PyTree[BoolAxisSpec] = None,
    out: PyTree[BoolAxisSpec] = doc_fn(is_array),
    **jitkwargs
) -> Callable:
    """Wraps together [`equinox.partition`][] and `jax.jit`.

    **Arguments:**

    - `fun` is a pure function to JIT compile.
    - `default` should be function `Leaf -> bool` that will be called on every leaf of
        every input to the function. Truthy values will be traced; falsey values will
        be held static.
    - `args` and `kwargs` are optional per-argument and per-keyword-argument overrides
        for `default`. These should be PyTrees whose structures are a prefix of the
        inputs to `fun`. It behaves as the `filter_spec` argument to
        [`equinox.filter`][]; truthy values will be traced; falsey values will be held
        static.
    - `out` is a PyTree whose structure should be a prefix of the
        structure of the outputs of `fun`. It behaves as the `filter_spec` argument to
        [`equinox.filter`][]. Truthy values should be tracers; falsey values are any
        (non-tracer) auxiliary information to return.
    - `fn` is a PyTree whose structure should be a prefix of the function `fun` itself.
        It behaves as the `filter_spec` argument to [`equinox.filter`][]. Truthy values
        will be traced; falsey values will be held static. (So that `fun` may be any
        callable -- such as a bound method, or a class implementing `__call__` -- and
        not necessarily just a normal Python function.)
    - `**jitkwargs` are any other keyword arguments to `jax.jit`.

    **Returns:**

    The JIT'd version of `fun`.

    !!! info

        The most common case is to trace all JAX arrays and treat all other objects as
        static. This is accomplished with `equinox.is_array`, which is the default.
        (It is relatively unusual to need different behaviour to this.)

    !!! example

        ```python
        @eqx.filter_jit
        def f(x, y):  # both args traced if arrays, static if non-arrays
            return x + y

        @eqx.filter_jit(kwargs=dict(x=False))
        def g(x, y):  # x held static; y is traced if array, static if non-array
            return x + y

        @eqx.filter_jit(args=(True,))
        def h(x):
            return x

        f(jnp.array(1), jnp.array(2))  # both args traced
        f(jnp.array(1), 2)  # first arg traced, second arg static
        f(1, 2)  # both args static

        g(1, jnp.array(2))  # first arg static, second arg traced
        g(1, 2)  # both args static

        h(1)  # traced
        h(jnp.array(1))  # traced
        h("hi")  # not a trace-able JAX type, so error
        ```
    """

    if fun is sentinel:
        return ft.partial(
            filter_jit,
            default=default,
            fn=fn,
            args=args,
            kwargs=kwargs,
            out=out,
            **jitkwargs
        )

    if any(
        x in jitkwargs for x in ("static_argnums", "static_argnames", "donate_argnums")
    ):
        raise ValueError(
            "`jitkwargs` cannot contain 'static_argnums', 'static_argnames' or "
            "'donate_argnums'."
        )

    if kwargs is None:
        kwargs = {}

    # Original names are to provide a nice API, but they're too ambiguous for the code.
    filter_default = default
    filter_fn = fn
    filter_args = args
    filter_kwargs = kwargs
    filter_out = out
    del default, fn, args, kwargs, out

    signature = inspect.signature(fun)

    # Backward compatibility
    filter_spec = jitkwargs.get("filter_spec", is_array)
    filter_spec_return = jitkwargs.get("filter_spec_return", is_array)
    filter_spec_fun = jitkwargs.get("filter_spec_fun", is_array)
    if any(
        x is not is_array for x in (filter_spec, filter_spec_return, filter_spec_fun)
    ):
        # Old API

        warnings.warn(
            "`filter_spec` is deprecated in favour of the new `args`, `kwargs` interface."
        )
        if (
            any(x is not is_array for x in (filter_default, filter_fn, filter_out))
            or filter_args != ()
            or filter_kwargs != {}
        ):
            raise ValueError(
                "Cannot use deprecated `filter_spec` at the same time as the new `args`, `kwargs` interface."
            )

        # We choose not to make a distinction between ([arg, ..., arg], kwargs) and ((arg, ..., arg), kwargs)
        if (
            isinstance(filter_spec, tuple)
            and len(filter_spec) == 2
            and isinstance(filter_spec[0], list)
        ):
            filter_spec = (tuple(filter_spec[0]), filter_spec[1])

        filter_fn = filter_spec_fun
        filter_out = filter_spec_return
        new_style = False
    else:
        # New API

        signature_default = signature.replace(
            parameters=[
                p.replace(default=filter_default) for p in signature.parameters.values()
            ]
        )
        filter_bound = signature_default.bind_partial(*filter_args, **filter_kwargs)
        filter_bound.apply_defaults()
        filter_spec = (filter_bound.args, filter_bound.kwargs)
        new_style = True
    # ~Backward compatibility

    _, unwrapped_fun_leaves, unwrapped_fun_treedef = hashable_partition(
        strip_wrapped_partial(fun), filter_fn
    )
    dynamic_fun, static_fun_leaves, static_fun_treedef = hashable_partition(
        fun, filter_fn
    )
    cached = _filter_jit_cache(unwrapped_fun_treedef, unwrapped_fun_leaves, **jitkwargs)

    def _fun_wrapper(is_lower, args, kwargs):
        if new_style:
            bound = signature.bind(*args, **kwargs)
            args = bound.args
            kwargs = bound.kwargs
        dynamic_spec, static_spec_leaves, static_spec_treedef = hashable_partition(
            (args, kwargs), filter_spec
        )
        dynamic = (dynamic_fun, dynamic_spec)
        static = (
            static_fun_treedef,
            static_fun_leaves,
            static_spec_treedef,
            static_spec_leaves,
            filter_out,
        )
        if is_lower:
            return cached.lower(dynamic, static)
        else:
            dynamic_out, static_out = cached(dynamic, static)
            return combine(dynamic_out, static_out.value)

    @ft.wraps(fun)
    def fun_wrapper(*args, **kwargs):
        return _fun_wrapper(False, args, kwargs)

    def lower(*args, **kwargs):
        return _fun_wrapper(True, args, kwargs)

    fun_wrapper.lower = lower

    return fun_wrapper
