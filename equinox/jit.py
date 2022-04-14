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
from .doc_utils import doc_strip_annotations
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
    default: BoolAxisSpec = is_array,
    fn: PyTree[BoolAxisSpec] = is_array,
    args: PyTree[BoolAxisSpec] = (),
    kwargs: PyTree[BoolAxisSpec] = None,
    out: PyTree[BoolAxisSpec] = is_array,
    **jitkwargs
) -> Callable:
    """Wraps together [`equinox.partition`][] and `jax.jit`.

    **Arguments:**

    In each of the following cases, `True` indicates that an argument should be traced,
    `False` indicates that an argument should be held static, and functions
    `Leaf -> bool` are mapped and evaluated on every leaf of their subtree.

    - `fun` is a pure function to JIT compile.
    - `default` should be a `bool` or a function `Leaf -> bool`, and is applied by
        default to every argument and keyword argument to `fun`.
    - `args` is an optional per-argument override for `default`, and should be a tuple
        of PyTrees with leaves that are either `bool`s or functions `Leaf -> bool`.
        The PyTree structures should be prefixes of the corresponding input to `fun`.
    - `kwargs` is an optional per-keyword-argument override for `default` and should be
        a dictionary, whose keys are the names of arguments to `fun`, and whose values
        are PyTrees with leaves that either `bool`s or functions `Leaf -> bool`. The
        PyTree structures should be prefixes of the corresponding input to `fun`.
    - `out` should be a PyTree with leaves that either `bool`s or functions
        `Leaf -> bool`. The PyTree structure should be a prefix of the output of `fun`.
        Truthy values should be tracers; falsey values are any (non-tracer) auxiliary
        information to return.
    - `fn` should be a PyTree with leaves that either `bool`s or functions
        `Leaf -> bool`. The PyTree structure should be a prefix of `fun` itself. (Note
        that `fun` may be any callable, e.g. a bound method, or a class implementing
        `__call__`, and doesn't have to be a normal Python function.)
    - `**jitkwargs` are any other keyword arguments to `jax.jit`.

    When `args`, `kwargs`, `out`, `fn` are prefixes of the corresponding input, their
    value will be mapped over the input PyTree.

    **Returns:**

    The JIT'd version of `fun`.

    !!! info

        The most common case is to trace all JAX arrays and treat all other objects as
        static. This is accomplished with [`equinox.is_array`][], which is the default.
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

        @eqx.filter_jit
        def apply(f, x):
            return f(x)

        f(jnp.array(1), jnp.array(2))  # both args traced
        f(jnp.array(1), 2)  # first arg traced, second arg static
        f(1, 2)  # both args static

        g(1, jnp.array(2))  # first arg static, second arg traced
        g(1, 2)  # both args static

        h(1)  # traced
        h(jnp.array(1))  # traced
        h("hi")  # not a trace-able JAX type, so error

        apply(lambda x: x + 1, jnp.array(1))  # first arg static, second arg traced.
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
    filter_spec = jitkwargs.pop("filter_spec", is_array)
    filter_spec_return = jitkwargs.pop("filter_spec_return", is_array)
    filter_spec_fun = jitkwargs.pop("filter_spec_fun", is_array)
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
            bound.apply_defaults()
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
