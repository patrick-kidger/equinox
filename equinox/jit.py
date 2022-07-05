import functools as ft
import inspect
import warnings
from types import FunctionType
from typing import Any, Callable, Sequence

import jax

from .compile_utils import (
    compile_cache,
    get_fun_names,
    hashable_combine,
    hashable_partition,
    Static,
)
from .custom_types import BoolAxisSpec, PyTree, sentinel, TreeDef
from .doc_utils import doc_strip_annotations
from .filters import combine, is_array, partition
from .module import Module, module_update_wrapper


@compile_cache
def _filter_jit_cache(fun_names, **jitkwargs):
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

    fun_name, fun_qualname = fun_names
    fun_wrapped.__name__ = fun_name
    fun_wrapped.__qualname__ = fun_qualname

    return jax.jit(fun_wrapped, static_argnums=1, **jitkwargs)


class _JitWrapper(Module):
    _new_style: bool
    _signature: inspect.Signature
    _dynamic_fun: PyTree[Any]
    _static_fun_treedef: TreeDef
    _static_fun_leaves: Sequence[Any]
    _filter_default: BoolAxisSpec
    _filter_spec: PyTree[BoolAxisSpec]
    _filter_out: PyTree[Any]
    _cached: FunctionType

    def _fun_wrapper(self, is_lower, args, kwargs):
        if self._new_style:
            bound = self._signature.bind(*args, **kwargs)
            bound.apply_defaults()
            args = bound.args
            kwargs = bound.kwargs

            filter_args, filter_kwargs = self._filter_spec
            filter_args = filter_args + (self._filter_default,) * (
                len(args) - len(filter_args)
            )
            filter_kwargs = {
                key: filter_kwargs.get(key, self._filter_default) for key in kwargs
            }
            filter_spec = (filter_args, filter_kwargs)
        else:
            filter_spec = self._filter_spec
        dynamic_spec, static_spec_leaves, static_spec_treedef = hashable_partition(
            (args, kwargs), filter_spec
        )
        dynamic = (self._dynamic_fun, dynamic_spec)
        static = (
            self._static_fun_treedef,
            self._static_fun_leaves,
            static_spec_treedef,
            static_spec_leaves,
            self._filter_out,
        )
        if is_lower:
            return self._cached.lower(dynamic, static)
        else:
            dynamic_out, static_out = self._cached(dynamic, static)
            return combine(dynamic_out, static_out.value)

    def __call__(__self, *args, **kwargs):
        return __self._fun_wrapper(False, args, kwargs)

    def lower(__self, *args, **kwargs):
        return __self._fun_wrapper(True, args, kwargs)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return jax.tree_util.Partial(self, instance)


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

    !!! info

        By default, all JAX arrays are traced, and all other types are held static.

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
                p
                if p.kind
                in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                else p.replace(default=filter_default)
                for p in signature.parameters.values()
            ]
        )
        filter_bound = signature_default.bind_partial(*filter_args, **filter_kwargs)
        filter_bound.apply_defaults()
        filter_spec = (filter_bound.args, filter_bound.kwargs)
        new_style = True
    # ~Backward compatibility

    dynamic_fun, static_fun_leaves, static_fun_treedef = hashable_partition(
        fun, filter_fn
    )
    cached = _filter_jit_cache(get_fun_names(fun), **jitkwargs)

    jit_wrapper = _JitWrapper(
        _new_style=new_style,
        _signature=signature,
        _dynamic_fun=dynamic_fun,
        _static_fun_treedef=static_fun_treedef,
        _static_fun_leaves=static_fun_leaves,
        _filter_default=filter_default,
        _filter_spec=filter_spec,
        _filter_out=filter_out,
        _cached=cached,
    )
    return module_update_wrapper(jit_wrapper, fun)
