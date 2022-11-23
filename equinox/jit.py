import functools as ft
import inspect
import warnings
from types import FunctionType
from typing import Any, Callable, Sequence, Tuple

import jax
import jax.tree_util as jtu
from jaxtyping import PyTree

from .compile_utils import (
    compile_cache,
    get_fun_names,
    hashable_combine,
    hashable_partition,
)
from .custom_types import BoolAxisSpec, sentinel, TreeDef
from .doc_utils import doc_strip_annotations
from .filters import combine, is_array, partition
from .module import Module, module_update_wrapper, Static


@compile_cache
def _filter_jit_cache(fun_names, **jitkwargs):
    def fun_wrapped(dynamic, static, donate):
        dynamic_fun, dynamic_spec = dynamic
        (static_fun, static_spec, filter_out) = static
        donate_fun, donate_spec = donate

        donate_or_dynamic_fun = combine(donate_fun, dynamic_fun)
        fun = hashable_combine(donate_or_dynamic_fun, static_fun)

        donate_or_dynamic_spec = combine(donate_spec, dynamic_spec)
        args, kwargs = hashable_combine(donate_or_dynamic_spec, static_spec)
        out = fun(*args, **kwargs)
        dynamic_out, static_out = partition(out, filter_out)
        return dynamic_out, Static(static_out)

    fun_name, fun_qualname = fun_names
    fun_wrapped.__name__ = fun_name
    fun_wrapped.__qualname__ = fun_qualname

    return jax.jit(fun_wrapped, static_argnums=1, donate_argnums=2, **jitkwargs)


class _JitWrapper(Module):
    _new_style: bool
    _signature: inspect.Signature
    _dynamic_fun: PyTree[Any]
    _static_fun: Tuple[TreeDef, Sequence[Any]]
    _donate_fun: PyTree[Any]
    _filter_default: BoolAxisSpec
    _filter_spec: PyTree[BoolAxisSpec]
    _donate_default: BoolAxisSpec
    _donate_spec: PyTree[BoolAxisSpec]
    _filter_out: PyTree[Any]
    _cached: FunctionType

    def _fun_wrapper(self, is_lower, args, kwargs):
        if self._new_style:
            bound = self._signature.bind(*args, **kwargs)
            bound.apply_defaults()
            args = bound.args
            kwargs = bound.kwargs

            def _merge_spec(p_spec, p_default):

                p_args, p_kwargs = p_spec
                p_args = p_args + (p_default,) * (len(args) - len(p_args))
                p_kwargs = {key: p_kwargs.get(key, p_default) for key in kwargs}
                p_spec = (p_args, p_kwargs)
                return p_spec

            filter_spec = _merge_spec(self._filter_spec, self._filter_default)
            _donate_spec = _merge_spec(self._donate_spec, self._donate_default)

        else:
            filter_spec = self._filter_spec
            _donate_spec = self._donate_spec
        donate_or_dynamic_spec, static_spec = hashable_partition(
            (args, kwargs), filter_spec
        )
        donate_spec, dynamic_spec = partition(donate_or_dynamic_spec, _donate_spec)
        dynamic = (self._dynamic_fun, dynamic_spec)
        static = (
            self._static_fun,
            static_spec,
            self._filter_out,
        )
        donate = (self._donate_fun, donate_spec)
        if is_lower:
            return self._cached.lower(dynamic, static, donate)
        else:
            dynamic_out, static_out = self._cached(dynamic, static, donate)
            return combine(dynamic_out, static_out.value)

    def __call__(__self, *args, **kwargs):
        return __self._fun_wrapper(False, args, kwargs)

    def lower(__self, *args, **kwargs):
        return __self._fun_wrapper(True, args, kwargs)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return jtu.Partial(self, instance)


@doc_strip_annotations
def filter_jit(
    fun: Callable = sentinel,
    *,
    default: BoolAxisSpec = is_array,
    fn: PyTree[BoolAxisSpec] = is_array,
    args: PyTree[BoolAxisSpec] = (),
    kwargs: PyTree[BoolAxisSpec] = None,
    out: PyTree[BoolAxisSpec] = is_array,
    donate_default: BoolAxisSpec = False,
    donate_args: PyTree[BoolAxisSpec] = (),
    donate_kwargs: PyTree[BoolAxisSpec] = None,
    donate_fn: PyTree[BoolAxisSpec] = False,
    **jitkwargs
) -> Callable:
    """Wraps together [`equinox.partition`][] and `jax.jit`.

    !!! info

        By default, all JAX arrays are traced, all other types are held static, no
        buffers are donated.

    **Arguments:**

    In each of the following cases, except those arguments prefixed with `donate_`,
    `True` indicates that an argument should be traced, `False` indicates that an
    argument should be held static, and functions `Leaf -> bool` are mapped and
    evaluated on every leaf of their subtree.
    In these `donate_*`, `True` indicates that an argument that has been marked as
    traced should donate buffer, `False` does nothing. and functions `Leaf -> bool` are
    mapped and evaluated on every leaf of their subtree.

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
    - `donate_default` is similar to `default` , except that only works with parameters
        marked as traced.
    - `donate_args` is similar to `args`, except that only works with parameters marked
        as traced.
    - `donate_kwargs` is similar to `kwargs`, except that only works with parameters
        marked as traced.
    - `donate_fn` is similar to `fn`, except that only works with parameters marked as
        traced.
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

        @eqx.filter_jit(donate_args=(True,))
        def h_inplace(x):
            return x+1

        f(jnp.array(1), jnp.array(2))  # both args traced
        f(jnp.array(1), 2)  # first arg traced, second arg static
        f(1, 2)  # both args static

        g(1, jnp.array(2))  # first arg static, second arg traced
        g(1, 2)  # both args static

        h(1)  # traced
        h(jnp.array(1))  # traced
        h("hi")  # not a trace-able JAX type, so error

        apply(lambda x: x + 1, jnp.array(1))  # first arg static, second arg traced.

        x = jnp.array(2)
        new_x = h_inplace(x) # first arg donated
        x.is_deleted() # True
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
            donate_default=donate_default,
            donate_args=donate_args,
            donate_kwargs=donate_kwargs,
            donate_fn=donate_fn,
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

    if donate_kwargs is None:
        donate_kwargs = {}

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
            or donate_default is not False
            or donate_args != ()
            or donate_kwargs != {}
            or donate_fn is not False
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
        donate_spec = (False, False)
        new_style = False
    else:
        # New API

        def _get_spec(sig, p_default, p_args, p_kwargs):
            signature_default = sig.replace(
                parameters=[
                    p
                    if p.kind
                    in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                    else p.replace(default=p_default)
                    for p in signature.parameters.values()
                ]
            )
            p_bound = signature_default.bind_partial(*p_args, **p_kwargs)
            p_bound.apply_defaults()
            p_spec = (p_bound.args, p_bound.kwargs)
            return p_spec

        filter_spec = _get_spec(signature, filter_default, filter_args, filter_kwargs)
        donate_spec = _get_spec(signature, donate_default, donate_args, donate_kwargs)

        new_style = True
    # ~Backward compatibility

    donate_or_dynamic_fun, static_fun = hashable_partition(fun, filter_fn)
    donate_fun, dynamic_fun = partition(donate_or_dynamic_fun, donate_fn)
    cached = _filter_jit_cache(get_fun_names(fun), **jitkwargs)

    jit_wrapper = _JitWrapper(
        _new_style=new_style,
        _signature=signature,
        _dynamic_fun=dynamic_fun,
        _static_fun=static_fun,
        _donate_fun=donate_fun,
        _filter_default=filter_default,
        _filter_spec=filter_spec,
        _filter_out=filter_out,
        _donate_default=donate_default,
        _donate_spec=donate_spec,
        _cached=cached,
    )
    return module_update_wrapper(jit_wrapper, fun)
