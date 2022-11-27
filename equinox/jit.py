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


def _hashable_partition(pytree, donate_default):
    donate_or_dynamic_part, static_part = hashable_partition(pytree, is_array)
    donate_part, dynamic_part = partition(donate_or_dynamic_part, donate_default)
    return donate_part, dynamic_part, static_part


def _hashable_combine(donate_part, dynamic_part, static_part):
    donate_or_dynamic_part = combine(donate_part, dynamic_part)
    return hashable_combine(donate_or_dynamic_part, static_part)


@compile_cache
def _filter_jit_cache(fun_names, **jitkwargs):
    def fun_wrapped(dynamic, static, donate):
        dynamic_fun, dynamic_spec = dynamic
        static_fun, static_spec = static
        donate_fun, donate_spec = donate

        fun = _hashable_combine(donate_fun, dynamic_fun, static_fun)

        args, kwargs = _hashable_combine(donate_spec, dynamic_spec, static_spec)
        out = fun(*args, **kwargs)
        dynamic_out, static_out = partition(out, is_array)
        return dynamic_out, Static(static_out)

    fun_name, fun_qualname = fun_names
    fun_wrapped.__name__ = fun_name
    fun_wrapped.__qualname__ = fun_qualname

    return jax.jit(fun_wrapped, static_argnums=1, donate_argnums=2, **jitkwargs)


class _JitWrapper(Module):
    _signature: inspect.Signature
    _dynamic_fun: PyTree[Any]
    _static_fun: Tuple[TreeDef, Sequence[Any]]
    _donate_fun: PyTree[Any]
    _donate_default: BoolAxisSpec
    _cached: FunctionType
    _filter_warning: bool

    def _fun_wrapper(self, is_lower, args, kwargs):
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        args = bound.args
        kwargs = bound.kwargs

        donate_spec, dynamic_spec, static_spec = _hashable_partition(
            (args, kwargs), self._donate_default
        )
        dynamic = (self._dynamic_fun, dynamic_spec)
        static = (self._static_fun, static_spec)
        donate = (self._donate_fun, donate_spec)
        if is_lower:
            return self._cached.lower(dynamic, static, donate)
        else:
            dynamic_out, static_out = self._cached(dynamic, static, donate)
            return combine(dynamic_out, static_out.value)

    def __call__(__self, *args, **kwargs):
        if __self._filter_warning is True:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Some donated buffers were not usable*"
                )
                return __self._fun_wrapper(False, args, kwargs)
        else:
            return __self._fun_wrapper(False, args, kwargs)

    def lower(__self, *args, **kwargs):
        return __self._fun_wrapper(True, args, kwargs)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return jtu.Partial(self, instance)


@doc_strip_annotations
def filter_jit(
    fun: Callable = sentinel, *, donate: str = "arrays", **jitkwargs
) -> Callable:
    """Wraps together [`equinox.partition`][] and `jax.jit`.

    !!! info

        By default, all JAX arrays are traced, all other types are held static, all
        buffers of JAX array are donated.

    **Arguments:**

    - `fun` is a pure function to JIT compile.
    - `donate` indicates whether the buffers of JAX arrays are donated or not, it
        should either be:
        - 'arrays': the default, donate all arrays and suppress all warnings about
            unused buffers;
        - 'warn': as above, but don't suppress unused buffer warnings;
        - 'none': disables buffer donation.
    - `**jitkwargs` are any other keyword arguments to `jax.jit`.

    **Returns:**

    The JIT'd version of `fun`.

    !!! example

        ```python
        @eqx.filter_jit
        def f(x, y):  # both args traced if arrays, static if non-arrays
            return x + y

        f(jnp.array(1), jnp.array(2))  # both args traced
        f(jnp.array(1), 2)  # first arg traced, second arg static
        f(1, 2)  # both args static

        x = jnp.array(2)
        new_x = f(x, 1) #  donated
        x.is_deleted() # True
        ```
    """

    if fun is sentinel:
        return ft.partial(filter_jit, donate=donate, **jitkwargs)

    if any(
        x in jitkwargs for x in ("static_argnums", "static_argnames", "donate_argnums")
    ):
        raise ValueError(
            "`jitkwargs` cannot contain 'static_argnums', 'static_argnames' or "
            "'donate_argnums'."
        )
    signature = inspect.signature(fun)
    donate_default_dict = {"arrays": True, "warn": True, "none": False}

    donate_default = donate_default_dict[donate]
    filter_warning = True if donate == "arrays" else False

    donate_fun, dynamic_fun, static_fun = _hashable_partition(fun, donate_default)
    cached = _filter_jit_cache(get_fun_names(fun), **jitkwargs)

    jit_wrapper = _JitWrapper(
        _signature=signature,
        _dynamic_fun=dynamic_fun,
        _static_fun=static_fun,
        _donate_fun=donate_fun,
        _donate_default=donate_default,
        _cached=cached,
        _filter_warning=filter_warning,
    )
    return module_update_wrapper(jit_wrapper, fun)
