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
from .custom_types import sentinel, TreeDef
from .doc_utils import doc_strip_annotations
from .filters import combine, is_array, partition
from .module import Module, module_update_wrapper, Static


@compile_cache
def _filter_jit_cache(fun_names, donate_default, **jitkwargs):
    def fun_wrapped(dynamic, static):
        dynamic_fun, dynamic_spec = dynamic
        static_fun, static_spec = static

        fun = hashable_combine(dynamic_fun, static_fun)

        args, kwargs = hashable_combine(dynamic_spec, static_spec)
        out = fun(*args, **kwargs)
        dynamic_out, static_out = partition(out, is_array)
        return dynamic_out, Static(static_out)

    fun_name, fun_qualname = fun_names
    fun_wrapped.__name__ = fun_name
    fun_wrapped.__qualname__ = fun_qualname

    if donate_default:
        return jax.jit(fun_wrapped, static_argnums=1, donate_argnums=0, **jitkwargs)
    else:
        return jax.jit(fun_wrapped, static_argnums=1, **jitkwargs)


class _JitWrapper(Module):
    _signature: inspect.Signature
    _dynamic_fun: PyTree[Any]
    _static_fun: Tuple[TreeDef, Sequence[Any]]
    _cached: FunctionType
    _filter_warning: bool

    def _fun_wrapper(self, is_lower, args, kwargs):
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        args = bound.args
        kwargs = bound.kwargs

        dynamic_spec, static_spec = hashable_partition((args, kwargs), is_array)
        dynamic = (self._dynamic_fun, dynamic_spec)
        static = (self._static_fun, static_spec)
        if is_lower:
            return self._cached.lower(dynamic, static)
        else:
            dynamic_out, static_out = self._cached(dynamic, static)
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
    """A simplified version of `jax.jit`.

    !!! info

        By default, all JAX arrays are traced, all other types are held static, and all
        buffers of JAX array are donated. ('Donation' meaning that the memory is reused
        for the outputs of the JIT'd function.)

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
        # Basic behaviour
        @eqx.filter_jit
        def f(x, y):  # both args traced if arrays, static if non-arrays
            return x + y, x - y

        f(jnp.array(1), jnp.array(2))  # both args traced
        f(jnp.array(1), 2)  # first arg traced, second arg static
        f(1, 2)  # both args static

        # Donation behaviour
        x = jnp.array(1)
        y = jnp.array(1)
        f(x, y)
        x.is_deleted()  # True
        y.is_deleted()  # True

        def f_donate_y_only(x, y):
            x = jnp.copy(x)  # avoid donation by copying an array
            return f(x, y)

        x = jnp.array(1)
        y = jnp.array(2)
        f_donate_y_only(x, y)
        x.is_deleted()  # False
        y.is_deleted()  # True

        # Trace int/float/bool/complex as well
        def f_trace_arraylike(x, y):
            x, y = jax.tree_util.tree_map(jnp.asarray, (x, y))
            return f(x, y)

        f_trace_arraylike(1, True)  # both args traced
        f_trace_arraylike(1+1j, 2.0)  # both args traced
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

    dynamic_fun, static_fun = hashable_partition(fun, is_array)
    cached = _filter_jit_cache(get_fun_names(fun), donate_default, **jitkwargs)

    jit_wrapper = _JitWrapper(
        _signature=signature,
        _dynamic_fun=dynamic_fun,
        _static_fun=static_fun,
        _cached=cached,
        _filter_warning=filter_warning,
    )
    return module_update_wrapper(jit_wrapper, fun)
