import functools as ft
import inspect
import warnings
from typing import Any, Callable, overload, TypeVar
from typing_extensions import ParamSpec

import jax
import jax.tree_util as jtu
from jaxtyping import PyTree

from .compile_utils import (
    compile_cache,
    get_fun_names,
    hashable_combine,
    hashable_partition,
)
from .custom_types import sentinel
from .deprecate import deprecated_0_10
from .doc_utils import doc_remove_args
from .filters import combine, is_array, partition
from .module import Module, module_update_wrapper, Static


_P = ParamSpec("_P")
_T = TypeVar("_T")


@compile_cache
def _filter_jit_cache(fun_names, jitkwargs):
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
    return jax.jit(fun_wrapped, static_argnums=1, **jitkwargs)


class _JitWrapper(Module):
    _signature: inspect.Signature
    _dynamic_fun: PyTree
    _static_fun: Any
    _cached: Any
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

    def __call__(self, /, *args, **kwargs):
        if self._filter_warning is True:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Some donated buffers were not usable*"
                )
                return self._fun_wrapper(False, args, kwargs)
        else:
            return self._fun_wrapper(False, args, kwargs)

    def lower(self, /, *args, **kwargs):
        return self._fun_wrapper(True, args, kwargs)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return jtu.Partial(self, instance)


@overload
def filter_jit(
    *, donate: str = "none"
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    ...


@overload
def filter_jit(fun: Callable[_P, _T], *, donate: str = "none") -> Callable[_P, _T]:
    ...


@doc_remove_args("jitkwargs")
def filter_jit(fun=sentinel, *, donate: str = "none", **jitkwargs):
    """An easier-to-use version of `jax.jit`. All JAX and NumPy arrays are traced, and
    all other types are held static.

    **Arguments:**

    - `fun` is a pure function to JIT compile.
    - `donate` indicates whether the buffers of JAX arrays are donated or not. It
        should either be:
        - `'all'`: donate all arrays and suppress all warnings about unused buffers;
        - `'warn'`: as above, but don't suppress unused buffer warnings;
        - `'none'`: no buffer donation. (This the default.)

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
        ```

    !!! info

        Donating arguments allows their underlying memory to be used in the
        computation. This can produce speed and memory improvements, but means that you
        cannot use any donated arguments again, as their underlying memory has been
        overwritten. (JAX will throw an error if you try to.)

    !!! info

        If you want to trace Python `bool`/`int`/`float`/`complex` as well then you
        can do this by wrapping them into a JAX array: `jnp.asarray(x)`.

        If you want to donate only some arguments then this can be done by setting
        `filter_jit(donate="all")` and then  `jnp.copy`ing any arguments you do not want
        to donate before passing them in.
    """

    if fun is sentinel:
        return ft.partial(filter_jit, donate=donate, **jitkwargs)

    deprecated_0_10(jitkwargs, "default")
    deprecated_0_10(jitkwargs, "fn")
    deprecated_0_10(jitkwargs, "args")
    deprecated_0_10(jitkwargs, "kwargs")
    deprecated_0_10(jitkwargs, "out")
    if any(
        x in jitkwargs for x in ("static_argnums", "static_argnames", "donate_argnums")
    ):
        raise ValueError(
            "`jitkwargs` cannot contain 'static_argnums', 'static_argnames' or "
            "'donate_argnums'"
        )
    signature = inspect.signature(fun)

    if donate not in {"all", "warn", "none"}:
        raise ValueError(
            "`filter_jit(..., donate=...)` must be one of 'all', 'warn', or 'none'"
        )
    filter_warning = True if donate == "all" else False
    if donate != "none":
        jitkwargs["donate_argnums"] = (0,)

    dynamic_fun, static_fun = hashable_partition(fun, is_array)
    cached = _filter_jit_cache(get_fun_names(fun), jitkwargs)

    jit_wrapper = _JitWrapper(
        _signature=signature,
        _dynamic_fun=dynamic_fun,
        _static_fun=static_fun,
        _cached=cached,
        _filter_warning=filter_warning,
    )
    return module_update_wrapper(jit_wrapper, fun)
