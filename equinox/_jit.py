import atexit
import functools as ft
import inspect
import logging
import os
import warnings
from collections.abc import Callable
from typing import Any, Literal, overload, TypeVar
from typing_extensions import ParamSpec

import jax
import jax._src.dispatch
import jax._src.traceback_util as traceback_util
import jax.core
import jax.errors
import jax.numpy as jnp
from jaxtyping import PyTree

from ._compile_utils import (
    compile_cache,
    get_fn_names,
    hashable_combine,
    hashable_filter,
    hashable_partition,
    Lowered,
)
from ._custom_types import sentinel
from ._deprecate import deprecated_0_10
from ._doc_utils import doc_remove_args
from ._filters import combine, is_array, partition
from ._module import field, Module, module_update_wrapper, Partial, Static


traceback_util.register_exclusion(__file__)


_P = ParamSpec("_P")
_T = TypeVar("_T")


@compile_cache
def _filter_jit_cache(fun_names, jitkwargs):
    def fun_wrapped(dynamic_donate, dynamic_nodonate, static):
        dynamic_dict = dict(**dynamic_donate, **dynamic_nodonate)
        dynamic_fun = dynamic_dict.pop("fun")
        dynamic_first = dynamic_dict.pop("first")
        dynamic_rest = dynamic_dict.pop("rest")
        static_fun, static_first, static_rest = static
        fun = hashable_combine(dynamic_fun, static_fun)
        first_arg = hashable_combine(dynamic_first, static_first)
        rest_args, kwargs = hashable_combine(dynamic_rest, static_rest)
        assert type(rest_args) is tuple
        *args, dummy_arg = (first_arg,) + rest_args
        assert dummy_arg is None
        out = fun(*args, **kwargs)
        dynamic_out, static_out = partition(out, is_array)
        marker = jnp.array(0)
        return marker, dynamic_out, Static(static_out)

    fun_name, fun_qualname = fun_names
    fun_wrapped.__name__ = fun_name
    fun_wrapped.__qualname__ = fun_qualname
    return jax.jit(fun_wrapped, donate_argnums=0, static_argnums=2, **jitkwargs)


def _bind(signature, args, kwargs):
    bound = signature.bind(*args, **kwargs)
    args = bound.args
    kwargs = bound.kwargs
    return args, kwargs


def _preprocess(info, args, kwargs, return_static: bool = False):
    signature, dynamic_fun, static_fun, donate_first, donate_rest = info
    args, kwargs = _bind(signature, args, kwargs)
    # add dummy to avoid special casing `len(args) == 0`.
    args = args + (None,)
    first_arg = args[0]
    rest_args = args[1:]
    if return_static:
        dynamic_first, static_first = hashable_partition(first_arg, is_array)
        dynamic_rest, static_rest = hashable_partition((rest_args, kwargs), is_array)
    else:
        dynamic_first = hashable_filter(first_arg, is_array)
        dynamic_rest = hashable_filter((rest_args, kwargs), is_array)
    dynamic_donate = dict()
    dynamic_nodonate = dict()
    if donate_first:
        dynamic_donate["first"] = dynamic_first
    else:
        dynamic_nodonate["first"] = dynamic_first
    if donate_rest:
        dynamic_donate["fun"] = dynamic_fun
        dynamic_donate["rest"] = dynamic_rest
    else:
        dynamic_nodonate["fun"] = dynamic_fun
        dynamic_nodonate["rest"] = dynamic_rest

    if return_static:
        static = (static_fun, static_first, static_rest)  # pyright: ignore
        return dynamic_donate, dynamic_nodonate, static
    else:
        return dynamic_donate, dynamic_nodonate


def _postprocess(out):
    _, dynamic_out, static_out = out
    return combine(dynamic_out, static_out.value)


try:
    # Added in JAX 0.4.34.
    JaxRuntimeError = jax.errors.JaxRuntimeError  # pyright: ignore
except AttributeError:
    try:
        # Forward compatibility in case they ever decide to fix the capitalization.
        JaxRuntimeError = jax.errors.JAXRuntimeError  # pyright: ignore
    except AttributeError:
        # Not public API, so wrap in a try-except for forward compatibility.
        try:
            JaxRuntimeError = jax.lib.xla_extension.XlaRuntimeError  # pyright: ignore
        except Exception:
            # Unused dummy
            class JaxRuntimeError(Exception):
                pass


try:
    wait_for_tokens = jax._src.dispatch.wait_for_tokens
except AttributeError:
    pass  # forward compatibility
else:
    # Fix for https://github.com/patrick-kidger/diffrax/issues/506
    def wait_for_tokens2():
        try:
            wait_for_tokens()
        except JaxRuntimeError:
            pass

    atexit.unregister(wait_for_tokens)
    atexit.register(wait_for_tokens2)


# This is the class we use to raise runtime errors from `eqx.error_if`.
class EquinoxRuntimeError(RuntimeError):
    pass


# Magic value that means error messages are displayed as `{__qualname__}: ...` rather
# than `{__module__}.{__qualname__}`. (At least, I checked the default Python
# interpreter, the default Python REPL, ptpython, ipython, pdb, and ipdb.)
EquinoxRuntimeError.__module__ = "builtins"
# Note that we don't also override `__name__` or `__qualname__`. Suppressing the
# `equinox._jit` module bit is useful for readability, but we don't want to go so far as
# deleting the name altogether. (Or even e.g. setting it to the 'Above is the stack...'
# first section of our error message below!) The reason is that whilst that gives a
# nicer displayed error in default Python, it doesn't necessarily do as well with other
# tools, e.g. debuggers. So what we have here is a compromise.


last_error_info: None | tuple[str, list[bytes | str]] = None


_on_error_msg = """Above is the stack outside of JIT. Below is the stack inside of JIT:
{stack}
equinox.EquinoxRuntimeError: {msg}

-------------------

An error occurred during the runtime of your JAX program.

1) Setting the environment variable `EQX_ON_ERROR=breakpoint` is usually the most useful
way to debug such errors. This can be interacted with using most of the usual commands
for the Python debugger: `u` and `d` to move up and down frames, the name of a variable
to print its value, etc.

2) You may also like to try setting `JAX_DISABLE_JIT=1`. This will mean that you can
(mostly) inspect the state of your program as if it was normal Python.

3) See `https://docs.kidger.site/equinox/api/debug/` for more suggestions.
"""


class _FilterCallback(logging.Filterer):
    def filter(self, record: logging.LogRecord):
        return not (
            record.name == "jax._src.callback"
            and record.getMessage() == "jax.pure_callback failed"
        )


class _JitWrapper(Module):
    fn: str  # this attribute exists solely to give a nice repr
    _signature: inspect.Signature = field(static=True, repr=False)
    _dynamic_fun: PyTree = field(repr=False)
    _static_fun: Any = field(static=True, repr=False)
    _cached: Any = field(static=True, repr=False)
    filter_warning: bool = field(static=True)
    donate_first: bool = field(static=True)
    donate_rest: bool = field(static=True)

    @property
    def __wrapped__(self):
        return hashable_combine(self._dynamic_fun, self._static_fun)

    def __call__(self, /, *args, **kwargs):
        __tracebackhide__ = True
        try:
            return _call(self, False, args, kwargs)
        except EquinoxRuntimeError as e:
            # Use a two-part try/except here and in `_call` to delete the
            # `raise EquinoxRuntimeError` line from the stack trace.
            e.__traceback__ = None
            raise

    def lower(self, /, *args, **kwargs) -> Lowered:
        return _call(self, True, args, kwargs)

    def __get__(self, instance, owner):
        del owner
        if instance is None:
            return self
        return Partial(self, instance)


# _call is not a member method of _JitWrapper (even though it effectively does
# the same thing) because we want to avoid _call being wrapped in a _wrap_method,
# which adds about ~90μs per call.
def _call(jit_wrapper: _JitWrapper, is_lower, args, kwargs):
    __tracebackhide__ = True
    __equinox_jit_id__ = os.urandom(16)
    info = (
        jit_wrapper._signature,
        jit_wrapper._dynamic_fun,
        jit_wrapper._static_fun,
        jit_wrapper.donate_first,
        jit_wrapper.donate_rest,
    )
    dynamic_donate, dynamic_nodonate, static = _preprocess(  # pyright: ignore
        info, args, kwargs, return_static=True
    )
    if is_lower:
        return Lowered(
            jit_wrapper._cached.lower(dynamic_donate, dynamic_nodonate, static),
            info,
            _preprocess,  # pyright: ignore
            _postprocess,  # pyright: ignore
        )
    else:
        filter = _FilterCallback()
        callback_logger = logging.getLogger("jax._src.callback")
        callback_logger.addFilter(filter)
        try:
            if jit_wrapper.filter_warning:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="Some donated buffers were not usable*"
                    )
                    marker, _, _ = out = jit_wrapper._cached(
                        dynamic_donate, dynamic_nodonate, static
                    )
            else:
                marker, _, _ = out = jit_wrapper._cached(
                    dynamic_donate, dynamic_nodonate, static
                )
            # We need to include the explicit `isinstance(marker, jax.Array)` check due
            # to https://github.com/patrick-kidger/equinox/issues/988
            if not isinstance(marker, jax.core.Tracer) and isinstance(
                marker, jax.Array
            ):
                marker.block_until_ready()
        except JaxRuntimeError as e:
            # Catch Equinox's runtime errors, and re-raise them with actually useful
            # information. (By default XlaRuntimeError produces a lot of terrifying
            # but useless information.)
            if last_error_info is not None and "_EquinoxRuntimeError: " in str(e):
                last_msg, last_stack = last_error_info
                last_stack_pieces: list[str] = []
                for id_or_str in last_stack:
                    if type(id_or_str) is str:
                        last_stack_pieces.append(id_or_str)
                    else:
                        if id_or_str == __equinox_jit_id__:
                            break
                last_stack_str = "".join(reversed(last_stack_pieces))
                raise EquinoxRuntimeError(
                    _on_error_msg.format(msg=last_msg, stack=last_stack_str)
                ) from None
                # `from None` to hide the large but uninformative XlaRuntimeError.
            else:
                raise
        finally:
            callback_logger.removeFilter(filter)
        return _postprocess(out)


@overload
def filter_jit(
    *,
    donate: Literal[
        "all", "all-except-first", "warn", "warn-except-first", "none"
    ] = "none",
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...


@overload
def filter_jit(
    fun: Callable[_P, _T],
    *,
    donate: Literal[
        "all", "all-except-first", "warn", "warn-except-first", "none"
    ] = "none",
) -> Callable[_P, _T]: ...


@doc_remove_args("jitkwargs")
def filter_jit(
    fun=sentinel,
    *,
    donate: Literal[
        "all", "all-except-first", "warn", "warn-except-first", "none"
    ] = "none",
    **jitkwargs,
):
    """An easier-to-use version of `jax.jit`. All JAX and NumPy arrays are traced, and
    all other types are held static.

    **Arguments:**

    - `fun` is a pure function to JIT compile.
    - `donate` indicates whether the buffers of JAX arrays are donated or not. It
        should either be:
        - `'all'`: donate all arrays and suppress all warnings about unused buffers;
        - `'all-except-first'`: donate all arrays except for those in the first
            argument, and suppress all warnings about unused buffers;
        - `'warn'`: as above, but don't suppress unused buffer warnings;
        - `'warn-except-first'`: as above, but don't suppress unused buffer warnings;
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
        `filter_jit(donate="all-except-first")` and then passing all arguments that you
        don't want to donate through the first argument. (Packing multiple values into
        a tuple if necessary.)
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

    if donate == "all":
        filter_warning = True
        donate_first = True
        donate_rest = True
    elif donate == "all-except-first":
        filter_warning = True
        donate_first = False
        donate_rest = True
    elif donate == "warn":
        filter_warning = False
        donate_first = True
        donate_rest = True
    elif donate == "warn-except-first":
        filter_warning = False
        donate_first = False
        donate_rest = True
    elif donate == "none":
        filter_warning = False
        donate_first = False
        donate_rest = False
    else:
        raise ValueError(
            "`filter_jit(..., donate=...)` must be one of 'all', 'all-except-first', "
            "'warn', 'warn-except-first', or 'none'."
        )

    _, name = get_fn_names(fun)
    dynamic_fun, static_fun = hashable_partition(fun, is_array)
    cached = _filter_jit_cache(fun, jitkwargs)

    jit_wrapper = _JitWrapper(
        fn=name,
        _signature=signature,
        _dynamic_fun=dynamic_fun,
        _static_fun=static_fun,
        _cached=cached,
        filter_warning=filter_warning,
        donate_first=donate_first,
        donate_rest=donate_rest,
    )
    return module_update_wrapper(jit_wrapper)
