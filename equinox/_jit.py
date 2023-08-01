import functools as ft
import inspect
import warnings
from collections.abc import Callable
from typing import Any, overload, TypeVar
from typing_extensions import ParamSpec

import jax
import jax._src.traceback_util as traceback_util
import jax.core
from jaxtyping import PyTree

from ._compile_utils import (
    compile_cache,
    hashable_combine,
    hashable_filter,
    hashable_partition,
    Lowered,
)
from ._custom_types import sentinel
from ._deprecate import deprecated_0_10
from ._doc_utils import doc_remove_args
from ._filters import combine, is_array, partition
from ._misc import currently_jitting
from ._module import Module, module_update_wrapper, Partial, Static


traceback_util.register_exclusion(__file__)


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


def _bind(signature, args, kwargs):
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    args = bound.args
    kwargs = bound.kwargs
    return args, kwargs


def _preprocess(info, args, kwargs):
    signature, dynamic_fun, static_fun = info
    args, kwargs = _bind(signature, args, kwargs)
    dynamic_spec = hashable_filter((args, kwargs), is_array)
    dynamic = (dynamic_fun, dynamic_spec)
    return dynamic


def _postprocess(out):
    dynamic_out, static_out = out
    return combine(dynamic_out, static_out.value)


try:
    # Not public API, so wrap in a try-except for forward compatibility.
    XlaRuntimeError = jax.lib.xla_extension.XlaRuntimeError  # pyright: ignore
except Exception:
    # Unused dummy
    class XlaRuntimeError(Exception):
        pass


_eqx_on_error_msg = """
-------
This error occurred during the runtime of your JAX program. Setting the environment
variable `EQX_ON_ERROR=breakpoint` is usually the most useful way to debug such errors.
(This can be navigated using most of the the usual commands for the Python debugger:
`u` and `d` to move through stack frames, the name of a variable to print its value,
etc.) See also `https://docs.kidger.site/equinox/api/errors/#equinox.error_if` for more
information.
"""

_eqx_traceback_filtering_msg = """
-------
This error has some JAX- and Equinox-internal frames removed from the traceback. Set
`JAX_TRACEBACK_FILTERING=off` to include these internal frames as well.
"""


def _modify_traceback(e: Exception):
    # Remove JAX's UnfilteredStackTrace, with its huge error messages.
    e.__cause__ = None
    # Remove _JitWrapper.__call__ and _JitWrapper._call from the traceback
    e.__traceback__ = e.__traceback__.tb_next.tb_next  # pyright: ignore
    # IPython ignores __tracebackhide__ directives for the frame that actually raises
    # the error. We fix that here.
    if jax.config.jax_traceback_filtering in (None, "auto"):  # pyright: ignore
        try:
            get_ipython()  # pyright: ignore
        except NameError:
            pass
        else:
            import IPython  # pyright: ignore

            # Check that IPython supports __tracebackhide__
            if IPython.version_info[:2] >= (7, 17):  # pyright: ignore
                tb = e.__traceback__
                tb_stack = []
                while tb is not None:
                    tb_stack.append(tb)
                    tb = tb.tb_next
                for tb in reversed(tb_stack):
                    if not tb.tb_frame.f_locals.get("__tracebackhide__", False):
                        tb.tb_next = None
                        break
                else:
                    e.__traceback__ = None


class _JitWrapper(Module):
    _signature: inspect.Signature
    _dynamic_fun: PyTree
    _static_fun: Any
    _cached: Any
    _filter_warning: bool

    @property
    def __wrapped__(self):
        return hashable_combine(self._dynamic_fun, self._static_fun)

    def _call(self, is_lower, args, kwargs):
        args, kwargs = _bind(self._signature, args, kwargs)
        dynamic_spec, static_spec = hashable_partition((args, kwargs), is_array)
        dynamic = (self._dynamic_fun, dynamic_spec)
        static = (self._static_fun, static_spec)
        if is_lower:
            return Lowered(
                self._cached.lower(dynamic, static),
                (self._signature, self._dynamic_fun, self._static_fun),
                _preprocess,  # pyright: ignore
                _postprocess,  # pyright: ignore
            )
        else:
            if self._filter_warning is True:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="Some donated buffers were not usable*"
                    )
                    out = self._cached(dynamic, static)
            else:
                out = self._cached(dynamic, static)
            return _postprocess(out)

    def __call__(self, /, *args, **kwargs):
        try:
            return self._call(False, args, kwargs)
        except XlaRuntimeError as e:
            # Catch Equinox's runtime errors, and strip the more intimidating parts of
            # the error message.
            (msg,) = e.args
            prefix = "INTERNAL: Generated function failed: CpuCallback error: EqxRuntimeError: "  # noqa: E501
            is_eqx_error = msg.startswith(prefix)
            if is_eqx_error:
                msg = msg.removeprefix(prefix)
                msg, _ = msg.rsplit("\n\nAt:\n", 1)
                msg = msg + _eqx_on_error_msg
                e.args = (msg,)
                if jax.config.jax_traceback_filtering in (  # pyright: ignore
                    None,
                    "auto",
                ):
                    _modify_traceback(e)
            raise
        except Exception as e:
            # Catch JAX's trace-time errors, and strip the terrifying-looking traceback.
            # This is one of the most off-putting things about JAX errors!
            if (
                jax.config.jax_traceback_filtering in (None, "auto")  # pyright: ignore
                and not currently_jitting()
            ):
                if len(e.args) == 0:  # `raise FooException`
                    msg = _eqx_traceback_filtering_msg
                elif len(e.args) == 1:  # `raise FooException("...")`
                    (msg,) = e.args
                    msg = msg + _eqx_traceback_filtering_msg
                else:  # No idea if this ever happens
                    raise
                # Edit in-place because JAX errors have a custom __init__.
                e.args = (msg,)
                _modify_traceback(e)
            raise

    def lower(self, /, *args, **kwargs) -> Lowered:
        return self._call(True, args, kwargs)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return Partial(self, instance)


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
    cached = _filter_jit_cache(fun, jitkwargs)

    jit_wrapper = _JitWrapper(
        _signature=signature,
        _dynamic_fun=dynamic_fun,
        _static_fun=static_fun,
        _cached=cached,
        _filter_warning=filter_warning,
    )
    return module_update_wrapper(jit_wrapper, fun)
