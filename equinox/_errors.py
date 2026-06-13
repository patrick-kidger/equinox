import functools as ft
import traceback
import types
import warnings
from collections.abc import Callable, Sequence
from typing import Literal

import jax
import jax._src.traceback_util as traceback_util
import jax.core
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import numpy.typing as npt
from jaxtyping import Array, ArrayLike, Bool, Int, PyTree

from . import _jit
from ._ad import filter_custom_jvp
from ._config import EQX_ON_ERROR, EQX_ON_ERROR_BREAKPOINT_FRAMES
from ._doc_utils import doc_remove_args
from ._filters import combine, is_array, partition
from ._misc import currently_jitting
from ._unvmap import unvmap_any


traceback_util.register_exclusion(__file__)


def _nan_like(x: Array | np.ndarray) -> Array | np.ndarray:
    dtype = np.result_type(x)
    if np.issubdtype(dtype, np.inexact):
        return np.broadcast_to(np.array(np.nan, dtype), x.shape)
    elif np.issubdtype(dtype, np.integer):
        return np.broadcast_to(np.array(np.iinfo(dtype).max, dtype), x.shape)
    elif np.issubdtype(dtype, np.bool_):
        return np.broadcast_to(np.array(True, dtype), x.shape)
    else:
        return x


def _tree_nan_like(x: PyTree) -> PyTree:
    return jtu.tree_map(_nan_like, x)


_tpu_msg = """

Computation halted with the above error message. No Python exception is being raised as
it looks like you're on the TPU, and the TPU runtime doesn't support raising errors.

If you are running this program interactively (e.g. in a Colab notebook), then you may
now press enter to attempt to finish running the program, using dummy values (e.g. NaN).

Otherwise, to avoid downstream errors the program will now stay in an infinite loop.
You will need to manually kill this job and/or restart the runtime.
"""


_frames_msg = f"""
-------------------

Opening a breakpoint with {EQX_ON_ERROR_BREAKPOINT_FRAMES} frames. You can control this
value by setting the environment variable `EQX_ON_ERROR_BREAKPOINT_FRAMES=<some value>`.
(Note that setting large values of this number may lead to crashes at trace time; see
`https://docs.kidger.site/equinox/api/errors/#equinox.error_if` for more information.)
"""


# The name of this is looked for in `_jit.py` in order to determine if we have a
# runtime error -- and if so then the custom reporting will engage.
#
# Note that this is *not* the class that is raised at runtime to a user: this is an
# internal implementation detail of Equinox. It is caught by `equinox.filter_jit` and
# replaced with the actual run time error. (Without any of the misleading baggage that
# XLA would otherwise attach.)
class _EquinoxRuntimeError(RuntimeError):
    pass


def _get_message(
    dynamic_x: PyTree[npt.ArrayLike],
    static_x: PyTree,
    pred: Bool[npt.ArrayLike, "*shape"],
    msg: Callable[[PyTree], str],
) -> str:
    dynamic_x = jtu.tree_map(np.asarray, dynamic_x)
    pred = np.asarray(pred)
    if pred.shape == ():
        # Common scalar case
        x = combine(dynamic_x, static_x)
        return msg(x)
    else:
        # Batched case, report which batch element had the error + potentially report
        # multiple errors.
        output = []
        for index in zip(*np.nonzero(pred)):
            index = tuple(np.asarray(i).item() for i in index)
            x = combine(jtu.tree_map(lambda _x: _x[index], dynamic_x), static_x)
            msg_string = msg(x)
            output.append(f"Batch index {index} had error:\n{msg_string}")
        return "\n\n".join(output)


@filter_custom_jvp
def _error_inner(
    dynamic_x, pred, *, static_x, msg: Callable[[PyTree], str], on_error, stack
):
    assert callable(msg)

    if on_error == "raise":

        def raises(_dynamic_x, _pred):
            msg_string = _get_message(_dynamic_x, static_x, _pred, msg)
            # Sneakily smuggle out the information about the error. Inspired by
            # `sys.last_value`.
            _jit.last_error_info = (msg_string, stack)
            raise _EquinoxRuntimeError(
                f"{msg_string}\n\n\n"
                "--------------------\n"
                "An error occurred during the runtime of your JAX program! "
                "Unfortunately you do not appear to be using `equinox.filter_jit` "
                "(perhaps you are using `jax.jit` instead?) and so further information "
                "about the error cannot be displayed. (Probably you are seeing a very "
                "large but uninformative error message right now.) Please wrap your "
                "program with `equinox.filter_jit`.\n"
                "--------------------\n"
            )

        def tpu_msg(_out, _pred):
            msg_string = _get_message(_out, static_x, _pred, msg)
            # `print` doesn't work; nor does `jax.debug.print`.
            # But both `input` and `jax.debug.breakpoint` do. The former allows us to
            # actually display something to the user.
            input(msg_string + _tpu_msg)
            # We do the tree_map inside the pure_callback, not outside, so that `out`
            # has a data dependency and doesn't get optimised out.
            return jtu.tree_map(_nan_like, _out)

        def handle_error():  # pyright: ignore
            out = jax.pure_callback(
                raises, dynamic_x, dynamic_x, pred, vmap_method="broadcast_all"
            )
            # If we make it this far then we're on the TPU, which squelches runtime
            # errors and returns dummy values instead.
            # Fortunately, we're able to outsmart it!
            return jax.pure_callback(
                tpu_msg, dynamic_x, out, pred, vmap_method="broadcast_all"
            )

        return lax.cond(unvmap_any(pred), handle_error, lambda: dynamic_x)

    elif on_error == "breakpoint":

        def display_msg(_dynamic_x, _pred):
            print(_frames_msg)
            msg_string = _get_message(_dynamic_x, static_x, _pred, msg)
            print("equinox.EquinoxRuntimeError: " + msg_string)
            return _dynamic_x

        def handle_error():
            out = jax.pure_callback(
                display_msg, dynamic_x, dynamic_x, pred, vmap_method="broadcast_all"
            )
            out = jax.debug.breakpoint(
                token=out, num_frames=EQX_ON_ERROR_BREAKPOINT_FRAMES
            )
            return jax.pure_callback(
                _tree_nan_like, dynamic_x, out, vmap_method="broadcast_all"
            )

        return lax.cond(unvmap_any(pred), handle_error, lambda: dynamic_x)

    elif on_error == "nan":
        return lax.cond(unvmap_any(pred), _tree_nan_like, lambda y: y, dynamic_x)
    elif on_error == "off":
        return dynamic_x
    else:
        assert False


# Use a custom_jvp to put the lax.cond outside of AD.
# This is needed as (a) lax.cond will unnecessarily promote symbolic
# zeros to non-symbolic-zeros, and we'd really like to avoid that, and (b) we need to
# wrap our pure_callbacks in custom JVP rules.
@_error_inner.def_jvp
def _error_inner_jvp(primals, tangents, *, static_x, msg, on_error, stack):
    x, pred = primals
    tx, _ = tangents
    return _error_inner(
        x, pred, static_x=static_x, msg=msg, on_error=on_error, stack=stack
    ), tx


def _error_outer(
    x: PyTree,
    pred: Bool[ArrayLike, ""],
    msg: Callable[[PyTree], str],
    *,
    on_error: Literal["default", "raise", "breakpoint", "off", "nan"],
) -> PyTree:
    if jnp.shape(pred) != ():
        raise ValueError("`equinox.error_if(..., pred=...)` must be a scalar.")
    if not jnp.issubdtype(jnp.result_type(pred), jnp.bool_):
        raise ValueError("`equinox.error_if(..., pred=...)` must be a boolean.")
    if on_error == "default":
        on_error = EQX_ON_ERROR
    if on_error not in ("raise", "breakpoint", "off", "nan"):
        raise RuntimeError("Unrecognised value for `on_error`.")
    # Short-circuit if the predicate is known-falsey at compile time, no need to include
    # this in the graph.
    with jax.ensure_compile_time_eval():
        if isinstance(pred, bool):
            tracepred = pred
        else:
            tracepred = unvmap_any(pred)
        if not isinstance(tracepred, jax.core.Tracer):
            if isinstance(tracepred, jax.Array):
                tracepred = tracepred.item()
            assert type(tracepred) is bool
            if tracepred is False:
                return x
    stack: list[bytes | str] = []
    for frame, lineno in traceback.walk_stack(None):
        frame_id = frame.f_locals.get("__equinox_jit_id__", None)
        if type(frame_id) is bytes:
            stack.append(frame_id)
        if traceback_util.include_frame(frame):
            # This seems to be the simplest way to format a single frame?
            frame_str: str = "".join(
                traceback.format_tb(
                    types.TracebackType(None, frame, frame.f_lasti, lineno)
                )
            )
            stack.append(frame_str)
    dynamic_x, static_x = partition(x, is_array)
    flat = jtu.tree_leaves(dynamic_x)
    if len(flat) == 0:
        raise ValueError("No arrays to thread error on to.")
    dynamic_x = _error_inner(
        dynamic_x, pred, static_x=static_x, msg=msg, on_error=on_error, stack=stack
    )
    return combine(dynamic_x, static_x)


# filter_jit does some work to produce nicer runtime error messages.
# We also place it here to ensure a consistent experience when using JAX in eager mode.
_error_outer_jit = _jit.filter_jit(_error_outer)


def _error_impl(
    x: PyTree,
    pred: Bool[ArrayLike, ""],
    msg: Callable[[PyTree], str],
    *,
    on_error: Literal["default", "raise", "breakpoint", "nan", "off"],
) -> PyTree:
    leaves = jtu.tree_leaves((x, pred))
    # This carefully does not perform any JAX operations if `pred` and `index` are
    # a bool and an int.
    # This ensures we can use `error_if` before init_google.
    if any(is_array(leaf) for leaf in leaves):
        return _error_outer_jit(x, pred, msg, on_error=on_error)
    else:
        return _error_outer(x, pred, msg, on_error=on_error)


if EQX_ON_ERROR == "breakpoint":
    # TODO: remove this branch once JAX issue #16732 is fixed.
    _old_jit = jax.jit

    @ft.wraps(jax.jit)
    def fixed_jit(fun, *args, **kwargs):
        jit_fun = _old_jit(fun, *args, **kwargs)

        def fixed_jit_impl(*args2, **kwargs2):
            if currently_jitting():
                warnings.warn(
                    "Ignoring intermediate `jax.jit` decorator, to work around JAX "
                    "issue #16732, as `EQX_ON_ERROR=breakpoint` is set."
                )
                return fun(*args2, **kwargs2)
            else:
                return jit_fun(*args2, **kwargs2)

        return fixed_jit_impl

    jax.jit = fixed_jit


# Remove the `on_error` argument from the public API for now. If you pass
# `on_error="breakpoint"` -- rather than setting `EQX_ON_ERROR=breakpoint` -- then our
# fix for JAX issue #16732 -- above -- can't kick in. So in practice this
# argument probably won't work.
@doc_remove_args("on_error")
def error_if(
    x: PyTree,
    pred: Bool[ArrayLike, ""],
    msg: str | Callable[[PyTree], str],
    *,
    on_error: Literal["default", "raise", "breakpoint", "nan", "off"] = "default",
) -> PyTree:
    """Throws an error based on runtime values. Works even under JIT.

    **Arguments:**

    - `x`: will be returned unchanged. This is used to determine where the error check
        happens in the overall computation: it will happen after `x` is computed and
        before the return value is used. `x` can be any PyTree, and it must contain at
        least one array.
    - `pred`: a boolean for whether to raise an error. If vmap'd then an error will be
        raised if any batch element has `True`.
    - `msg`: the string to display as an error message.

    **Returns:**

    The original argument `x` unchanged. **If this return value is unused then the error
    check will not be performed.** (It will be removed as part of dead code
    elimination.)

    !!! Example

        ```python
        @jax.jit
        def f(x):
            x = error_if(x, x < 0, "x must be >= 0")
            # ...use x in your computation...
            return x

        f(jax.numpy.array(-1))
        ```

    **Configuration:**

    The `EQX_ON_ERROR` environment variable is checked for how any runtime errors should
    be handled. Possible values are:

    - `EQX_ON_ERROR=raise` will raise a runtime error.
    - `EQX_ON_ERROR=nan` will return `NaN` instead of `x`, and then continue the
        computation.
    - `EQX_ON_ERROR=breakpoint` will open a debugger.
        - Note that this option may prevent certain compiler optimisations, so
            permanently fixing this value is not recommended.
        - You will need to also pass the `-s` flag to `pytest`, if you are
            also using that.
        - By default this only allows you to see a single frame in the debugger. This is
            to work around JAX bug [#16732](https://github.com/google/jax/issues/16732).
            (Bugs whilst debugging bugs, eek!) In practice you may like to set the
            `EQX_ON_ERROR_BREAKPOINT_FRAMES` environment variable to a small integer,
            which specifies how many frames upwards the debugger should capture. The
            JAX bug is triggered when taking too many frames.
    - `EQX_ON_ERROR=off` turns off all error checking. This is useful for removing
        performance penalties incurred from use of `error_if`.

    After changing an environment variable, the Python process must be restarted.
    """
    if isinstance(msg, str):
        old_msg = msg
        msg = lambda _: old_msg
    return _error_impl(x, pred, msg, on_error=on_error)


@doc_remove_args("on_error")
def branched_error_if(
    x: PyTree,
    pred: Bool[ArrayLike, ""],
    index: Int[ArrayLike, ""],
    msgs: Sequence[str],
    *,
    on_error: Literal["default", "raise", "breakpoint", "nan", "off"] = "default",
) -> PyTree:
    def msg(x__index) -> str:
        _, _index = x__index
        _index = _index.item()
        assert type(_index) is int
        if _index < 0:
            return (
                f"Got a negative value `{_index}` for "
                "`equinox.branched_error_if(..., index=...)`, which is not supported."
            )
        if _index >= len(msgs):
            return (
                f"Got value `{_index}` for `equinox.branched_error_if(..., index=...)`,"
                f" which is not a valid index given `len(msgs)={len(msgs)}`."
            )
        return msgs[_index]

    warnings.warn(
        "`equinox.branched_error_if` is deprecated in favour of passing a callable to "
        "`equinox.error_if(..., msg=...)`.",
        stacklevel=2,
    )
    index = jnp.asarray(index)
    if jnp.shape(index) != ():
        raise ValueError("`branched_error_if(..., index=...)` must be a scalar.")
    if not jnp.issubdtype(jnp.result_type(index), jnp.integer):
        raise ValueError("`branched_error_if(..., index=...)` must have integer dtype.")
    out, _ = _error_impl((x, index), pred, msg=msg, on_error=on_error)
    return out


def assert_dce(
    x: PyTree,
    msg: str,
    *,
    on_error: Literal["default", "raise", "breakpoint", "off", "nan"] = "default",
) -> PyTree:
    """Asserts that a particular array (or PyTree of arrays) is DCE'd."""

    if currently_jitting():
        return error_if(x, True, msg, on_error=on_error)
    else:
        # Don't run if not JIT'ing, as without the compiler nothing will be DCE'd.
        return x
