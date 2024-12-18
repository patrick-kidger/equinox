import functools as ft
import inspect
import traceback
import types
import warnings
from collections.abc import Sequence
from typing import Literal, Union

import jax
import jax._src.traceback_util as traceback_util
import jax.core
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Int, PyTree

from . import _jit
from ._ad import filter_custom_jvp
from ._config import EQX_ON_ERROR, EQX_ON_ERROR_BREAKPOINT_FRAMES
from ._doc_utils import doc_remove_args
from ._filters import combine, is_array, partition
from ._misc import currently_jitting
from ._unvmap import unvmap_any, unvmap_max


traceback_util.register_exclusion(__file__)


def _nan_like(x: Union[Array, np.ndarray]) -> Union[Array, np.ndarray]:
    dtype = np.result_type(x)
    if np.issubdtype(dtype, np.inexact):
        return np.broadcast_to(np.array(np.nan, dtype), x.shape)
    elif np.issubdtype(dtype, np.integer):
        return np.broadcast_to(np.array(np.iinfo(dtype).max, dtype), x.shape)
    elif np.issubdtype(dtype, np.bool_):
        return np.broadcast_to(np.array(True, dtype), x.shape)
    else:
        return x


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


class EquinoxTracetimeError(RuntimeError):
    pass


@filter_custom_jvp
def _error(x, pred, index, *, msgs, on_error, stack):
    if on_error == "raise":

        def raises(_index):
            # Sneakily smuggle out the information about the error. Inspired by
            # `sys.last_value`.
            _jit.last_msg = msg = msgs[_index.item()]
            _jit.last_stack = stack
            raise _EquinoxRuntimeError(
                f"{msg}\n\n\n"
                "--------------------\n"
                "An error occurred during the runtime of your JAX program! "
                "Unfortunately you do not appear to be using `equinox.filter_jit` "
                "(perhaps you are using `jax.jit` instead?) and so further information "
                "about the error cannot be displayed. (Probably you are seeing a very "
                "large but uninformative error message right now.) Please wrap your "
                "program with `equinox.filter_jit`.\n"
                "--------------------\n"
            )

        def tpu_msg(_out, _index):
            msg = msgs[_index.item()]
            # `print` doesn't work; nor does `jax.debug.print`.
            # But both `input` and `jax.debug.breakpoint` do. The former allows us to
            # actually display something to the user.
            input(msg + _tpu_msg)
            # We do the tree_map inside the pure_callback, not outside, so that `out`
            # has a data dependency and doesn't get optimised out.
            return jtu.tree_map(_nan_like, _out)

        def handle_error():  # pyright: ignore
            out = jax.pure_callback(raises, struct, index)
            # If we make it this far then we're on the TPU, which squelches runtime
            # errors and returns dummy values instead.
            # Fortunately, we're able to outsmart it!
            return jax.pure_callback(tpu_msg, struct, out, index)

        struct = jax.eval_shape(lambda: x)
        return lax.cond(pred, handle_error, lambda: x)

    elif on_error == "breakpoint":

        def display_msg(_index):
            print(_frames_msg)
            print("equinox.EquinoxRuntimeError: " + msgs[_index.item()])
            return _index

        def to_nan(_index):
            del _index
            return jtu.tree_map(_nan_like, struct)

        def handle_error():
            index_struct = jax.eval_shape(lambda: index)
            _index = jax.pure_callback(
                display_msg, index_struct, index, vectorized=True
            )
            # Support JAX with and without DCE behaviour on breakpoints.
            breakpoint_params = inspect.signature(
                jax.debug.breakpoint
            ).parameters.keys()
            breakpoint_kwargs = {}
            if "token" in breakpoint_params:
                breakpoint_kwargs["token"] = _index
            if "vectorized" in breakpoint_params:
                breakpoint_kwargs["vectorized"] = True
            if EQX_ON_ERROR_BREAKPOINT_FRAMES is not None:
                breakpoint_kwargs["num_frames"] = EQX_ON_ERROR_BREAKPOINT_FRAMES
            _index = jax.debug.breakpoint(**breakpoint_kwargs)
            return jax.pure_callback(to_nan, struct, _index, vectorized=True)

        struct = jax.eval_shape(lambda: x)
        return lax.cond(pred, handle_error, lambda: x)

    elif on_error == "nan":
        return lax.cond(pred, ft.partial(jtu.tree_map, _nan_like), lambda y: y, x)
    else:
        assert False


# Use a custom_jvp to put the lax.cond outside of AD.
# This is needed as (a) lax.cond will unnecessarily promote symbolic
# zeros to non-symbolic-zeros, and we'd really like to avoid that, and (b) we need to
# wrap our pure_callbacks in custom JVP rules.
@_error.def_jvp
def _error_jvp(primals, tangents, *, msgs, on_error, stack):
    x, pred, index = primals
    tx, _, _ = tangents
    return _error(x, pred, index, msgs=msgs, on_error=on_error, stack=stack), tx


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
    pred: Bool[ArrayLike, "..."],
    msg: str,
    *,
    on_error: Literal["default", "raise", "breakpoint", "nan"] = "default",
) -> PyTree:
    """Throws an error based on runtime values. Works even under JIT.

    **Arguments:**

    - `x`: will be returned unchanged. This is used to determine where the error check
        happens in the overall computation: it will happen after `x` is computed and
        before the return value is used. `x` can be any PyTree, and it must contain at
        least one array.
    - `pred`: a boolean for whether to raise an error. Can be an array of bools; an
        error will be raised if any of them are `True`. If vmap'd then an error will be
        raised if any batch element has `True`.
    - `msg`: the string to display as an error message.

    In addition, the `EQX_ON_ERROR` environment variable is checked for how any runtime
    errors should be handled. Possible values are:

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

    After changing an environment variable, the Python process must be restarted.

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
    """
    return branched_error_if(x, pred, 0, [msg], on_error=on_error)


@doc_remove_args("on_error")
def branched_error_if(
    x: PyTree,
    pred: Bool[ArrayLike, "..."],
    index: Int[ArrayLike, "..."],
    msgs: Sequence[str],
    *,
    on_error: Literal["default", "raise", "breakpoint", "nan"] = "default",
) -> PyTree:
    """As [`equinox.error_if`][], but will raise one of
    several `msgs` depending on the value of `index`. If `index` is vmap'd, then the
    error message from the largest value (across the whole batch) will be used.
    """
    leaves = jtu.tree_leaves((x, pred, index))
    # This carefully does not perform any JAX operations if `pred` and `index` are
    # a bool and an int.
    # This ensures we can use `error_if` before init_google.
    if any(is_array(leaf) for leaf in leaves):
        return branched_error_if_impl_jit(x, pred, index, msgs, on_error=on_error)
    else:
        return branched_error_if_impl(x, pred, index, msgs, on_error=on_error)


def branched_error_if_impl(
    x: PyTree,
    pred: Bool[ArrayLike, "..."],
    index: Int[ArrayLike, "..."],
    msgs: Sequence[str],
    *,
    on_error: Literal["default", "raise", "breakpoint", "nan"],
) -> PyTree:
    if on_error == "default":
        on_error = EQX_ON_ERROR
    elif on_error not in ("raise", "breakpoint", "nan"):
        raise RuntimeError("Unrecognised value for `on_error`.")
    with jax.ensure_compile_time_eval():
        # This carefully does not perform any JAX operations if `pred` and `index` are
        # a bool and an int.
        # This ensures we can use `error_if` before init_google.
        if not isinstance(pred, bool):
            pred = unvmap_any(pred)
        if not isinstance(index, int):
            index = unvmap_max(index)
        if not isinstance(pred, jax.core.Tracer):
            if isinstance(pred, Array):
                pred = pred.item()
            assert type(pred) is bool
            if pred:
                if not isinstance(index, jax.core.Tracer):
                    if isinstance(index, Array):
                        index = index.item()
                    assert type(index) is int
                    if on_error == "raise":
                        raise EquinoxTracetimeError(msgs[index])
                    elif on_error == "breakpoint":
                        print(msgs[index])
                        breakpoint()
                    elif on_error == "nan":
                        warnings.warn(
                            "Resolving error at trace time (because the predicate is "
                            "statically resolvable), by substituting NaNs (because "
                            "`on_error='nan'`)."
                        )
                        return jtu.tree_map(_nan_like, x)
                    else:
                        assert False
                # else defer error to runtime, when the index is known.
            else:
                return x

    tb = None
    for f, lineno in traceback.walk_stack(None):
        if f.f_locals.get("__equinox_filter_jit__", False):
            break
        if traceback_util.include_frame(f):
            tb = types.TracebackType(tb, f, f.f_lasti, lineno)
    stack = "".join(traceback.format_tb(tb)).rstrip()
    dynamic_x, static_x = partition(x, is_array)
    flat = jtu.tree_leaves(dynamic_x)
    if len(flat) == 0:
        raise ValueError("No arrays to thread error on to.")
    dynamic_x = _error(
        dynamic_x, pred, index, msgs=msgs, on_error=on_error, stack=stack
    )
    return combine(dynamic_x, static_x)


# filter_jit does some work to produce nicer runtime error messages.
# We also place it here to ensure a consistent experience when using JAX in eager mode.
branched_error_if_impl_jit = _jit.filter_jit(branched_error_if_impl)


def assert_dce(
    x: PyTree,
    msg: str,
    *,
    on_error: Literal["default", "raise", "breakpoint", "nan"] = "default",
) -> PyTree:
    """Asserts that a particular array (or PyTree of arrays) is DCE'd."""

    if currently_jitting():
        pred = jnp.invert(False)  # Prevent the trace-time error-raising from running.
        return error_if(x, pred, msg, on_error=on_error)
    else:
        # Don't run if not JIT'ing, as without the compiler nothing will be DCE'd.
        return x
