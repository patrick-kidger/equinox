import os
import warnings
from collections.abc import Sequence
from typing import cast, Literal

import jax
import jax._src.traceback_util as traceback_util
import jax.core
import jax.lax as lax
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Int, PyTree

from ._ad import filter_custom_jvp
from ._filters import combine, is_array, partition
from ._unvmap import unvmap_any, unvmap_max


traceback_util.register_exclusion(__file__)


def _nan_like(x: np.ndarray) -> np.ndarray:
    dtype = np.result_type(x)
    if np.issubdtype(dtype, np.inexact):
        return np.full(x.shape, np.nan, dtype)
    elif np.issubdtype(dtype, np.integer):
        return np.full(x.shape, np.iinfo(dtype).max, dtype)
    elif np.issubdtype(dtype, np.bool_):
        return np.full(x.shape, True, dtype)
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


@filter_custom_jvp
def _error(x, pred, index, *, msgs, on_error):
    if on_error == "raise":

        def raises(_index):
            raise RuntimeError(msgs[_index.item()])

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
            out = jax.pure_callback(raises, struct, index)  # pyright: ignore
            # If we make it this far then we're on the TPU, which squelches runtime
            # errors and returns dummy values instead.
            # Fortunately, we're able to outsmart it!
            return jax.pure_callback(tpu_msg, struct, out, index)  # pyright: ignore

        struct = jax.eval_shape(lambda: x)
        return lax.cond(pred, handle_error, lambda: x)

    elif on_error == "breakpoint":
        # TODO: find a way to have this be DCE'd if `x` is.

        def display_msg(x, _index):
            print(msgs[_index.item()])
            return x

        def handle_error(x):
            x = jax.pure_callback(display_msg, x, index)  # pyright: ignore
            return jax.debug.breakpoint(token=x)

        return lax.cond(pred, handle_error, lambda y: y, x)
    else:
        assert False


# Use a custom_jvp to put the lax.cond outside of AD.
# This is needed as lax.cond will unnecessarily promote symbolic
# zeros to non-symbolic-zeros, and we'd really like to avoid that.
@_error.def_jvp
def _error_jvp(primals, tangents, *, msgs, on_error):
    x, pred, index = primals
    tx, _, _ = tangents
    return _error(x, pred, index, msgs=msgs, on_error=on_error), tx


def error_if(
    x: PyTree,
    pred: Bool[ArrayLike, "..."],
    msg: str,
    on_error: Literal["default", "raise", "breakpoint", "unsafe_ignore"] = "default",
) -> PyTree:
    """Throws an error based on runtime values. Works even under JIT.

    **Arguments:**

    - `x`: will be returned unchanged; used to determine where the error check happens
        in the overall computation. Can be any PyTree; must contain at least one array.
    - `pred`: a boolean for whether to raise an error. Can be an array of bools; an
        error will be raised if any of them are `True`. If vmap'd then an error will be
        raised if any batch element has `True`.
    - `msg`: the string to display as an error message.
    - `on_error`: how to behave when an error is raised. Valid values are either
        `"default"`, `"raise"`, `"breakpoint"`, or `"unsafe_ignore"`. The default value
        of `"default"` defers to the `EQX_ON_ERROR` environment variable, which itself
        defaults to `"raise"`. Of the other three: `"raise"` will raise a runtime error,
        `"breakpoint"` will open a debugger, and `"unsafe_ignore"` will do nothing at
        all.

    **Returns:**

    The original argument `x` unchanged. If this return value is unused then the error
    check will not be performed.

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
    return branched_error_if(x, pred, 0, [msg], on_error)


def branched_error_if(
    x: PyTree,
    pred: Bool[ArrayLike, "..."],
    index: Int[ArrayLike, "..."],
    msgs: Sequence[str],
    on_error: Literal["default", "raise", "breakpoint", "unsafe_ignore"] = "default",
) -> PyTree:
    """As [`equinox.internal.error_if`][], but will raise one of
    several `msgs` depending on the value of `index`.
    """
    if on_error == "default":
        on_error = os.environ.get("EQX_ON_ERROR", "raise")  # pyright: ignore
        if on_error not in ("raise", "breakpoint", "unsafe_ignore"):
            raise RuntimeError("Unrecognised value for `EQX_ON_ERROR`.")
        on_error = cast(Literal["raise", "breakpoint"], on_error)
    else:
        if on_error not in ("raise", "breakpoint", "unsafe_ignore"):
            raise RuntimeError("Unrecognised value for `on_error`.")
    if on_error == "unsafe_ignore":
        return x
    with jax.ensure_compile_time_eval():
        pred = unvmap_any(pred)
        index = unvmap_max(index)
        if not isinstance(pred, jax.core.Tracer):
            if pred.item():
                if not isinstance(index, jax.core.Tracer):
                    if isinstance(index, Array):
                        index = index.item()
                    index = cast(int, index)
                    warnings.warn(
                        "`Error can be resolved statically. Handling at trace-time "
                        "rather than waiting until runtime."
                    )
                    if on_error == "raise":
                        raise RuntimeError(msgs[index])
                    elif on_error == "breakpoint":
                        print(msgs[index])
                        breakpoint()
                    else:
                        assert False
                # else defer error to runtime, when the index is known.
            else:
                return x

    dynamic_x, static_x = partition(x, is_array)
    flat = jtu.tree_leaves(dynamic_x)
    if len(flat) == 0:
        raise ValueError("No arrays to thread error on to.")
    dynamic_x = _error(dynamic_x, pred, index, msgs=msgs, on_error=on_error)
    return combine(dynamic_x, static_x)
