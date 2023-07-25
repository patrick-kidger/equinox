import functools as ft
import inspect
import os
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

from ._ad import filter_custom_jvp
from ._doc_utils import doc_remove_args
from ._filters import combine, is_array, partition
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

        def display_msg(_index):
            print(msgs[_index.item()])
            return _index

        def to_nan(_index):
            del _index
            return jtu.tree_map(_nan_like, struct)

        def handle_error():
            index_struct = jax.eval_shape(lambda: index)
            _index = jax.pure_callback(  # pyright: ignore
                display_msg, index_struct, index, vectorized=True
            )
            # Support JAX with and without DCE behaviour on breakpoints.
            if "token" in inspect.signature(jax.debug.breakpoint).parameters.keys():
                breakpoint_kwargs = dict(token=_index)
            else:
                breakpoint_kwargs = {}
            _index = jax.debug.breakpoint(**breakpoint_kwargs)
            return jax.pure_callback(  # pyright: ignore
                to_nan, struct, _index, vectorized=True
            )

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
def _error_jvp(primals, tangents, *, msgs, on_error):
    x, pred, index = primals
    tx, _, _ = tangents
    return _error(x, pred, index, msgs=msgs, on_error=on_error), tx


def _currently_jitting():
    return isinstance(jnp.array(1) + 1, jax.core.Tracer)


if os.environ.get("EQX_ON_ERROR") == "breakpoint":
    # TODO: remove this branch once JAX issue #16732 is fixed.
    _old_jit = jax.jit

    @ft.wraps(jax.jit)
    def fixed_jit(fun, *args, **kwargs):
        jit_fun = _old_jit(fun, *args, **kwargs)

        def fixed_jit_impl(*args2, **kwargs2):
            if _currently_jitting():
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
    - `EQX_ON_ERROR=breakpoint` will open a debugger. Note that this option may prevent
        certain compiler optimisations, so permanently fixing this value is not
        recommended. You will need to also pass the `-s` flag to `pytest`, if you are
        also using that.
    - `EQX_ON_ERROR=nan` will return `NaN` instead of `x`, and then continue the
        computation.

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
    if on_error == "default":
        on_error = os.environ.get("EQX_ON_ERROR", "raise")  # pyright: ignore
        if on_error not in ("raise", "breakpoint", "nan"):
            raise RuntimeError("Unrecognised value for `EQX_ON_ERROR`.")
    else:
        if on_error not in ("raise", "breakpoint", "nan"):
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
                    warnings.warn(
                        "`Error can be resolved statically. Handling at trace-time "
                        "rather than waiting until runtime."
                    )
                    if on_error == "raise":
                        raise RuntimeError(msgs[index])
                    elif on_error == "breakpoint":
                        print(msgs[index])
                        breakpoint()
                    elif on_error == "nan":
                        return jtu.tree_map(_nan_like, x)
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


def assert_dce(
    x: PyTree,
    msg: str,
    *,
    on_error: Literal["default", "raise", "breakpoint", "nan"] = "default",
) -> PyTree:
    """Asserts that a particular array (or PyTree of arrays) is DCE'd."""

    if _currently_jitting():
        pred = jnp.invert(False)  # Prevent the trace-time error-raising from running.
        return error_if(x, pred, msg, on_error=on_error)
    else:
        # Don't run if not JIT'ing, as without the compiler nothing will be DCE'd.
        return x
