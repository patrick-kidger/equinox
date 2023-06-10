from collections.abc import Sequence
from typing import cast

import jax
import jax.core
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.lax as lax
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Int, PyTree

from ._filters import combine, is_array_like, partition
from ._unvmap import unvmap_any, unvmap_max


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


def _error_impl(pred, index, *x, msgs):
    def raises(_index):
        raise RuntimeError(msgs[_index.item()])

    def tpu_msg(_out, _index):
        msg = msgs[_index.item()]
        # `print` doesn't work; nor does `jax.debug.print`.
        # But both `input` and `jax.debug.breakpoint` do. The former allows us to
        # actually display something to the user.
        input(msg + _tpu_msg)
        # We do the tree_map inside the pure_callback, not outside, so that `out` has a
        # data dependency and doesn't get optimised out.
        return jtu.tree_map(_nan_like, _out)

    def callback():
        out = jax.pure_callback(raises, struct, index)  # pyright: ignore
        # If we make it this far then we're on the TPU, which squelches runtime errors
        # and returns dummy values instead.
        # Fortunately, we're able to outsmart it!
        return jax.pure_callback(tpu_msg, struct, out, index)  # pyright: ignore

    struct = jax.eval_shape(lambda: x)
    return lax.cond(pred, callback, lambda: x)


def _error_abstract(pred, index, *x, msgs):
    return x


def _error_jvp(primals, tangents, *, msgs):
    _, _, *tx = tangents
    return branched_error_p.bind(*primals, msgs=msgs), tx


def _error_transpose(cts, pred, index, *x, msgs):
    return [None, None] + cts


def _error_batching(inputs, batch_axes, *, msgs):
    pred_bdim, index_bdim, *xs_bdim = batch_axes
    assert pred_bdim is batching.not_mapped
    assert index_bdim is batching.not_mapped
    return branched_error_p.bind(*inputs, msgs=msgs), tuple(xs_bdim)


branched_error_p = jax.core.Primitive("branched_error")
branched_error_p.multiple_results = True
branched_error_p.def_impl(_error_impl)
branched_error_p.def_abstract_eval(_error_abstract)
ad.primitive_jvps[branched_error_p] = _error_jvp
ad.primitive_transposes[branched_error_p] = _error_transpose
batching.primitive_batchers[branched_error_p] = _error_batching
mlir.register_lowering(
    branched_error_p, mlir.lower_fun(_error_impl, multiple_results=True)
)


def error_if(
    x: PyTree,
    pred: Bool[ArrayLike, "..."],
    msg: str,
) -> PyTree:
    """Throws an error based on runtime values. Works even under JIT.

    **Arguments:**

    - `x`: will be returned unchanged; used to determine where the error check happens
        in the overall computation. Can be any PyTree; must contain at least one array.
    - `pred`: a boolean for whether to raise an error. Can be an array of bools; an
        error will be raised if any of them are `True`. If vmap'd then an error will be
        raised if any batch element has `True`.
    - `msg`: the string to display as an error message.

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
    return branched_error_if(x, pred, 0, [msg])


def branched_error_if(
    x: PyTree,
    pred: Bool[ArrayLike, "..."],
    index: Int[ArrayLike, "..."],
    msgs: Sequence[str],
) -> PyTree:
    """As [`equinox.internal.error_if`][], but will raise one of
    several `msgs` depending on the value of `index`.
    """

    with jax.ensure_compile_time_eval():
        pred = unvmap_any(pred)
        index = unvmap_max(index)
        if not isinstance(pred, jax.core.Tracer):
            if pred.item():
                if not isinstance(index, jax.core.Tracer):
                    if isinstance(index, Array):
                        index = index.item()
                    index = cast(int, index)
                    raise RuntimeError(msgs[index])
                # else defer error to runtime, when the index is known.
            else:
                return x

    dynamic_x, static_x = partition(x, is_array_like)
    flat, treedef = jtu.tree_flatten(dynamic_x)
    if len(flat) == 0:
        raise ValueError("No array-likes to thread error on to")
    # Use a primitive to put the lax.cond inside the impl rule.
    # This is needed as lax.cond will unnecessarily promote symbolic
    # zeros to non-symbolic-zeros, and we'd really like to avoid that.
    flat = branched_error_p.bind(pred, index, *flat, msgs=msgs)
    dynamic_x = jtu.tree_unflatten(treedef, flat)
    return combine(dynamic_x, static_x)
