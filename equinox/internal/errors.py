from typing import Sequence, Union

import jax
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.lax as lax
import jax.tree_util as jtu
from jaxtyping import Array, Bool, Int, PyTree

from ..filters import combine, is_array_like, partition
from .unvmap import unvmap_any, unvmap_max


def _error_impl(pred, index, *x, msgs):
    def raises(_index):
        raise RuntimeError(msgs[_index.item()])

    struct = jax.eval_shape(lambda: x)
    return lax.cond(pred, lambda: jax.pure_callback(raises, struct, index), lambda: x)


def _error_abstract(pred, index, *x, msgs):
    return x


def _error_jvp(primals, tangents, *, msgs):
    _, _, *tx = tangents
    return _branched_error_p.bind(*primals, msgs=msgs), tx


def _error_transpose(cts, pred, index, *x, msgs):
    return (None, None) + cts


def _error_batching(inputs, batch_axes, *, msgs):
    pred_bdim, index_bdim, *xs_bdim = batch_axes
    assert pred_bdim is batching.not_mapped
    assert index_bdim is batching.not_mapped
    return _branched_error_p.bind(*inputs, msgs=msgs), xs_bdim


_branched_error_p = jax.core.Primitive("branched_error")
_branched_error_p.multiple_results = True
_branched_error_p.def_impl(_error_impl)
_branched_error_p.def_abstract_eval(_error_abstract)
ad.primitive_jvps[_branched_error_p] = _error_jvp
ad.primitive_transposes[_branched_error_p] = _error_transpose
batching.primitive_batchers[_branched_error_p] = _error_batching
mlir.register_lowering(
    _branched_error_p, mlir.lower_fun(_error_impl, multiple_results=True)
)


def error_if(
    x: PyTree,
    pred: Union[bool, Bool[Array, "..."]],
    msg: str,
) -> PyTree:
    """Throws an error based on runtime values.
    Works even under JIT.

    Note that this probably won't work on TPU (a dummy value
    is propagated through the computation instead of an error
    being thrown).

    The first argument `x` is returned unchanged, and used to determine
    where the error check happens in the overall computation. If the
    return value is ever unused then the error check may be DCE'd.

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
    pred: Union[bool, Bool[Array, "..."]],
    index: Union[int, Int[Array, "..."]],
    msgs: Sequence[str],
) -> PyTree:
    """As [`equinox.internal.error_if`][], but will raise one of
    several messages depending on the value of `index`.
    """

    with jax.ensure_compile_time_eval():
        pred = unvmap_any(pred)
        if not isinstance(pred, jax.core.Tracer) and pred.item() is False:
            return x

    index = unvmap_max(index)
    dynamic_x, static_x = partition(x, is_array_like)
    flat, treedef = jtu.tree_flatten(dynamic_x)
    if len(flat) == 0:
        raise ValueError("No array-likes to thread error on to")
    # Use a primitive to put the lax.cond inside the impl rule.
    # This is needed as lax.cond will unnecessarily promote symbolic
    # zeros to non-symbolic-zeros, and we'd really like to avoid that.
    flat = _branched_error_p.bind(pred, index, *flat, msgs=msgs)
    dynamic_x = jtu.tree_unflatten(treedef, flat)
    return combine(dynamic_x, static_x)
