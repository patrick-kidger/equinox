from typing import Sequence, Union

import jax
import jax.lax as lax
from jaxtyping import Array, Bool, Int, PyTree

from ..filters import combine, is_array, partition
from .unvmap import unvmap_any


def error_if(
    x: PyTree,
    pred: Union[bool, Bool[Array, "..."]],
    msg: str,
) -> PyTree:
    """For use as part of validating inputs.
    Works even under JIT.

    Example:
        @jax.jit
        def f(x):
            x = error_if(x, x < 0, "x must be >= 0")
            # ...use x in your computation...
            return x

        f(jax.numpy.array(-1))
    """
    return branched_error_if(x, pred, 0, [msg])


def branched_error_if(
    x: PyTree,
    pred: Union[bool, Bool[Array, "..."]],
    index: Union[int, Int[Array, ""]],
    msgs: Sequence[str],
) -> PyTree:
    def raises(_, _index):
        raise RuntimeError(msgs[_index.item()])

    pred = unvmap_any(pred)
    dynamic_x, static_x = partition(x, is_array)
    struct = jax.eval_shape(lambda: dynamic_x)
    dynamic_x = lax.cond(
        pred,
        lambda: jax.pure_callback(raises, struct, dynamic_x, index, vectorized=True),
        lambda: dynamic_x,
    )
    return combine(dynamic_x, static_x)
