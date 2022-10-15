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
    """Throws an error based on runtime values.
    Works even under JIT.

    Note that this probably won't work on TPU (a dummy value
    is propagated through the computation instead of an error
    being thrown.

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
    index: Union[int, Int[Array, ""]],
    msgs: Sequence[str],
) -> PyTree:
    """As [`equinox.internal.error_if`][], but will raise one of
    several messages depending on the value of `index`.
    """

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
