import jax
import jax._src.traceback_util as traceback_util
import jax.core
import jax.lax as lax
from jaxtyping import Array, Bool

from .._unvmap import unvmap_any


traceback_util.register_exclusion(__file__)


def breakpoint_if(pred: Bool[Array, "..."], **kwargs):
    """As `jax.debug.breakpoint`, but only triggers if `pred` is True.

    **Arguments:**

    - `pred`: the predicate for whether to trigger the breakpoint.
    - `**kwargs`: any other keyword arguments to forward to `jax.debug.breakpoint`.

    **Returns:**

    Nothing.
    """
    # We can't just write `jax.debug.breakpoint` for the second branch. For some reason
    # it needs as lambda wrapper.
    token = kwargs.get("token", None)
    return lax.cond(
        unvmap_any(pred), lambda: jax.debug.breakpoint(**kwargs), lambda: token
    )
