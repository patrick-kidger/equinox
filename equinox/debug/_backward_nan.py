from typing import Any

import jax._src.traceback_util as traceback_util
import jax.numpy as jnp
import jax.tree_util as jtu

from ._print import print as debug_print
from .._ad import filter_custom_vjp
from .._errors import error_if
from .._filters import filter, is_array_like
from .._module import Module
from .._pretty_print import tree_pformat


traceback_util.register_exclusion(__file__)


def backward_nan(x, name=None, print=False, terminate=True):
    """Debug NaNs that only occur on the backward pass.

    **Arguments:**

    - `x`: a variable to intercept.
    - `name`: an optional name to appear in printed debug statements.
    - `print`: whether to print out the tangent and cotangent.
    - `terminate`: whether to halt the computation if a NaN cotangent is found. If
        `True` then an error will be raised via [`equinox.error_if`][]. (So you can
        also arrange for a breakpoint to trigger by setting `EQX_ON_ERROR`
        appropriately.)

    **Returns:**

    The `x` argument is returned unchanged.
    """
    return _backward_nan(x, name, print, terminate)


@filter_custom_vjp
def _backward_nan(x, name, print, terminate):
    return x


@_backward_nan.def_fwd
def _backward_nan_fwd(perturbed, x, name, print, terminate):
    del perturbed
    return backward_nan(x, name, print, terminate), None


class _LongRepr(Module):
    obj: Any

    def __repr__(self):
        return tree_pformat(self.obj, short_arrays=False)


@_backward_nan.def_bwd
def _backward_nan_bwd(residuals, grad_x, perturbed, x, name, print, terminate):
    del residuals, perturbed
    msg = "   primals={x}\ncotangents={grad_x}"
    if name is not None:
        msg = f"{name}:\n" + msg
    if print:
        grad_x = debug_print(grad_x, msg, x=_LongRepr(x), grad_x=_LongRepr(grad_x))
    if terminate:
        nans = [
            jnp.isnan(a).any() for a in jtu.tree_leaves(filter(grad_x, is_array_like))
        ]
        grad_x = error_if(grad_x, jnp.any(jnp.stack(nans)), f"Encountered NaN\n{msg}")
    return grad_x
