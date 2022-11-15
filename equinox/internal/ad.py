import functools as ft
from typing import Optional

import jax
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.interpreters.xla as xla
import jax.tree_util as jtu
from jaxtyping import PyTree

from ..filters import combine, is_array, partition
from .errors import error_if


_identity = lambda x, *, msg: x


_nondifferentiable_p = jax.core.Primitive("nondifferentiable")


def _nondifferentiable_batch(x, batch_axes, *, msg):
    (x,) = x
    (batch_axes,) = batch_axes
    return nondifferentiable(x, msg=msg), batch_axes


def _nondifferentiable_jvp(primals, tangents, *, msg):
    raise RuntimeError(msg)


_nondifferentiable_p.def_impl(_identity)
_nondifferentiable_p.def_abstract_eval(_identity)
batching.primitive_batchers[_nondifferentiable_p] = _nondifferentiable_batch
if hasattr(xla, "lower_fun"):
    xla.register_translation(
        _nondifferentiable_p,
        xla.lower_fun(_identity, multiple_results=False, new_style=True),
    )
mlir.register_lowering(
    _nondifferentiable_p,
    mlir.lower_fun(_identity, multiple_results=False),
)
ad.primitive_jvps[_nondifferentiable_p] = _nondifferentiable_jvp


def nondifferentiable(
    x: PyTree, *, name: Optional[str] = None, msg: Optional[str] = None
) -> PyTree:
    """
    Consumes a PyTree with arbitrary leaves and returns an identical PyTree.
    If any of the JAX arrays in this PyTree are ever differentiated (in
    forward or reverse mode) then an error will be thrown.
    """
    dynamic, static = partition(x, is_array)
    flat, treedef = jtu.tree_flatten(dynamic)
    if msg is None:
        if name is None:
            name = "This operation"
        msg = f"Unexpected tangent. {name} cannot be autodifferentiated."
    bind = ft.partial(_nondifferentiable_p.bind, msg=msg)
    flat = map(bind, flat)
    return combine(jtu.tree_unflatten(treedef, flat), static)


_nondifferentiable_backward_p = jax.core.Primitive("nondifferentiable_backward")


def _nondifferentiable_backward_batch(x, batch_axes, *, msg):
    (x,) = x
    (batch_axes,) = batch_axes
    return nondifferentiable_backward(x, msg=msg), batch_axes


def _nondifferentiable_backward_jvp(primals, tangents, *, msg):
    (primals,) = primals
    (tangents,) = tangents
    return nondifferentiable_backward(primals, msg=msg), nondifferentiable_backward(
        tangents, msg=msg
    )


def _nondifferentiable_backward_transpose(cts_in, _, *, msg):
    if isinstance(cts_in, ad.Zero):
        return ad.Zero  # the class, not an instance
    else:
        # Unfortunately there are legitimate cases where we get all-zero non-symbolic
        # cotangents, so we have to use a runtime error here instead of just erroring
        # at trace time.
        # This happens when doing something like:
        #
        # x = nondifferentiable_backward(x)
        # x, y = lax.cond(pred, lambda: (x, y), lambda: (x+1, y+1))
        # return y
        return [error_if(cts_in, (cts_in != 0).any(), msg)]


_nondifferentiable_backward_p.def_impl(_identity)
_nondifferentiable_backward_p.def_abstract_eval(_identity)
batching.primitive_batchers[
    _nondifferentiable_backward_p
] = _nondifferentiable_backward_batch
if hasattr(xla, "lower_fun"):
    xla.register_translation(
        _nondifferentiable_backward_p,
        xla.lower_fun(_identity, multiple_results=False, new_style=True),
    )
mlir.register_lowering(
    _nondifferentiable_backward_p,
    mlir.lower_fun(_identity, multiple_results=False),
)
ad.primitive_jvps[_nondifferentiable_backward_p] = _nondifferentiable_backward_jvp
ad.primitive_transposes[
    _nondifferentiable_backward_p
] = _nondifferentiable_backward_transpose


def nondifferentiable_backward(
    x: PyTree, name: Optional[str] = None, msg: Optional[str] = None
) -> PyTree:
    """
    Consumes a PyTree with arbitrary leaves and returns an identical PyTree.
    If any of the JAX arrays in this PyTree are ever differentiated in
    reverse mode then an error will be thrown.
    """
    dynamic, static = partition(x, is_array)
    flat, treedef = jtu.tree_flatten(dynamic)
    if msg is None:
        if name is None:
            name = "This operation"
        msg = f"Unexpected cotangent. {name} cannot be reverse-mode autodifferentiated."
    bind = ft.partial(_nondifferentiable_backward_p.bind, msg=msg)
    flat = map(bind, flat)
    return combine(jtu.tree_unflatten(treedef, flat), static)
