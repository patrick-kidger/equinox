import functools as ft
from typing import Optional

import jax
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.tree_util as jtu
from jaxtyping import Array, PyTree

from ..filters import combine, is_array, partition
from .errors import error_if


_identity = lambda x, *, msg: x


@ft.partial(jax.custom_jvp, nondiff_argnums=(0,))
def _nondifferentiable(msg: str, x: PyTree[Array]):
    return x


@_nondifferentiable.defjvp
def _nondifferentiable_jvp(msg: str, primals, tangents):
    raise RuntimeError(msg)


def nondifferentiable(
    x: PyTree, *, name: Optional[str] = None, msg: Optional[str] = None
) -> PyTree:
    """
    Consumes a PyTree with arbitrary leaves and returns an identical PyTree.
    If any of the JAX arrays in this PyTree are ever differentiated (in
    forward or reverse mode) then an error will be thrown.
    """
    dynamic, static = partition(x, is_array)
    if msg is None:
        if name is None:
            name = "This operation"
        msg = f"Unexpected tangent. {name} cannot be autodifferentiated."
    dynamic = _nondifferentiable(msg, dynamic)
    return combine(dynamic, static)


nondifferentiable_backward_p = jax.core.Primitive("nondifferentiable_backward")


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


nondifferentiable_backward_p.def_impl(_identity)
nondifferentiable_backward_p.def_abstract_eval(_identity)
ad.primitive_jvps[nondifferentiable_backward_p] = _nondifferentiable_backward_jvp
ad.primitive_transposes[
    nondifferentiable_backward_p
] = _nondifferentiable_backward_transpose
batching.primitive_batchers[
    nondifferentiable_backward_p
] = _nondifferentiable_backward_batch
mlir.register_lowering(
    nondifferentiable_backward_p,
    mlir.lower_fun(_identity, multiple_results=False),
)


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
    bind = ft.partial(nondifferentiable_backward_p.bind, msg=msg)
    flat = map(bind, flat)
    return combine(jtu.tree_unflatten(treedef, flat), static)
