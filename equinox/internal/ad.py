import jax
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.interpreters.xla as xla
import jax.tree_util as jtu
from jaxtyping import PyTree

from ..filters import combine, is_array, partition


_identity = lambda x: x


_nondifferentiable_p = jax.core.Primitive("nondifferentiable")


def _nondifferentiable_batch(x, batch_axes):
    (x,) = x
    (batch_axes,) = batch_axes
    return nondifferentiable(x), batch_axes


def _nondifferentiable_jvp(primals, tangents):
    raise RuntimeError(
        "Unexpected tangent. This operation cannot be autodifferentiated."
    )


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


def nondifferentiable(x: PyTree) -> PyTree:
    dynamic, static = partition(x, is_array)
    flat, treedef = jtu.tree_flatten(dynamic)
    flat = map(_nondifferentiable_p.bind, flat)
    return combine(jtu.tree_unflatten(treedef, flat), static)


_nondifferentiable_backward_p = jax.core.Primitive("nondifferentiable_backward")


def _nondifferentiable_backward_batch(x, batch_axes):
    (x,) = x
    (batch_axes,) = batch_axes
    return nondifferentiable_backward(x), batch_axes


def _nondifferentiable_backward_jvp(primals, tangents):
    (primals,) = primals
    (tangents,) = tangents
    return nondifferentiable_backward(primals), nondifferentiable_backward(tangents)


def _nondifferentiable_backward_transpose(cts_in, _):
    if isinstance(cts_in, ad.Zero):
        return ad.Zero  # the class, not an instance
    else:
        raise RuntimeError(
            "Unexpected cotangent. This operation cannot be reverse-mode autodifferentiated."
        )


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


def nondifferentiable_backward(x: PyTree) -> PyTree:
    dynamic, static = partition(x, is_array)
    flat, treedef = jtu.tree_flatten(dynamic)
    flat = map(_nondifferentiable_backward_p.bind, flat)
    return combine(jtu.tree_unflatten(treedef, flat), static)
