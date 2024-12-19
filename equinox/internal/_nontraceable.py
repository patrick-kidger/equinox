"""This module provides operations to assert that a particular value is always
unbatched, or not differentiated, etc.
"""

import functools as ft
from typing import Optional

import jax
import jax.extend.core
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree

from .._ad import nondifferentiable as nondifferentiable  # public re-export
from .._errors import error_if
from .._filters import combine, is_array, partition


_nontraceable_impl = lambda x, *, name: x


def _make_error(opname):
    def _error(*args, name):
        raise RuntimeError(f"Detected {opname} of {name}")

    return _error


nontraceable_p = jax.extend.core.Primitive("nontraceable")
nontraceable_p.def_impl(_nontraceable_impl)
nontraceable_p.def_abstract_eval(_nontraceable_impl)
ad.primitive_jvps[nontraceable_p] = _make_error("differentiation")
ad.primitive_transposes[nontraceable_p] = _make_error("transposition")
batching.primitive_batchers[nontraceable_p] = _make_error("batching")
mlir.register_lowering(
    nontraceable_p, mlir.lower_fun(_nontraceable_impl, multiple_results=False)
)


def nontraceable(x, *, name="nontraceable operation"):
    """Identity function, which raises an error if it is transformed in any way. (i.e.
    in `jax.grad`, `jax.vmap` etc.)

    This is useful at the end of the `impl` rule for higher-order final-style
    primitives, for checking that no other tracers were captured via closure.
    """
    dynamic, static = partition(x, is_array)
    bind = ft.partial(nontraceable_p.bind, name=name)
    dynamic = jtu.tree_map(bind, dynamic)
    return combine(dynamic, static)


nondifferentiable_backward_p = jax.extend.core.Primitive("nondifferentiable_backward")


def _nondifferentiable_backward_batch(x, batch_axes, *, msg, symbolic):
    (x,) = x
    (batch_axes,) = batch_axes
    return nondifferentiable_backward(x, msg=msg, symbolic=symbolic), batch_axes


def _nondifferentiable_backward_jvp(primals, tangents, *, msg, symbolic):
    (primals,) = primals
    (tangents,) = tangents
    primal_out = nondifferentiable_backward(primals, msg=msg, symbolic=symbolic)
    tangent_out = nondifferentiable_backward(tangents, msg=msg, symbolic=symbolic)
    return primal_out, tangent_out


def _nondifferentiable_backward_transpose(cts_in, _, *, msg, symbolic):
    if isinstance(cts_in, ad.Zero):
        return ad.Zero  # the class, not an instance
    else:
        if symbolic:
            raise RuntimeError(msg)
        else:
            # Unfortunately there are legitimate cases where we get all-zero
            # non-symbolic cotangents, so we have to use a runtime error here instead of
            # just erroring at trace time.
            # This happens when doing something like:
            #
            # x = nondifferentiable_backward(x)
            # x, y = lax.cond(pred, lambda: (x, y), lambda: (x+1, y+1))
            # return y
            return [error_if(cts_in, (cts_in != 0).any(), msg)]


_nondifferentiable_backward_impl = lambda x, *, msg, symbolic: x


nondifferentiable_backward_p.def_impl(_nondifferentiable_backward_impl)
nondifferentiable_backward_p.def_abstract_eval(_nondifferentiable_backward_impl)
ad.primitive_jvps[nondifferentiable_backward_p] = _nondifferentiable_backward_jvp
ad.primitive_transposes[nondifferentiable_backward_p] = (
    _nondifferentiable_backward_transpose
)
batching.primitive_batchers[nondifferentiable_backward_p] = (
    _nondifferentiable_backward_batch
)
mlir.register_lowering(
    nondifferentiable_backward_p,
    mlir.lower_fun(_nondifferentiable_backward_impl, multiple_results=False),
)


def nondifferentiable_backward(
    x: PyTree,
    name: Optional[str] = None,
    msg: Optional[str] = None,
    symbolic: bool = True,
) -> PyTree:
    """Identity function. Raises an error if it is differentiated in reverse mode."""
    dynamic, static = partition(x, is_array)
    flat, treedef = jtu.tree_flatten(dynamic)
    if msg is None:
        if name is None:
            name = "This operation"
        msg = f"Unexpected cotangent. {name} cannot be reverse-mode autodifferentiated."
    bind = ft.partial(nondifferentiable_backward_p.bind, msg=msg, symbolic=symbolic)
    flat = map(bind, flat)
    return combine(jtu.tree_unflatten(treedef, flat), static)


def _cannot_batch(x, b, *, msg, allow_constant_across_batch):
    (x,) = x
    (b,) = b
    if b is batching.not_mapped:
        return x, b
    else:
        if allow_constant_across_batch:
            x = error_if(x, jnp.min(x, axis=b) != jnp.max(x, axis=b), msg)
            return jnp.take(x, 0, axis=b), batching.not_mapped
        else:
            raise ValueError(msg)


nonbatchable_p = jax.extend.core.Primitive("nonbatchable")
nonbatchable_p.def_impl(lambda x, *, msg, allow_constant_across_batch: x)
nonbatchable_p.def_abstract_eval(lambda x, *, msg, allow_constant_across_batch: x)
batching.primitive_batchers[nonbatchable_p] = _cannot_batch
mlir.register_lowering(
    nonbatchable_p,
    mlir.lower_fun(
        lambda x, *, msg, allow_constant_across_batch: x, multiple_results=False
    ),
)


def nonbatchable(
    x: PyTree,
    *,
    name: Optional[str] = None,
    msg: Optional[str] = None,
    allow_constant_across_batch: bool = False,
) -> PyTree:
    """Identity function. Raises a trace-time assert if it is batched."""
    dynamic, static = partition(x, is_array)
    flat, treedef = jtu.tree_flatten(dynamic)
    if msg is None:
        if name is None:
            name = "This operation"
        if allow_constant_across_batch:
            msg = (
                f"Nonconstant batch. {name} has received a batch of values that were "
                "expected to be constant. This is probably an internal error in the "
                "library you are using."
            )
        else:
            msg = (
                f"Unexpected batch tracer. {name} cannot be vmap'd. This is probably "
                "an internal error in the library you are using."
            )
    bind = ft.partial(
        nonbatchable_p.bind,
        msg=msg,
        allow_constant_across_batch=allow_constant_across_batch,
    )
    flat = map(bind, flat)
    return combine(jtu.tree_unflatten(treedef, flat), static)
