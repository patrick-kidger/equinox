from typing import Any, Callable

import jax
import jax.core
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.numpy as jnp
import jax.tree_util as jtu

from .._ad import filter_custom_vjp
from .._filters import combine, filter, is_array, is_array_like, partition
from .._module import Module
from .._pretty_print import tree_pformat
from ._errors import error_if


def announce_transform(
    x, name=None, intermediates=False, announce: Callable[[str], Any] = print
):
    """Identity function on an arbitrary PyTree. Announces each time it is parsed as
    part of a jaxpr.
    """
    if name is None:
        name = ""
    else:
        name = f"({name}) "
    array, nonarray = partition(x, is_array)
    flat, treedef = jtu.tree_flatten(array)
    flat = announce_jaxpr_p.bind(
        *flat, stack=(), name=name, intermediates=intermediates, announce=announce
    )
    array = jtu.tree_unflatten(treedef, flat)
    return combine(array, nonarray)


def _impl(*x, stack, name, intermediates, announce):
    del intermediates
    stack = stack + ("impl,")
    announce(name + ":".join(stack))
    return x


def _abstract(*x, stack, name, intermediates, announce):
    del intermediates
    stack = stack + ("abstract",)
    announce(name + ":".join(stack))
    return x


def _jvp(p, t, *, stack, name, intermediates, announce):
    if intermediates:
        _stack = stack + ("jvp",)
        announce(name + ":".join(_stack))
    p_stack = stack + ("jvp_p",)
    t_stack = stack + ("jvp_t",)
    p_out = announce_jaxpr_p.bind(
        *p, stack=p_stack, name=name, intermediates=intermediates, announce=announce
    )
    t_out = announce_jaxpr_p.bind(
        *t, stack=t_stack, name=name, intermediates=intermediates, announce=announce
    )
    return p_out, t_out


def _transpose(ct, p, *, stack, name, intermediates, announce):
    stack = stack + ("transpose",)
    if intermediates:
        announce(name + ":".join(stack))
    return announce_jaxpr_p.bind(
        *ct, stack=stack, name=name, intermediates=intermediates, announce=announce
    )


def _batching(p, b, *, stack, name, intermediates, announce):
    stack = stack + ("vmap",)
    if intermediates:
        announce(name + ":".join(stack))
    out = announce_jaxpr_p.bind(
        *p, stack=stack, name=name, intermediates=intermediates, announce=announce
    )
    return out, b


def _mlir(*x, stack, name, intermediates, announce):
    del intermediates
    stack = stack + ("mlir",)
    announce(name + ":".join(stack))
    return x


announce_jaxpr_p = jax.core.Primitive("announce_jaxpr")
announce_jaxpr_p.multiple_results = True
announce_jaxpr_p.def_impl(_impl)
announce_jaxpr_p.def_abstract_eval(_abstract)
ad.primitive_jvps[announce_jaxpr_p] = _jvp
ad.primitive_transposes[announce_jaxpr_p] = _transpose
batching.primitive_batchers[announce_jaxpr_p] = _batching
mlir.register_lowering(announce_jaxpr_p, mlir.lower_fun(_mlir, multiple_results=True))


def debug_backward_nan(x, name=None, terminate=True):
    """Debug NaNs that only occur on the backward pass.

    **Arguments:**

    - `x`: a variable to intercept.
    - `name`: an optional name to appear in printed debug statements.
    - `terminate`: whether to halt the computation if a NaN cotangent is found.

    **Returns:**

    The `x` argument is returned unchanged.

    As a side-effect, both the primal and the cotangent for `x` will be printed out
    during the backward pass.
    """
    return _debug_backward_nan(x, name, terminate)


@filter_custom_vjp
def _debug_backward_nan(x, name, terminate):
    return x


def _debug_backward_nan_fwd(x, name, terminate):
    return debug_backward_nan(x, name, terminate), None


class _LongRepr(Module):
    obj: Any

    def __repr__(self):
        return tree_pformat(self.obj, short_arrays=False)


def _debug_backward_nan_bwd(_, grad_x, x, name, terminate):
    msg = "   primals={x}\ncotangents={grad_x}"
    if name is not None:
        msg = f"{name}:\n" + msg
    jax.debug.print(  # pyright: ignore
        msg, x=_LongRepr(x), grad_x=_LongRepr(grad_x), ordered=True
    )
    if terminate:
        nans = [
            jnp.isnan(a).any() for a in jtu.tree_leaves(filter(grad_x, is_array_like))
        ]
        grad_x = error_if(grad_x, jnp.any(jnp.stack(nans)), "Encountered NaN")
    return grad_x


_debug_backward_nan.defvjp(_debug_backward_nan_fwd, _debug_backward_nan_bwd)
