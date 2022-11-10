import jax
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.tree_util as jtu
import numpy as np

from ..callback import filter_pure_callback
from ..filters import combine, filter, is_array, partition
from ..grad import filter_custom_vjp
from ..pretty_print import tree_pformat


def announce_transform(x, name=None, intermediates=False, announce=print):
    """Identity function on an arbitrary PyTree. Announces each time it is parsed as part of a jaxpr."""
    array, nonarray = partition(x, is_array)
    flat, treedef = jtu.tree_flatten(array)
    flat = _announce_jaxpr_p.bind(
        *flat, stack=(), name=name, intermediates=intermediates, announce=announce
    )
    array = jtu.tree_unflatten(treedef, flat)
    return combine(array, nonarray)


def _impl(*x, stack, name, intermediates, announce):
    del intermediates
    if name is None:
        stack = ("impl",) + stack
    else:
        stack = (name, "impl") + stack
    announce(":".join(stack))
    return x


def _abstract(*x, stack, name, intermediates, announce):
    del intermediates
    if name is None:
        stack = ("abstract",) + stack
    else:
        stack = (name, "abstract") + stack
    announce(":".join(stack))
    return x


def _jvp(p, t, *, stack, name, intermediates, announce):
    if intermediates:
        if name is None:
            _stack = ("jvp",) + stack
        else:
            _stack = (name, "jvp") + stack
        announce(":".join(_stack))
    p_stack = ("jvp_p",) + stack
    t_stack = ("jvp_t",) + stack
    p_out = _announce_jaxpr_p.bind(
        *p, stack=p_stack, name=name, intermediates=intermediates, announce=announce
    )
    t_out = _announce_jaxpr_p.bind(
        *t, stack=t_stack, name=name, intermediates=intermediates, announce=announce
    )
    return p_out, t_out


def _transpose(ct, p, *, stack, name, intermediates, announce):
    if intermediates:
        if name is None:
            _stack = ("transpose",) + stack
        else:
            _stack = (name, "transpose") + stack
        announce(":".join(_stack))
    stack = ("transpose",) + stack
    return _announce_jaxpr_p.bind(
        *ct, stack=stack, name=name, intermediates=intermediates, announce=announce
    )


def _batching(p, b, *, stack, name, intermediates, announce):
    if intermediates:
        if name is None:
            _stack = ("vmap",) + stack
        else:
            _stack = (name, "vmap") + stack
        announce(":".join(_stack))
    stack = ("vmap",) + stack
    out = _announce_jaxpr_p.bind(
        *p, stack=stack, name=name, intermediates=intermediates, announce=announce
    )
    return out, b


def _mlir(*x, stack, name, intermediates, announce):
    del intermediates
    if name is None:
        stack = ("mlir",) + stack
    else:
        stack = (name, "mlir") + stack
    announce(":".join(stack))
    return x


_announce_jaxpr_p = jax.core.Primitive("announce_jaxpr")
_announce_jaxpr_p.multiple_results = True
_announce_jaxpr_p.def_impl(_impl)
_announce_jaxpr_p.def_abstract_eval(_abstract)
ad.primitive_jvps[_announce_jaxpr_p] = _jvp
ad.primitive_transposes[_announce_jaxpr_p] = _transpose
batching.primitive_batchers[_announce_jaxpr_p] = _batching
mlir.register_lowering(_announce_jaxpr_p, mlir.lower_fun(_mlir, multiple_results=True))


def debug_backward_nan(x, name=None, terminate=True):
    return _debug_backward_nan(x, name, terminate)


@filter_custom_vjp
def _debug_backward_nan(x, name, terminate):
    return x


def _debug_backward_nan_fwd(x, name, terminate):
    return debug_backward_nan(x, name, terminate), None


def _debug_backward_nan_bwd(_, grad_x, x, name, terminate):
    def _callback(_x, _grad_x):
        primals = tree_pformat(_x, short_arrays=False)
        cotangents = tree_pformat(_grad_x, short_arrays=False)
        msg = f"   primals={primals}\ncotangents={cotangents}"
        if name is not None:
            msg = f"{name}:\n" + msg
        print(msg)
        if terminate:
            arrays = jtu.tree_leaves(filter(_grad_x, is_array))
            if any(np.isnan(a).any() for a in arrays):
                raise RuntimeError("Encountered NaN")
        return _grad_x

    grad_x = filter_pure_callback(
        _callback, x, grad_x, result_shape_dtypes=grad_x, vectorized=True
    )
    return grad_x


_debug_backward_nan.defvjp(_debug_backward_nan_fwd, _debug_backward_nan_bwd)
