import jax
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.tree_util as jtu

from ..filters import combine, is_array, partition


def announce_jaxpr(x, name=None, intermediates=False, announce=print):
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
