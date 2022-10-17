import jax
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.tree_util as jtu

from ..filters import combine, is_array, partition


def announce_jaxpr(x, name):
    """Identity function on an arbitrary PyTree. Announces each time it is parsed as part of a jaxpr."""
    array, nonarray = partition(x, is_array)
    flat, treedef = jtu.tree_flatten(array)
    flat = _announce_jaxpr_p.bind(*flat, stack=(), name=name)
    array = jtu.tree_unflatten(treedef, flat)
    return combine(array, nonarray)


def _impl(*x, stack, name):
    stack = (name, "impl") + stack
    print(":".join(stack))
    return x


def _abstract(*x, stack, name):
    stack = (name, "abstract") + stack
    print(":".join(stack))
    return x


def _jvp(p, t, *, stack, name):
    _stack = (name, "jvp") + stack
    print(":".join(_stack))
    p_stack = ("jvp_p",) + stack
    t_stack = ("jvp_t",) + stack
    return _announce_jaxpr_p.bind(*p, stack=p_stack, name=name), _announce_jaxpr_p.bind(
        *t, stack=t_stack, name=name
    )


def _transpose(ct, p, *, stack, name):
    _stack = (name, "transpose") + stack
    stack = ("transpose",) + stack
    print(":".join(_stack))
    return _announce_jaxpr_p.bind(*ct, stack=stack, name=name)


def _batching(p, b, *, stack, name):
    _stack = (name, "vmap") + stack
    stack = ("vmap",) + stack
    print(":".join(_stack))
    return _announce_jaxpr_p.bind(*p, stack=stack, name=name), b


def _mlir(*x, stack, name):
    stack = (name, "mlir") + stack
    print(":".join(stack))
    return x


_announce_jaxpr_p = jax.core.Primitive("announce_jaxpr")
_announce_jaxpr_p.multiple_results = True
_announce_jaxpr_p.def_impl(_impl)
_announce_jaxpr_p.def_abstract_eval(_abstract)
ad.primitive_jvps[_announce_jaxpr_p] = _jvp
ad.primitive_transposes[_announce_jaxpr_p] = _transpose
batching.primitive_batchers[_announce_jaxpr_p] = _batching
mlir.register_lowering(_announce_jaxpr_p, mlir.lower_fun(_mlir, multiple_results=True))
