import jax
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.tree_util as jtu

from ..filters import combine, is_array, partition


def announce(name, accepted):
    def announce_impl(x, stack):
        del x
        if stack[0] in accepted:
            if name is not None:
                stack = (name,) + stack
            print(":".join(stack))

    return announce_impl


def hook_transform(x, hook=announce(None, ("impl", "abstract", "mlir"))):
    """Identity function on an arbitrary PyTree. Calls a hook each time it
    is transformed or lowered.

    The default hook is to print the transform stack on impl, abstract,
    and mlir.
    """
    array, nonarray = partition(x, is_array)
    flat, treedef = jtu.tree_flatten(array)
    hook(flat, ("call",))
    flat = _hook_p.bind(*flat, stack=(), hook=hook)
    array = jtu.tree_unflatten(treedef, flat)
    return combine(array, nonarray)


def _impl(*x, stack, hook):
    hook(x, ("impl",) + stack)
    return x


def _abstract(*x, stack, hook):
    hook(x, ("abstract",) + stack)
    return x


def _jvp(p, t, *, stack, hook):
    hook((p, t), ("jvp",) + stack)
    p_out = _hook_p.bind(*p, stack=("jvp_p",) + stack, hook=hook)
    t_out = _hook_p.bind(*t, stack=("jvp_t",) + stack, hook=hook)
    return p_out, t_out


def _transpose(cts, *p, stack, hook):
    stack = ("transpose",) + stack
    hook((cts, p), stack)
    return _hook_p.bind(*cts, stack=stack, hook=hook)


def _batching(p, b, *, stack, hook):
    stack = ("vmap",) + stack
    hook((p, b), stack)
    out = _hook_p.bind(*p, stack=stack, hook=hook)
    return out, b


def _mlir(*x, stack, hook):
    hook(x, ("mlir",) + stack)
    return x


_hook_p = jax.core.Primitive("hook")
_hook_p.multiple_results = True
_hook_p.def_impl(_impl)
_hook_p.def_abstract_eval(_abstract)
ad.primitive_jvps[_hook_p] = _jvp
ad.primitive_transposes[_hook_p] = _transpose
batching.primitive_batchers[_hook_p] = _batching
mlir.register_lowering(_hook_p, mlir.lower_fun(_mlir, multiple_results=True))
