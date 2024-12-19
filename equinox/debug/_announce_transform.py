from collections.abc import Callable
from typing import Any

import jax
import jax.extend.core
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.tree_util as jtu

from .._filters import combine, is_array, partition


def announce_transform(
    x, name=None, intermediates=False, announce: Callable[[str], Any] = print
):
    """Identity function on an arbitrary PyTree. Announces each time a JAX transform is
    applied (grad, vmap, etc.).

    !!! warning

        This API is not stable. It should be used for one-off debugging purposes only.

    **Arguments:**

    - `x`: a variable to intercept.
    - `intermediates`: whether to include intermediate transforms, that haven't yet
        finished being transformed. E.g. if
        `intermediates=False`, then `jit(vmap(...))` will print out
        ```
        vmap:abstract
        vmap:mlir`
        ```
        whilst if `intermediates=True`, then `jit(vmap(...))` will print out
        ```
        vmap
        vmap:abstract
        vmap:mlir`
        ```
    - `announce`: the function to announce via. Defaults to just `print`.

    **Returns:**

    The `x` argument is returned unchanged.

    As a side-effect, the transforms applied to `x` will be printed out.
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
    stack = stack + ("impl",)
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
    t_nonzero = [ti for ti in t if type(ti) is not ad.Zero]
    if len(t_nonzero) > 0:
        t_nonzero_out = announce_jaxpr_p.bind(
            *t_nonzero,
            stack=t_stack,
            name=name,
            intermediates=intermediates,
            announce=announce,
        )
    else:
        t_nonzero_out = []
    t_nonzero_out = iter(t_nonzero_out)
    t_out = [ti if type(ti) is ad.Zero else next(t_nonzero_out) for ti in t]
    assert next(t_nonzero_out, None) is None
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


announce_jaxpr_p = jax.extend.core.Primitive("announce_jaxpr")
announce_jaxpr_p.multiple_results = True
announce_jaxpr_p.def_impl(_impl)
announce_jaxpr_p.def_abstract_eval(_abstract)
ad.primitive_jvps[announce_jaxpr_p] = _jvp
ad.primitive_transposes[announce_jaxpr_p] = _transpose
batching.primitive_batchers[announce_jaxpr_p] = _batching
mlir.register_lowering(announce_jaxpr_p, mlir.lower_fun(_mlir, multiple_results=True))
