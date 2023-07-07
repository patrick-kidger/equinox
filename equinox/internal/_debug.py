from collections.abc import Callable, Hashable
from typing import Any

import jax
import jax.core
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree

from .._ad import filter_custom_vjp
from .._doc_utils import WithRepr
from .._errors import error_if
from .._filters import combine, filter, is_array, is_array_like, partition
from .._module import Module
from .._pretty_print import pformat_short_array_text, tree_pformat, tree_pprint


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


@_debug_backward_nan.def_fwd
def _debug_backward_nan_fwd(perturbed, x, name, terminate):
    del perturbed
    return debug_backward_nan(x, name, terminate), None


class _LongRepr(Module):
    obj: Any

    def __repr__(self):
        return tree_pformat(self.obj, short_arrays=False)


@_debug_backward_nan.def_bwd
def _debug_backward_nan_bwd(residuals, grad_x, perturbed, x, name, terminate):
    del residuals, perturbed
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


_dce_store = {}


def _register_alive(name: Hashable, tag: object):
    def _register_alive_impl(i, x):
        leaves, _ = _dce_store[name][tag]
        leaves[i.item()] = (x.shape, x.dtype.name)
        return x

    return _register_alive_impl


def store_dce(x: PyTree[Array], name: Hashable = None):
    """Used to check whether an array (or pytree of arrays) is DCE'd.

    `store_dce` must be used within a JIT'd function, and acts as the identity
    function. When the JIT'd function is called, then whether each array got DCE'd or
    not is recorded. This can subsequently be inspected using `inspect_dce`.

    !!! Example:

        ```python
        @jax.jit
        def f(x):
            a, _ = eqxi.store_dce((x**2, x + 1))
            return a

        f(1)
        eqxi.inspect_dce()
        # Found 1 call to `equinox.internal.register_dce`.
        # Entry 0:
        # (i32[], <DCE'd>)
        ```

    **Arguments:**

    - `x`: Any PyTree of JAX arrays.
    - `name`: Optional argument. Any hashable value used to distinguish this call site
        from another call site. If used, then it should be passed to `inspect_dce` to
        print only those entries with this name.

    **Returns:**

    `x` is returned unchanged.
    """
    if not isinstance(jnp.array(1) + 1, jax.core.Tracer):
        raise RuntimeError(
            "`equinox.internal.store_dce` should be used insnide of JIT."
        )
    tag = object()
    leaves, treedef = jtu.tree_flatten(x)
    try:
        _tag_store = _dce_store[name]
    except KeyError:
        _tag_store = _dce_store[name] = {}
    _tag_store[tag] = ({}, treedef)
    leaves = [
        jax.pure_callback(  # pyright: ignore
            _register_alive(name, tag), x, i, x, vectorized=True
        )
        for i, x in enumerate(leaves)
    ]
    return jtu.tree_unflatten(treedef, leaves)


def inspect_dce(name: Hashable = None):
    """Used in conjunction with `equinox.internal.check_dce`; see documentation there.

    Must be called outside of any JIT'd function.

    **Arguments:**

    - `name`: Optional argument. Whatever name was used with `check_dce`.

    **Returns:**

    Nothing. DCE information is printed to stdout.
    """
    if isinstance(jnp.array(1) + 1, jax.core.Tracer):
        raise RuntimeError(
            "`equinox.internal.inspect_dce` should be used outside of JIT."
        )
    try:
        _tag_store = _dce_store[name]
    except KeyError as e:
        raise ValueError(
            "`equinox.internal.inspect_dce` should be called after "
            "`equinox.internal.store_dce` has run."
        ) from e
    new_leaves = []
    maybe_s = "" if len(_tag_store) == 1 else "s"
    print(f"Found {len(_tag_store)} call{maybe_s} to `equinox.internal.register_dce`.")
    for i, (leaves, treedef) in enumerate(_tag_store.values()):
        for j in range(treedef.num_leaves):
            try:
                shape, dtype = leaves[j]
            except KeyError:
                value = "<DCE'd>"
            else:
                value = pformat_short_array_text(shape, dtype)
            new_leaves.append(WithRepr(None, value))
        tree = jtu.tree_unflatten(treedef, new_leaves)
        print(f"Entry {i}:")
        tree_pprint(tree)
