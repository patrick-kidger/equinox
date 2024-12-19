"""Defines a new transformation, "finalisation", which replaces all custom primitives
with the results of their `impl` rule.

This is useful prior to e.g. ONNX export, to replace all custom primitives (e.g.
`equinox.internal.unvmap_any`) with the JAX operations that they wrap.

Most end users will find `finalise_fn` and `finalise_make_jaxpr` to be the most useful
operations provided by this module.

Library authors may wish to register their primitives with `primitive_finalisations`.

!!! warning

    Note that you should not perform any other JAX transformations -- e.g. `jax.vmap`,
    `jax.grad` etc. -- after finalisation. The result may be silently incorrect.
"""

import functools as ft
from collections.abc import Callable
from typing import Any, cast, Literal, overload, Union

import jax
import jax.core
import jax.custom_derivatives
import jax.extend.core
import jax.tree_util as jtu
from jaxtyping import PyTree


def _safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f"length mismatch: {list(map(len, args))}"
    return list(map(f, *args))


def _maybe_finalise_jaxpr(val: Any):
    is_open_jaxpr = False
    if isinstance(val, jax.extend.core.Jaxpr):
        if len(val.constvars) == 0:
            is_open_jaxpr = True
            val = jax.extend.core.ClosedJaxpr(val, [])
        else:
            return val
    if isinstance(val, jax.extend.core.ClosedJaxpr):
        val = finalise_jaxpr(val)
    if is_open_jaxpr:
        val = val.jaxpr
    return val


def _finalise_jaxprs_in_params(params):
    new_params = {}
    for key, val in params.items():
        if type(val) is tuple:  # not isinstance to avoid breaking namedtuples
            val = tuple(_maybe_finalise_jaxpr(v) for v in val)
        else:
            val = _maybe_finalise_jaxpr(val)
        new_params[key] = val
    return new_params


def _default_finalisation(prim: jax.extend.core.Primitive, *args, **kwargs):
    return prim.bind(*args, **kwargs)


def _impl_finalisation(prim: jax.extend.core.Primitive, *args, **kwargs):
    return prim.impl(*args, **kwargs)


primitive_finalisations = {}


def register_impl_finalisation(prim: jax.extend.core.Primitive):
    primitive_finalisations[prim] = ft.partial(_impl_finalisation, prim)


def finalise_eval_jaxpr(jaxpr: jax.extend.core.Jaxpr, consts, *args):
    """As jax.core.eval_jaxpr, but finalises (typically by calling `impl` rather than
    `bind` for custom primitives).
    """

    def read(v: jax.core.Atom) -> Any:
        return v.val if isinstance(v, jax.extend.core.Literal) else env[v]

    def write(v: jax.extend.core.Var, val: Any) -> None:
        env[v] = val

    env: dict[jax.extend.core.Var, Any] = {}
    _safe_map(write, jaxpr.constvars, consts)
    _safe_map(write, jaxpr.invars, args)
    for eqn in jaxpr.eqns:
        params = _finalise_jaxprs_in_params(eqn.params)
        subfuns, bind_params = eqn.primitive.get_bind_params(params)
        try:
            call = primitive_finalisations[eqn.primitive]
        except KeyError:
            call = ft.partial(_default_finalisation, eqn.primitive)
        ans = call(*subfuns, *_safe_map(read, eqn.invars), **bind_params)
        if eqn.primitive.multiple_results:
            _safe_map(write, eqn.outvars, ans)
        else:
            write(eqn.outvars[0], ans)
    return _safe_map(read, jaxpr.outvars)


def finalise_jaxpr_as_fn(jaxpr: jax.extend.core.ClosedJaxpr):
    """As `jax.core.jaxpr_as_fn`, but the result is finalised."""
    return ft.partial(finalise_eval_jaxpr, jaxpr.jaxpr, jaxpr.consts)


def finalise_jaxpr(jaxpr: jax.extend.core.ClosedJaxpr) -> jax.extend.core.ClosedJaxpr:
    """A jaxpr-to-jaxpr transformation that performs finalisation."""
    fn = finalise_jaxpr_as_fn(jaxpr)
    args = [
        jax.ShapeDtypeStruct(x.aval.shape, x.aval.dtype) for x in jaxpr.jaxpr.invars
    ]
    return cast(jax.extend.core.ClosedJaxpr, jax.make_jaxpr(fn)(*args))


def finalise_fn(fn):
    """Wraps a function to perform finalisation. (In a manner similar to e.g. `jax.vmap`
    wrapping a function to perform batching.)
    """

    def _finalise_fn(*args):
        jaxpr, struct = jax.make_jaxpr(fn, return_shape=True)(*args)  # pyright: ignore
        flat_args = jtu.tree_leaves(args)
        out = finalise_eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *flat_args)
        treedef = jtu.tree_structure(struct)
        return jtu.tree_unflatten(treedef, out)

    return _finalise_fn


@overload
def finalise_make_jaxpr(
    fn, *, return_shape: Literal[False] = False
) -> Callable[..., jax.extend.core.ClosedJaxpr]: ...


@overload
def finalise_make_jaxpr(
    fn, *, return_shape: Literal[True] = True
) -> Callable[
    ..., tuple[jax.extend.core.ClosedJaxpr, PyTree[jax.ShapeDtypeStruct]]
]: ...


@overload
def finalise_make_jaxpr(
    fn, *, return_shape: bool = False
) -> Callable[
    ...,
    Union[
        jax.extend.core.ClosedJaxpr,
        tuple[jax.extend.core.ClosedJaxpr, PyTree[jax.ShapeDtypeStruct]],
    ],
]: ...


def finalise_make_jaxpr(fn, *, return_shape: bool = False):
    """As `jax.make_jaxpr`, but finalises the result."""

    def _finalise_make_jaxpr(*args):
        jaxpr_struct = jax.make_jaxpr(fn, return_shape=return_shape)(  # pyright: ignore
            *args
        )
        if return_shape:
            jaxpr_struct = cast(tuple[jax.extend.core.ClosedJaxpr, Any], jaxpr_struct)
            jaxpr, struct = jaxpr_struct
            jaxpr = finalise_jaxpr(jaxpr)
            return jaxpr, struct
        else:
            jaxpr_struct = cast(jax.extend.core.ClosedJaxpr, jaxpr_struct)
            jaxpr = finalise_jaxpr(jaxpr_struct)
            return jaxpr

    return _finalise_make_jaxpr


# Register finalisation rules for Equinox's custom primitives.
from .._unvmap import unvmap_all_p, unvmap_any_p, unvmap_max_p
from ..debug._announce_transform import announce_jaxpr_p
from ._loop import maybe_set_p, select_if_vmap_p
from ._noinline import noinline_p
from ._nontraceable import nonbatchable_p, nondifferentiable_backward_p, nontraceable_p


for prim in (
    unvmap_all_p,
    unvmap_any_p,
    unvmap_max_p,
    nondifferentiable_backward_p,
    announce_jaxpr_p,
    select_if_vmap_p,
    maybe_set_p,
    noinline_p,
    nontraceable_p,
    nonbatchable_p,
):
    register_impl_finalisation(prim)


# To make this also useful as debugging tool, we also inline some calls.


def _jvp_call_p_finalisation(fun, jvp, *args, symbolic_zeros=None):
    return fun.call_wrapped(*args)


def _stop_gradient_finalisation(x):
    return x


primitive_finalisations[jax.custom_derivatives.custom_jvp_call_p] = (
    _jvp_call_p_finalisation
)
primitive_finalisations[jax.lax.stop_gradient_p] = _stop_gradient_finalisation
