import functools as ft
from typing import Any, Dict

import jax
import jax.tree_util as jtu


def _safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f"length mismatch: {list(map(len, args))}"
    return list(map(f, *args))


def finalise_eval_jaxpr(jaxpr: jax.core.Jaxpr, consts, *args, wrapper_prims=()):
    def read(v: jax.core.Atom) -> Any:
        return v.val if isinstance(v, jax.core.Literal) else env[v]

    def write(v: jax.core.Var, val: Any) -> None:
        env[v] = val

    env: Dict[jax.core.Var, Any] = {}
    _safe_map(write, jaxpr.constvars, consts)
    _safe_map(write, jaxpr.invars, args)
    for eqn in jaxpr.eqns:
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        prim = eqn.primitive
        call = prim.impl if prim in wrapper_prims else prim.bind
        ans = call(*subfuns, *_safe_map(read, eqn.invars), **bind_params)
        if eqn.primitive.multiple_results:
            _safe_map(write, eqn.outvars, ans)
        else:
            write(eqn.outvars[0], ans)
    return _safe_map(read, jaxpr.outvars)


def finalise_jaxpr_as_fn(jaxpr: jax.core.ClosedJaxpr, *, wrapper_prims=()):
    return ft.partial(
        finalise_eval_jaxpr, jaxpr.jaxpr, jaxpr.consts, wrapper_prims=wrapper_prims
    )


def finalise_jaxpr(
    jaxpr: jax.core.ClosedJaxpr, *, return_shape=False, wrapper_prims=()
):
    fn = finalise_jaxpr_as_fn(jaxpr, wrapper_prims=wrapper_prims)
    args = [
        jax.ShapeDtypeStruct(x.aval.shape, x.aval.dtype) for x in jaxpr.jaxpr.invars
    ]
    return jax.make_jaxpr(fn, return_shape=return_shape)(*args)


def finalise_fn(fn, *, wrapper_prims=()):
    def _finalise_fn(*args):
        jaxpr, struct = jax.make_jaxpr(fn, return_shape=True)(*args)
        out = finalise_eval_jaxpr(
            jaxpr.jaxpr, jaxpr.consts, *args, wrapper_prims=wrapper_prims
        )
        treedef = jtu.tree_structure(struct)
        return jtu.tree_unflatten(treedef, out)

    return _finalise_fn
