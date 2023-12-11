import collections as co

import equinox as eqx
import jax
import jax.custom_derivatives as custom_derivatives
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu


def test_basic(getkey):
    mlp_pytree = eqx.nn.MLP(2, 2, 2, 2, key=getkey())
    mlp_arg = eqx.nn.MLP(2, 2, 2, 2, key=getkey())
    mlp_closure = eqx.nn.MLP(2, 2, 2, 2, key=getkey())
    int_pytree = jnp.array([1])
    int_closure = jnp.array([1])

    sentinel = object()

    class _M(eqx.Module):
        mlp_pytree: eqx.nn.MLP
        int_pytree: jax.Array

        def __call__(self, mlp_arg, x):
            x = self.mlp_pytree(x)
            x = mlp_arg(x)
            x = mlp_closure(x)
            with jax.numpy_dtype_promotion("standard"):
                x = x + self.int_pytree + int_closure
            return x, sentinel

    m = _M(mlp_pytree, int_pytree)
    x = jnp.array([1.0, 2.0])
    closed_jaxpr, dynamic_out_struct, out_static = eqx.filter_make_jaxpr(m)(mlp_arg, x)
    jaxpr = closed_jaxpr.jaxpr
    consts = closed_jaxpr.consts

    prims = co.defaultdict(int)
    for eqn in jaxpr.eqns:
        prims[eqn.primitive] += 1
    assert prims == {
        lax.dot_general_p: 9,
        lax.add_p: 11,
        custom_derivatives.custom_jvp_call_p: 6,
        lax.convert_element_type_p: 2,
    }

    params = {
        id(x)
        for x in jtu.tree_leaves((mlp_pytree, mlp_closure, int_pytree, int_closure))
        if eqx.is_array(x)
    }
    assert {id(x) for x in consts} == params

    assert dynamic_out_struct == (jax.ShapeDtypeStruct((2,), jnp.float32), None)
    assert out_static == (None, sentinel)


def test_struct():
    struct = jax.ShapeDtypeStruct((2,), jnp.float32)

    def f(x):
        x1 = x[None]  # (1, 2)
        x2 = x[:, None]  # (2, 1)
        # That is, what if a ShapeDtypeStruct is itself part of the static return value?
        return x1 * x2, jax.ShapeDtypeStruct((3,), jnp.float32)

    closed_jaxpr, dynamic_out_struct, out_static = eqx.filter_make_jaxpr(f)(struct)
    jaxpr = closed_jaxpr.jaxpr
    consts = closed_jaxpr.consts

    prims = co.defaultdict(int)
    for eqn in jaxpr.eqns:
        prims[eqn.primitive] += 1
    assert prims == {lax.broadcast_in_dim_p: 2, lax.mul_p: 1}
    assert consts == []
    assert dynamic_out_struct == (jax.ShapeDtypeStruct((2, 2), jnp.float32), None)
    assert out_static == (None, jax.ShapeDtypeStruct((3,), jnp.float32))
