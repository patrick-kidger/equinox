import functools as ft

import jax
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.numpy as jnp
import jax.tree_util as jtu

import equinox as eqx
import equinox.internal as eqxi

from .helpers import shaped_allclose


def test_filter_primitives():
    newprim_p = jax.core.Primitive("newprim")
    newprim_p.multiple_results = True

    newprim = ft.partial(eqxi.filter_primitive_bind, newprim_p)

    jvpd = False
    batchd = False
    transposed = False

    call = lambda f, x: f(x)

    @eqxi.filter_primitive_def
    def impl_rule(f, x):
        return call(f, x)

    def _to_struct(x):
        if isinstance(x, jax.core.ShapedArray):
            return jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
        else:
            return x

    def _to_shapedarray(x):
        if isinstance(x, jax.ShapeDtypeStruct):
            return jax.core.ShapedArray(shape=x.shape, dtype=x.dtype)
        else:
            return x

    @eqxi.filter_primitive_def
    def abstract_rule(f, x):
        x = jtu.tree_map(_to_struct, x)
        out = eqx.filter_eval_shape(f, x)
        out = jtu.tree_map(_to_shapedarray, out)
        return out

    def _jvp(in_):
        primals, tangents = in_
        _, tout = eqx.filter_jvp(call, primals, tangents)
        return tout

    @eqxi.filter_primitive_jvp
    def jvp_rule(primals, tangents):
        nonlocal jvpd
        jvpd = True
        f, x = primals
        primals_out = newprim(f, x)
        tangents_out = newprim(_jvp, (primals, tangents))
        return primals_out, tangents_out

    _is_undefined = lambda x: isinstance(x, ad.UndefinedPrimal)

    @eqxi.filter_primitive_transpose
    def transpose_rule(inputs, cts_out):
        nonlocal transposed
        transposed = True
        undefined, defined = eqx.partition(inputs, _is_undefined, is_leaf=_is_undefined)
        undefined = jtu.tree_map(lambda x: x.aval, undefined, is_leaf=_is_undefined)

        def transpose(in_):
            _defined, _cts_out = in_

            def _transpose(_undefined):
                _f, _x = eqx.combine(_defined, _undefined)
                return call(_f, _x)

            (_cts,) = jax.linear_transpose(_transpose, undefined)(_cts_out)
            return _cts

        cts = newprim(transpose, (defined, cts_out))
        return cts

    @eqxi.filter_primitive_batching
    def batching_rule(inputs, batch_axes):
        nonlocal batchd
        batchd = True

        def vcall(y):
            f, x = y
            return eqx.filter_vmap(call, args=batch_axes)(f, x)

        out = newprim(vcall, inputs)
        make_zero = lambda x: 0 if eqx.is_array(x) else None
        zeros = jtu.tree_map(make_zero, out)
        return out, zeros

    newprim_p.def_impl(impl_rule)
    newprim_p.def_abstract_eval(abstract_rule)
    ad.primitive_jvps[newprim_p] = jvp_rule
    ad.primitive_transposes[newprim_p] = transpose_rule
    batching.primitive_batchers[newprim_p] = batching_rule
    mlir.register_lowering(newprim_p, mlir.lower_fun(impl_rule, multiple_results=True))

    def fn(x):
        return x**2 + 1

    bound = ft.partial(newprim, fn)
    assert shaped_allclose(bound(2), 5)
    assert shaped_allclose(jax.jit(bound)(2), jnp.array(5))

    vbound = jax.vmap(bound)
    y = jnp.array([1.0, 2.0])
    assert shaped_allclose(vbound(y), jnp.array([2.0, 5.0]))
    assert shaped_allclose(jax.jit(vbound)(y), jnp.array([2.0, 5.0]))

    primals, tangents = jax.jvp(bound, (y,), (y,))
    assert shaped_allclose(primals, jnp.array([2.0, 5.0]))
    assert shaped_allclose(tangents, jnp.array([2.0, 8.0]))

    primals, tangents = jax.jvp(vbound, (y,), (y,))
    assert shaped_allclose(primals, jnp.array([2.0, 5.0]))
    assert shaped_allclose(tangents, jnp.array([2.0, 8.0]))

    cotangents = jax.grad(bound)(5.0)
    assert shaped_allclose(cotangents, jnp.array(10.0))

    assert jvpd
    assert batchd
    assert transposed


def test_vprim():
    impl = lambda x: (2 * x + 1,)
    abstract = lambda x: (x,)
    jvp = lambda p, t: ((2 * p[0] + 1,), (2 * t[0],))
    multwo_addone_p = eqxi.create_vprim(
        "multwo_addone", impl, abstract, jvp, transpose=None
    )
    bind = lambda x: multwo_addone_p.bind(x)[0]

    o1 = bind(1)
    o2 = jax.jit(bind)(1)
    o3 = jax.eval_shape(bind, 1)
    o4 = jax.jvp(bind, (1.0,), (1.0,))

    o5 = jax.vmap(bind)(jnp.array([1, 2]))
    o6 = jax.jit(jax.vmap(bind))(jnp.array([1, 2]))
    o7 = jax.vmap(jax.jit(bind))(jnp.array([1, 2]))
    o8 = jax.eval_shape(jax.vmap(bind), jnp.array([1, 2]))
    o9 = jax.jvp(jax.vmap(bind), (jnp.array([1.0, 2.0]),), (jnp.array([3.0, 4.0]),))

    o10 = jax.vmap(jax.vmap(bind))(jnp.array([[1, 2], [3, 4]]))

    assert shaped_allclose(o1, 3)
    assert shaped_allclose(o2, jnp.array(3))
    assert type(o3) is jax.ShapeDtypeStruct
    assert o3.shape == ()
    assert o3.dtype == jnp.int32
    assert shaped_allclose(o4, (3.0, 2.0))

    assert shaped_allclose(o5, jnp.array([3, 5]))
    assert shaped_allclose(o6, jnp.array([3, 5]))
    assert shaped_allclose(o7, jnp.array([3, 5]))
    assert type(o8) is jax.ShapeDtypeStruct
    assert o8.shape == (2,)
    assert o8.dtype == jnp.int32
    assert shaped_allclose(o9, (jnp.array([3.0, 5.0]), jnp.array([6.0, 8.0])))

    assert shaped_allclose(o10, jnp.array([[3, 5], [7, 9]]))
