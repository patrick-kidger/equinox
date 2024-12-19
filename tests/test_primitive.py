import functools as ft

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.core
import jax.extend.core
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.numpy as jnp
import jax.tree_util as jtu

from .helpers import tree_allclose


def test_call():
    newprim_p = jax.extend.core.Primitive("newprim")
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
            return eqx.filter_vmap(call, in_axes=batch_axes)(f, x)

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
    assert tree_allclose(bound(2), 5)
    assert tree_allclose(jax.jit(bound)(2), jnp.array(5))

    vbound = jax.vmap(bound)
    y = jnp.array([1.0, 2.0])
    assert tree_allclose(vbound(y), jnp.array([2.0, 5.0]))
    assert tree_allclose(jax.jit(vbound)(y), jnp.array([2.0, 5.0]))

    primals, tangents = jax.jvp(bound, (y,), (y,))
    assert tree_allclose(primals, jnp.array([2.0, 5.0]))
    assert tree_allclose(tangents, jnp.array([2.0, 8.0]))

    primals, tangents = jax.jvp(vbound, (y,), (y,))
    assert tree_allclose(primals, jnp.array([2.0, 5.0]))
    assert tree_allclose(tangents, jnp.array([2.0, 8.0]))

    cotangents = jax.grad(bound)(5.0)
    assert tree_allclose(cotangents, jnp.array(10.0))

    assert jvpd
    assert batchd
    assert transposed


def test_vprim():
    def impl(x):
        assert x.shape == (2,)
        return [2 * x, jnp.concatenate([x, jnp.flip(x)])]

    def abstract(x):
        assert type(x) is jax.core.ShapedArray
        return [x, jax.core.ShapedArray((4,), x.dtype)]

    def jvp(primals, tangents):
        (x,) = primals
        (tx,) = tangents
        primals_out = newprim_p.bind(x)
        tangents_out = (jnp.flip(tx), jnp.concatenate([x, x + 1]))
        return primals_out, tangents_out

    def transpose(cts_out, x):
        ct_out1, ct_out2 = cts_out
        assert ct_out1.shape == (2,)
        assert ct_out2.shape == (4,)
        return [ct_out1 + ct_out2[:2]]

    newprim_p = eqxi.create_vprim("newprim", impl, abstract, jvp, transpose)
    bind = newprim_p.bind
    y = jnp.array([1.0, 3.5])
    y2 = jnp.array([[1.0, 3.5], [2.0, 1.5], [0.0, 0.2]])

    o1 = [jnp.array([2.0, 7.0]), jnp.array([1.0, 3.5, 3.5, 1.0])]
    assert tree_allclose(bind(y), o1)
    assert tree_allclose(jax.jit(bind)(y), o1)

    dtype = jnp.array(1.0).dtype  # default floating-point dtype
    o2 = [jax.ShapeDtypeStruct((2,), dtype), jax.ShapeDtypeStruct((4,), dtype)]
    assert tree_allclose(jax.eval_shape(bind, y), o2)

    t_o1 = [jnp.array([5.5, 3.0]), jnp.array([1.0, 3.5, 2.0, 4.5])]
    o3 = (o1, t_o1)
    assert tree_allclose(jax.jvp(bind, (y,), (y + 2,)), o3)

    o4 = [
        jnp.array([[2.0, 7.0], [4.0, 3.0], [0.0, 0.4]]),
        jnp.array([[1.0, 3.5, 3.5, 1.0], [2.0, 1.5, 1.5, 2.0], [0.0, 0.2, 0.2, 0.0]]),
    ]
    assert tree_allclose(jax.vmap(bind)(y2), o4)
    assert tree_allclose(jax.jit(jax.vmap(bind))(y2), o4)
    assert tree_allclose(jax.vmap(jax.jit(bind))(y2), o4)

    assert tree_allclose(jax.vmap(jax.vmap(bind))(y2[None]), [o4[0][None], o4[1][None]])

    o5 = [jax.ShapeDtypeStruct((3, 2), dtype), jax.ShapeDtypeStruct((3, 4), dtype)]
    assert tree_allclose(jax.eval_shape(jax.vmap(bind), y2), o5)

    t_o4 = [
        jnp.array([[5.5, 3.0], [3.5, 4.0], [2.2, 2.0]]),
        jnp.array([[1.0, 3.5, 2.0, 4.5], [2.0, 1.5, 3.0, 2.5], [0.0, 0.2, 1.0, 1.2]]),
    ]
    o6 = (o4, t_o4)
    assert tree_allclose(jax.jvp(jax.vmap(bind), (y2,), (y2 + 2,)), o6)

    o7 = jnp.array([3.0, 10.5])
    assert tree_allclose(jax.linear_transpose(bind, y)(o1), (o7,))

    o8 = jnp.array([[3.0, 10.5], [6.0, 4.5], [0.0, 0.6]])
    assert tree_allclose(jax.linear_transpose(jax.vmap(bind), y2)(o4), (o8,))
