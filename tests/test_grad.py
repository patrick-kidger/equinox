from typing import Union

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import numpy as np
import pytest

import equinox as eqx

from .helpers import shaped_allclose


@pytest.mark.parametrize("api_version", (0, 1))
def test_filter_grad1(api_version, getkey):
    a = jrandom.normal(getkey(), (2, 3))

    def f(x):
        return jnp.sum(x)

    if api_version == 0:
        f = eqx.filter_grad(f, filter_spec=lambda _: True)
    else:
        f = eqx.filter_grad(arg=True)(f)

    grad_f = f(a)
    assert jnp.all(grad_f == 1)


@pytest.mark.parametrize("api_version", (0, 1))
def test_filter_grad2(api_version, getkey):
    a = jrandom.normal(getkey(), (2, 3))
    b = jrandom.normal(getkey(), (2, 3))

    def f(x):
        sum = 0.0
        for arg in jtu.tree_leaves(x):
            if eqx.is_array_like(arg):
                sum = sum + jnp.sum(arg)
        return sum

    if api_version == 0:
        f = eqx.filter_grad(f, filter_spec=eqx.is_inexact_array)
    else:
        f = eqx.filter_grad(arg=eqx.is_inexact_array)(f)

    ga, gb = f([a, b])
    assert jnp.all(ga == 1)
    assert jnp.all(gb == 1)

    gtrue, ghi, gobject, ga = f([True, "hi", object(), a])
    assert gtrue is None
    assert ghi is None
    assert gobject is None
    assert jnp.all(ga == 1)

    gtrue, gdict, (g5, g1), gnp = f(
        [
            True,
            {"hi": eqx.nn.Linear(1, 1, key=getkey())},
            (5, 1.0),
            np.array([2.0, 3.0]),
        ]
    )
    assert gtrue is None
    assert list(gdict.keys()) == ["hi"]
    assert isinstance(gdict["hi"], eqx.nn.Linear)
    assert jnp.all(gdict["hi"].weight == 1)
    assert jnp.all(gdict["hi"].bias == 1)
    assert g5 is None
    assert g1 is None
    assert jnp.all(gnp == 1)


@pytest.mark.parametrize("api_version", (0, 1))
def test_filter_grad3(api_version, getkey):
    a = jrandom.normal(getkey(), (2, 3))
    b = jrandom.normal(getkey(), (1, 2))
    c = jrandom.normal(getkey(), ())

    def f(x):
        return jnp.sum(x[0]) + jnp.sum(x[1])

    if api_version == 0:
        f = eqx.filter_grad(f, filter_spec=[True, False])
    else:
        f = eqx.filter_grad(arg=[True, False])(f)

    ga, gb = f([a, b])
    assert jnp.all(ga == 1)
    assert gb is None

    def h(x, y):
        return jnp.sum(x["a"]) * jnp.sum(x["b"]) * y

    if api_version == 0:
        h = eqx.filter_grad(h, filter_spec={"a": True, "b": False})
    else:
        h = eqx.filter_grad(arg={"a": True, "b": False})(h)

    grad = h({"a": a, "b": b}, c)
    assert jnp.allclose(grad["a"], jnp.sum(b) * c)
    assert grad["b"] is None

    with pytest.raises(ValueError):
        grad = h(c, {"a": a, "b": b})


# TODO: more comprehensive tests on this.
@pytest.mark.parametrize("api_version", (0, 1))
def test_filter_value_and_grad(api_version, getkey):
    a = jrandom.normal(getkey(), (2, 3))

    def f(x):
        return jnp.sum(x)

    if api_version == 0:
        f = eqx.filter_value_and_grad(f, filter_spec=eqx.is_inexact_array)
    else:
        f = eqx.filter_value_and_grad(arg=eqx.is_inexact_array)(f)

    val, grad = f(a)
    assert val == jnp.sum(a)
    assert jnp.all(grad == 1)


@pytest.mark.parametrize("api_version", (0, 1))
def test_aux(api_version, getkey):
    a = jrandom.normal(getkey(), (2, 3))

    def f(x):
        return jnp.sum(x), "hi"

    if api_version == 0:
        f = eqx.filter_grad(f, has_aux=True, filter_spec=eqx.is_inexact_array)
    else:
        f = eqx.filter_grad(has_aux=True, arg=eqx.is_inexact_array)(f)

    grad, aux = f(a)
    assert aux == "hi"
    assert jnp.all(grad == 1)

    def f(x):
        return jnp.sum(x), "hi"

    if api_version == 0:
        f = eqx.filter_value_and_grad(f, has_aux=True, filter_spec=eqx.is_inexact_array)
    else:
        f = eqx.filter_value_and_grad(has_aux=True, arg=eqx.is_inexact_array)(f)

    (value, aux), grad = f(a)
    assert value == jnp.sum(a)
    assert aux == "hi"
    assert jnp.all(grad == 1)


@pytest.mark.parametrize("call", [False, True])
@pytest.mark.parametrize("outer", [False, True])
def test_methods(call, outer):
    class M(eqx.Module):
        increment: Union[int, jnp.ndarray]

        if call:

            def __call__(self, x):
                return x + self.increment

            if not outer:
                __call__ = eqx.filter_grad(__call__)
        else:

            def method(self, x):
                return x + self.increment

            if not outer:
                method = eqx.filter_grad(method)

    m = M(jnp.array(5.0))
    grad_m = M(jnp.array(1.0))
    y = jnp.array(1.0)

    if call:
        if outer:
            assert eqx.filter_grad(m)(y) == 1
        else:
            assert m(y) == grad_m
    else:
        if outer:
            assert eqx.filter_grad(m.method)(y) == 1
        else:
            assert m.method(y) == grad_m


def test_grad_jit():
    num_traces = 0

    @eqx.filter_custom_vjp
    def f(x):
        return x

    def f_fwd(x):
        return x, None

    def f_bwd(_, g, __):
        nonlocal num_traces
        num_traces += 1
        return g + 2

    f.defvjp(f_fwd, f_bwd)
    x = jnp.array(1.0)

    jitf = jax.jit(f)
    assert eqx.filter_grad(jitf)(x) == 3
    assert eqx.filter_grad(jitf)(x) == 3
    assert num_traces == 1
    assert eqx.filter_grad(eqx.filter_jit(f))(x) == 3
    assert eqx.filter_grad(eqx.filter_jit(f))(x) == 3
    assert num_traces == 2


def test_filter_jvp():
    after_filter_jit = eqx.filter_jit(
        args=(False, eqx.is_array_like, eqx.is_array_like)
    )
    identity = lambda f: f
    for before_jit in (eqx.filter_jit(default=eqx.is_array_like), identity):
        for after_jit in (after_filter_jit, identity):

            @before_jit
            def f(x):
                return x + 1

            primals, tangents = after_jit(eqx.filter_jvp)(f, (1.0,), (1.0,))
            assert shaped_allclose(primals, jnp.array(2.0))
            assert shaped_allclose(tangents, jnp.array(1.0))

            another_object = object()

            @before_jit
            def g(*x):
                y = (
                    1 + x[0] ** 2,
                    2 + x[1] * 3,
                    3 + x[2] * 4,
                    4 + x[3] * 5,
                    5 + x[4] * 6,
                    another_object,
                )
                return y

            primals_in = (
                jnp.array(1.0),
                jnp.array(1),
                1.0,
                1,
                jnp.array(1.0),
                object(),
            )
            true_primals_out = (
                jnp.array(2.0),
                jnp.array(5),
                jnp.array(7.0),
                jnp.array(9),
                jnp.array(11.0),
                another_object,
            )

            tangents_in1 = (jnp.array(5.0), None, None, None, None, None)
            true_tangents_out1 = (jnp.array(10.0), None, None, None, None, None)

            tangents_in2 = (jnp.array(5.0), None, jnp.array(3.0), None, None, None)
            true_tangents_out2 = (
                jnp.array(10.0),
                None,
                jnp.array(12.0),
                None,
                None,
                None,
            )

            tangents_in3 = (None, None, None, None, None, None)
            true_tangents_out3 = (None, None, None, None, None, None)

            all_tangents_in = (tangents_in1, tangents_in2, tangents_in3)
            all_true_tangents_out = (
                true_tangents_out1,
                true_tangents_out2,
                true_tangents_out3,
            )

            for tangents_in, true_tangents_out in zip(
                all_tangents_in, all_true_tangents_out
            ):
                primals_out, tangents_out = after_jit(eqx.filter_jvp)(
                    g, primals_in, tangents_in
                )
                assert primals_out == true_primals_out
                assert shaped_allclose(tangents_out, true_tangents_out)

            bad_tangents_in1 = (jnp.array(5), None, None, None, None, None)
            bad_tangents_in2 = (None, None, None, None, None, object())
            bad_tangents_in3 = (None, jnp.array(1.0), None, None, None, None)

            for tangents_in in (bad_tangents_in1, bad_tangents_in2, bad_tangents_in3):
                with pytest.raises(TypeError):
                    after_jit(eqx.filter_jvp)(g, primals_in, tangents_in)


def test_filter_vjp(getkey):
    mlp = eqx.nn.MLP(2, 3, 2, 2, key=getkey())
    x = jnp.array(1)
    y = 1.0
    sentinel = object()
    sentinel2 = object()

    def f(_mlp, _x, _y, _sentinel):
        return _mlp(jnp.array([1.0, 2.0])) + _x + _y, 2.0, sentinel2

    def g(_mlp, _x, _y, _sentinel):
        return f(_mlp, _x, _y, _sentinel), _sentinel

    def _check(x, y):
        assert x.shape == y.shape
        assert x.dtype == y.dtype

    out1, vjpfun1 = eqx.filter_vjp(f, mlp, x, y, sentinel)
    out2, vjpfun2, aux = eqx.filter_vjp(g, mlp, x, y, sentinel, has_aux=True)
    assert aux is sentinel
    for out, vjpfun in ((out1, vjpfun1), (out2, vjpfun2)):
        out_array, out_float, out_sentinel = out
        assert out_array.shape == (3,)
        assert out_array.dtype == jnp.float32
        assert shaped_allclose(out_float, 2.0)
        assert out_sentinel is sentinel2
        ct_mlp, ct_x, ct_y, ct_sentinel = vjpfun(
            (jnp.array([1.0, 2.0, 3.0]), None, None)
        )
        mlp_dyn = eqx.filter(mlp, eqx.is_array)
        assert jtu.tree_structure(mlp_dyn) == jtu.tree_structure(ct_mlp)
        jtu.tree_map(_check, mlp_dyn, ct_mlp)
        assert ct_y is None
        assert ct_sentinel is None


def test_closure_convert_basic():
    @jax.grad
    def f(x, y):
        z = x + y
        g = lambda a: z + a  # closes over z
        g2 = eqx.filter_closure_convert(g, 1)
        assert [id(b) for b in g2.consts] == [id(z)]
        return z

    f(1.0, 1.0)


def test_filter_custom_jvp():
    @eqx.filter_custom_jvp
    def call(fn, x):
        return fn(x)

    was_called = False

    @call.defjvp
    def call_jvp(primals, tangents):
        nonlocal was_called
        was_called = True
        fn, x = primals
        tfn, tx = tangents
        assert tfn is None
        primal_out = call(fn, x)
        tangent_out = tx**2
        return primal_out, tangent_out

    f = lambda a: a**2 + 1
    assert shaped_allclose(call(f, 1), 2)
    assert shaped_allclose(call(f, 1.0), 2.0)
    assert shaped_allclose(call(f, jnp.array(1)), jnp.array(2))
    assert shaped_allclose(call(f, jnp.array(1.0)), jnp.array(2.0))

    def jvpcall(x, tx):
        def _jvpcall(_x):
            return call(f, _x)

        return jax.jvp(_jvpcall, (x,), (tx,))

    primal_out, tangent_out = jvpcall(2.0, 3.0)
    assert was_called
    assert shaped_allclose(primal_out, 5.0)
    assert shaped_allclose(tangent_out, 9.0)
