import warnings
from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import numpy as np
import pytest

from .helpers import tree_allclose


def test_filter_grad(getkey):
    a = jrandom.normal(getkey(), (2, 3))
    b = jrandom.normal(getkey(), (2, 3))

    @eqx.filter_grad
    def f(x):
        sum = jnp.array(0.0)
        for arg in jtu.tree_leaves(x):
            if eqx.is_array_like(arg):
                sum = sum + jnp.sum(arg).astype(sum.dtype)
        return sum

    ga, gb = f([a, b])
    assert tree_allclose(ga, jnp.ones((2, 3)))
    assert tree_allclose(gb, jnp.ones((2, 3)))

    gtrue, ghi, gobject, ga = f([True, "hi", object(), a])
    assert gtrue is None
    assert ghi is None
    assert gobject is None
    assert tree_allclose(ga, jnp.ones((2, 3)))

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
    assert tree_allclose(gdict["hi"].weight, jnp.ones((1, 1)))
    assert tree_allclose(gdict["hi"].bias, jnp.ones(1))
    assert g5 is None
    assert g1 is None
    assert tree_allclose(gnp, jnp.ones(2))


# TODO: more comprehensive tests on this.
def test_filter_value_and_grad(getkey):
    a = jrandom.normal(getkey(), (2, 3))

    @eqx.filter_value_and_grad
    def f(x):
        return jnp.sum(x)

    val, grad = f(a)
    assert tree_allclose(val, jnp.sum(a))
    assert tree_allclose(grad, jnp.ones((2, 3)))


def test_aux(getkey):
    a = jrandom.normal(getkey(), (2, 3))

    @eqx.filter_grad(has_aux=True)
    def f(x):
        return jnp.sum(x), "hi"

    grad, aux = f(a)
    assert aux == "hi"
    assert jnp.all(grad == 1)

    @eqx.filter_value_and_grad(has_aux=True)
    def g(x):
        return jnp.sum(x), "hi"

    (value, aux), grad = g(a)
    assert value == jnp.sum(a)
    assert aux == "hi"
    assert jnp.all(grad == 1)


@pytest.mark.parametrize("call", [False, True])
@pytest.mark.parametrize("outer", [False, True])
def test_methods(call, outer):
    class M(eqx.Module):
        increment: Union[int, jax.Array]

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


def test_grad_jit_new():
    num_traces = 0

    @eqx.filter_custom_vjp
    def f(x):
        return x

    @f.def_fwd
    def f_fwd(perturbed, x):
        del perturbed
        return x, None

    @f.def_bwd
    def f_bwd(residual, g, perturbed, x):
        del residual, perturbed, x
        nonlocal num_traces
        num_traces += 1
        return g + 2

    x = jnp.array(1.0)

    jitf = jax.jit(f)
    assert eqx.filter_grad(jitf)(x) == 3
    assert eqx.filter_grad(jitf)(x) == 3
    assert num_traces == 1
    assert eqx.filter_grad(eqx.filter_jit(f))(x) == 3  # pyright: ignore
    assert eqx.filter_grad(eqx.filter_jit(f))(x) == 3  # pyright: ignore
    assert num_traces == 2


def test_grad_jit_old():
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f.defvjp(f_fwd, f_bwd)

    x = jnp.array(1.0)

    jitf = jax.jit(f)
    assert eqx.filter_grad(jitf)(x) == 3
    assert eqx.filter_grad(jitf)(x) == 3
    assert num_traces == 1
    assert eqx.filter_grad(eqx.filter_jit(f))(x) == 3  # pyright: ignore
    assert eqx.filter_grad(eqx.filter_jit(f))(x) == 3  # pyright: ignore
    assert num_traces == 2


def test_filter_jvp():
    _map_is_array_like = lambda pytree: jtu.tree_map(
        lambda x: jnp.array(x) if eqx.is_array_like(x) else x, pytree
    )

    def after_filter_jit(fun):
        fun_jit = eqx.filter_jit(fun, donate="none")

        def _fun(x, y, z, *args):
            y = _map_is_array_like(y)
            z = _map_is_array_like(z)
            return fun_jit(x, y, z, *args)

        return _fun

    def before_filter_jit(fun):
        fun_jit = eqx.filter_jit(fun, donate="none")

        def _fun(*args):
            args = _map_is_array_like(args)
            return fun_jit(*args)

        return _fun

    identity = lambda f: f
    for before_jit in (before_filter_jit, identity):
        for after_jit in (after_filter_jit, identity):

            @before_jit
            def f(x):
                return x + 1

            primals, tangents = after_jit(eqx.filter_jvp)(f, (1.0,), (1.0,))
            assert tree_allclose(primals, jnp.array(2.0))
            assert tree_allclose(tangents, jnp.array(1.0))

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
                assert tree_allclose(tangents_out, true_tangents_out)

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
        assert tree_allclose(out_float, 2.0)
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


def test_closure_convert_trivial():
    def f(a):
        return a + 1

    f2 = eqx.filter_closure_convert(f, 1)
    f2.out_struct
    assert type(f2).__name__ == "_TrivialClosureConvert"


def test_closure_convert_custom_jvp():
    @eqx.filter_custom_jvp
    def call(f, x):
        return f(x)

    @call.def_jvp
    def call_jvp(primals, tangents):
        f, x = primals
        tf, tx = tangents
        out = call(f, x)
        tsum = sum(jnp.sum(x) for x in jtu.tree_leaves((tf, tx)))
        tout = jtu.tree_map(lambda x: jnp.full(x.shape, tsum, x.dtype), out)
        return out, tout

    @jax.grad
    def run(x):
        x1, x2 = x
        f = lambda y: x1 * y + x2
        f = eqx.filter_closure_convert(f, 3.0)
        return call(f, 3.0)

    assert tree_allclose(run((2.0, 4.0)), (jnp.array(1.0), jnp.array(1.0)))


def test_filter_custom_jvp_no_kwargs():
    @eqx.filter_custom_jvp
    def call(fn, x):
        return fn(x)

    was_called = False

    @call.def_jvp
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
    assert tree_allclose(call(f, 1), 2)
    assert tree_allclose(call(f, 1.0), 2.0)
    assert tree_allclose(call(f, jnp.array(1)), jnp.array(2))
    assert tree_allclose(call(f, jnp.array(1.0)), jnp.array(2.0))

    def jvpcall(x, tx):
        def _jvpcall(_x):
            return call(f, _x)

        return jax.jvp(_jvpcall, (x,), (tx,))

    primal_out, tangent_out = jvpcall(2.0, 3.0)
    assert was_called
    assert tree_allclose(primal_out, 5.0)
    assert tree_allclose(tangent_out, 9.0)


def test_filter_custom_jvp_kwargs():
    @eqx.filter_custom_jvp
    def call(x, y, *, fn):
        return fn(x, y)

    was_called = False

    @call.def_jvp
    def call_jvp(primals, tangents, *, fn):
        nonlocal was_called
        was_called = True
        x, y = primals
        tx, ty = tangents
        primal_out = call(x, y, fn=fn)
        tangent_out = tx**2 + ty
        return primal_out, tangent_out

    f = lambda a, b: a**2 + b
    assert tree_allclose(call(1, 2, fn=f), 3)
    assert tree_allclose(call(1.0, 2.0, fn=f), 3.0)
    assert tree_allclose(call(jnp.array(1), 2, fn=f), jnp.array(3))
    assert tree_allclose(call(jnp.array(1.0), 2, fn=f), jnp.array(3.0))

    def jvpcall(x, y, tx, ty):
        def _jvpcall(_x, _y):
            return call(_x, _y, fn=f)

        return jax.jvp(_jvpcall, (x, y), (tx, ty))

    primal_out, tangent_out = jvpcall(2.0, 1.5, 3.0, 4.0)
    assert was_called
    assert tree_allclose(primal_out, 5.5)
    assert tree_allclose(tangent_out, 13.0)


# Checks that we don't get a leaked tracer error from passing an array through
# nondiff_argnums.
def test_filter_custom_jvp_exact():
    @eqx.filter_custom_jvp
    def f(x, y):
        return y

    @f.def_jvp
    def f_jvp(primals, tangents):
        x, y = primals
        tang_x, tang_y = tangents
        assert tang_x is None
        return x + y, tang_y

    def g(y, x):
        return jax.lax.cond(x < y, f, lambda _x, _y: _y, x, y)

    jax.grad(g)(1.0, 1)


def test_filter_hessian_and_jacfwd_and_jacrev():
    # filter_hessian is implemented in terms of the other 2, so this tests all 3.

    def f(x, y):
        return jnp.sin(x) * y

    def f_fwd(x, y):
        return f(x, y), (jnp.cos(x), jnp.sin(x), y)

    def f_bwd(res, g):
        cos_x, sin_x, y = res
        return (cos_x * g * y, sin_x * g)

    f_custom = jax.custom_vjp(f)
    f_custom.defvjp(f_fwd, f_bwd)
    jax_hess = jax.hessian(f)(1.0, 2.0)
    eqx_hess = eqx.filter_hessian(f)(jnp.array(1.0), jnp.array(2.0))
    assert tree_allclose(jax_hess, eqx_hess)


def test_pytree_jacfwd():
    class NeuralNetwork(eqx.Module):
        layers: list
        extra_bias: jax.Array

        def __init__(self, key):
            key1, key2, key3 = jax.random.split(key, 3)
            self.layers = [
                eqx.nn.Linear(2, 8, key=key1),
                eqx.nn.Linear(8, 8, key=key2),
                eqx.nn.Linear(8, 2, key=key3),
            ]
            self.extra_bias = jax.numpy.ones(2)

        def __call__(self, x):
            for layer in self.layers[:-1]:
                x = jax.nn.relu(layer(x))
            return self.layers[-1](x) + self.extra_bias

    def loss(model, x, y):
        pred_y = jax.vmap(model)(x)
        return jax.numpy.mean((y - pred_y) ** 2)

    x_key, y_key, model_key = jax.random.split(jax.random.PRNGKey(0), 3)
    x = jax.random.normal(x_key, (3, 2))
    y = jax.random.normal(y_key, (3, 2))
    model = NeuralNetwork(model_key)
    assert tree_allclose(
        eqx.filter_grad(loss)(model, x, y), eqx.filter_jacfwd(loss)(model, x, y)
    )
    assert tree_allclose(
        eqx.filter_grad(loss)(model, x, y), eqx.filter_jacrev(loss)(model, x, y)
    )


def test_filter_custom_jvp_symbolic_zero():
    @eqx.filter_custom_jvp
    def f(x, y):
        return x + y

    @f.def_jvp
    def f_jvp(primals, tangents):
        x, y = primals
        tx, ty = tangents
        assert ty is None
        return x + y, tx

    one = jnp.array(1.0)
    jax.jvp(lambda x: f(x, one), (one,), (one,))


def test_filter_custom_vjp_nondiff_args():
    @eqx.filter_custom_vjp
    def f(x, y):
        return x + y

    @f.def_fwd
    def f_fwd(perturbed, x, y):
        return x + y, None

    @f.def_bwd
    def f_bwd(res, ct, perturbed, x, y):
        return ct

    with pytest.raises(RuntimeError):
        jax.grad(lambda x: f(x, x))(1.0)


def test_filter_custom_vjp_symbolic_zero():
    called = False

    @eqx.filter_custom_vjp
    def f(x__y, *, foo):
        x, y = x__y
        return x + y + foo, x, y, foo

    @f.def_fwd
    def f_fwd(perturbed, x__y, *, foo):
        p_x, p_y = perturbed
        assert p_x is True
        assert p_y is False
        x, y = x__y
        return f((x, y), foo=foo), None

    @f.def_bwd
    def f_bwd(res, ct, perturbed, x__y, *, foo):
        nonlocal called
        called = True
        p_x, p_y = perturbed
        assert p_x is True
        assert p_y is False
        ct1, ct2, ct3, ct4 = ct
        assert ct1 is not None
        assert ct2 is None
        assert ct3 is not None
        assert ct4 is None
        return ct1, ct3

    @jax.grad
    def run(x, y):
        out1, out2, out3, out4 = f((x, jnp.array(1.0)), foo=y)
        del out2, out4
        return out1 + out3

    run(1.0, 2.0)
    assert called


def test_filter_custom_vjp_defvjp():
    @eqx.filter_custom_vjp
    def f(x):
        a, b = x
        return (a + 1, b + 2, b + 3)

    def f_fwd(x):
        return f(x), None

    def f_bwd(res, g, x):
        g0, g1, g2 = g
        return g0 + g1, g2

    with pytest.warns():
        f.defvjp(f_fwd, f_bwd)
    jax.grad(lambda x: sum(f(x)))((jnp.array(1.0), jnp.array(1.0)))


# We expect the gradient of the second output of `g` to be a symbolic zero, even though
# it's carrying a JVP tracer from the higher level of the stack.
def test_double_filter_jvp():
    def f(x):
        def g(y):
            return y, x

        (a, b), (ta, tb) = eqx.filter_jvp(g, (x,), (1.0,))
        assert a is not None
        assert b is not None
        assert ta is not None
        assert tb is None

    eqx.filter_jvp(f, (1.0,), (1.0,))


def test_filter_custom_vjp_nonarray_residual():
    @eqx.filter_custom_vjp
    def f(x):
        return 2 * x

    @f.def_fwd
    def f_fwd(_, x):
        return 2 * x, object()

    @f.def_bwd
    def f_bwd(token, ct, _, x):
        return 2 * ct

    jax.grad(f)(1.0)


def test_positional_first_argument():
    x = jnp.array(1.0)
    with pytest.raises(TypeError, match="Functions wrapped with"):
        eqx.filter_grad(lambda x: x + 1)(x=x)
    with pytest.raises(TypeError, match="Functions wrapped with"):
        eqx.filter_value_and_grad(lambda x, y: x + y)(x=x, y=x)
