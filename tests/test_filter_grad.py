from typing import Union

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

import equinox as eqx


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
        for arg in jax.tree_leaves(x):
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
    assert gnp is None


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
