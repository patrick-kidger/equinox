import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

import equinox as eqx


def _eq(a, b):
    return (type(a) is type(b)) and (a == b)


@pytest.mark.parametrize("api_version", (0, 1))
def test_filter_jit1(api_version, getkey):
    a = jrandom.normal(getkey(), (2, 3))
    b = jrandom.normal(getkey(), (3,))
    c = jrandom.normal(getkey(), (1, 4))
    general_tree = [
        1,
        True,
        object(),
        {"a": a, "tuple": (2.0, b)},
        c,
        eqx.nn.MLP(2, 2, 2, 2, key=getkey()),
    ]
    array_tree = [{"a": a, "b": b}, (c,)]
    _mlp = jax.tree_map(lambda u: u if eqx.is_array_like(u) else None, general_tree[-1])

    if api_version == 0:

        @ft.partial(eqx.filter_jit, filter_spec=lambda _: True)
        def f(x):
            return x

    else:

        @eqx.filter_jit(default=True)
        def f(x):
            return x

    assert jnp.all(a == f(a))
    f1 = f(array_tree)
    assert jnp.all(f1[0]["a"] == a)
    assert jnp.all(f1[0]["b"] == b)
    assert jnp.all(f1[1][0] == c)

    with pytest.raises(TypeError):
        f(general_tree)

    def g(x):
        return jax.tree_map(lambda u: u if eqx.is_array_like(u) else None, x)

    if api_version == 0:
        g = eqx.filter_jit(g, filter_spec=eqx.is_inexact_array)
    else:
        g = eqx.filter_jit(default=eqx.is_inexact_array)(g)

    assert jnp.all(a == g(a))
    g1 = g(array_tree)
    assert jnp.all(g1[0]["a"] == a)
    assert jnp.all(g1[0]["b"] == b)
    assert jnp.all(g1[1][0] == c)
    g2 = g(general_tree)
    assert _eq(g2[0], 1)
    assert _eq(g2[1], True)
    assert _eq(g2[2], None)
    assert jnp.all(g2[3]["a"] == a)
    assert _eq(g2[3]["tuple"][0], 2.0)
    assert jnp.all(g2[3]["tuple"][1] == b)
    assert jnp.all(g2[4] == c)
    assert _eq(g2[5], _mlp)

    def h(x):
        return jax.tree_map(lambda u: u if eqx.is_array_like(u) else None, x)

    if api_version == 0:
        h = eqx.filter_jit(h, filter_spec=eqx.is_array_like)
    else:
        h = eqx.filter_jit(h, default=eqx.is_array_like)

    assert jnp.all(a == h(a))
    h1 = h(array_tree)
    assert jnp.all(h1[0]["a"] == a)
    assert jnp.all(h1[0]["b"] == b)
    assert jnp.all(h1[1][0] == c)
    h2 = h(general_tree)
    assert _eq(h2[0], jnp.array(1))
    assert _eq(h2[1], jnp.array(True))
    assert _eq(h2[2], None)
    assert jnp.all(h2[3]["a"] == a)
    assert _eq(h2[3]["tuple"][0], jnp.array(2.0))
    assert jnp.all(h2[3]["tuple"][1] == b)
    assert jnp.all(h2[4] == c)
    assert _eq(h2[5], _mlp)


@pytest.mark.parametrize("api_version", (0, 1))
def test_filter_jit2(api_version, getkey):
    a = jrandom.normal(getkey(), (2, 3))
    b = jrandom.normal(getkey(), (3,))
    c = jrandom.normal(getkey(), (1, 4))
    general_tree = [
        1,
        True,
        object(),
        {"a": a, "tuple": (2.0, b)},
        c,
        eqx.nn.MLP(2, 2, 2, 2, key=getkey()),
    ]
    _mlp = jax.tree_map(lambda u: u if eqx.is_array_like(u) else None, general_tree[-1])

    spec = [
        True,
        True,
        False,
        {"a": True, "tuple": (False, True)},
        True,
        eqx.is_inexact_array,
    ]

    def f(x):
        return jax.tree_map(lambda u: u if eqx.is_array_like(u) else None, x)

    if api_version == 0:
        wrappers = [ft.partial(eqx.filter_jit, filter_spec=((spec,), {}))]
    else:
        wrappers = [eqx.filter_jit(args=(spec,)), eqx.filter_jit(kwargs=dict(x=spec))]

    for wrapper in wrappers:
        _f = wrapper(f)
        f1 = _f(general_tree)
        assert _eq(f1[0], jnp.array(1))
        assert _eq(f1[1], jnp.array(True))
        assert _eq(f1[2], None)
        assert jnp.all(f1[3]["a"] == a)
        assert _eq(f1[3]["tuple"][0], 2.0)
        assert jnp.all(f1[3]["tuple"][1] == b)
        assert jnp.all(f1[4] == c)
        assert _eq(f1[5], _mlp)


@pytest.mark.parametrize("api_version", (0, 1))
def test_num_traces(api_version):
    num_traces = 0

    def f(x):
        nonlocal num_traces
        num_traces += 1

    if api_version == 0:
        f = eqx.filter_jit(f, filter_spec=lambda _: True)
    else:
        f = eqx.filter_jit(default=True)(f)

    f(jnp.zeros(2))
    f(jnp.zeros(2))
    assert num_traces == 1

    f(jnp.zeros(3))
    f(jnp.zeros(3))
    assert num_traces == 2

    f([jnp.zeros(2)])
    f([jnp.zeros(2), jnp.zeros(3)])
    f([jnp.zeros(2), True])
    assert num_traces == 5

    num_traces = 0

    def g(x, y):
        nonlocal num_traces
        num_traces += 1

    if api_version == 0:
        g = eqx.filter_jit(g, filter_spec=([eqx.is_array_like, False], {}))
    else:
        g = eqx.filter_jit(args=[eqx.is_array_like, False])(g)

    g(jnp.zeros(2), True)
    g(jnp.zeros(2), False)
    assert num_traces == 2

    num_traces = 0

    def h(x, y, z, w):
        nonlocal num_traces
        num_traces += 1

    if api_version == 0:
        h = eqx.filter_jit(
            h, filter_spec=([False, {"a": True, "b": False}, False, False], {})
        )
    else:
        h = eqx.filter_jit(
            h, args=[False, {"a": True, "b": False}], kwargs=dict(z=False, w=False)
        )

    h(True, {"a": 1, "b": 1}, True, True)
    h(False, {"a": 1, "b": 1}, True, True)
    h(True, {"a": 1, "b": 0}, True, True)
    h(True, {"a": 1, "b": 1}, True, 2)
    h(True, {"a": 1, "b": 1}, 5, True)
    assert num_traces == 5
    h(True, {"a": 2, "b": 1}, True, True)
    assert num_traces == 5


def test_bound_method():
    num_traces = 0

    class M(eqx.Module):
        def method(self, x):
            nonlocal num_traces
            num_traces += 1
            return x + 1

    m = M()
    y = jnp.array(1.0)
    eqx.filter_jit(m.method)(y)
    eqx.filter_jit(m.method)(y)
    assert num_traces == 1


def test_callable_class():
    num_traces = 0

    class M(eqx.Module):
        def __call__(self, x):
            nonlocal num_traces
            num_traces += 1
            return x + 1

    m = M()
    y = jnp.array(1.0)
    eqx.filter_jit(m)(y)
    eqx.filter_jit(m)(y)
    assert num_traces == 1


@pytest.fixture
def log_compiles_config():
    """Setup and teardown of jax_log_compiles flag"""
    with jax.log_compiles(True):
        yield


def test_function_name_warning(log_compiles_config, caplog):
    """Test that the proper function names are used when compiling a function decorated with `filter_jit`"""

    @eqx.filter_jit
    def the_test_function_name(x):
        return x + 1

    # Trigger compile to log a warning message
    the_test_function_name(jnp.array(1.0))

    warning_text = caplog.text

    # Check that the warning message contains the function name
    assert "Finished XLA compilation of the_test_function_name in" in warning_text

    # Check that it works for filter_grad also
    @eqx.filter_jit
    @eqx.filter_grad
    def the_test_function_name_grad(x):
        return x + 1

    # Trigger compile to log a warning message
    the_test_function_name_grad(jnp.array(1.0))

    warning_text = caplog.text

    assert "Finished XLA compilation of the_test_function_name_grad in" in warning_text

    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def the_test_function_name_value_and_grad(x):
        return x + 1

    # Trigger compile to log a warning message
    the_test_function_name_value_and_grad(jnp.array(1.0))

    warning_text = caplog.text

    assert (
        "Finished XLA compilation of the_test_function_name_value_and_grad in"
        in warning_text
    )

    def wrapped_fun(y):
        pass

    def the_test_function_name(x, y):
        return x + y

    fun = eqx.filter_jit(
        ft.wraps(wrapped_fun)(ft.partial(the_test_function_name, jnp.array(1.0)))
    )

    fun(jnp.array(1.0))

    warning_text = caplog.text

    assert "Finished XLA compilation of wrapped_fun in" in warning_text
