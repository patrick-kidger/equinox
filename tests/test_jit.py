import warnings
from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import pytest

from .helpers import tree_allclose


# We can't just use `lambda x: x` or any function just rearrange without modification
# because JAX simplifies this away to an empty XLA computation.


@pytest.mark.parametrize("donate", ("all", "none"))
def test_filter_jit(donate, getkey):
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
    mlp_add = jtu.tree_map(lambda u: u + 1 if eqx.is_array(u) else u, general_tree[-1])

    array_tree_ = jtu.tree_map(jnp.copy, array_tree)
    general_tree_ = jtu.tree_map(
        lambda x: jnp.copy(x) if eqx.is_array(x) else x, general_tree
    )

    @eqx.filter_jit(donate=donate)
    def f(x):
        x = jtu.tree_map(lambda x: x + 1 if eqx.is_array(x) else x, x)
        return x

    assert jnp.all(a + 1 == f(jnp.copy(a)))
    f1 = f(array_tree_)
    assert jnp.all(f1[0]["a"] == a + 1)
    assert jnp.all(f1[0]["b"] == b + 1)
    assert jnp.all(f1[1][0] == c + 1)
    f2 = f(general_tree_)
    assert tree_allclose(f2[0], 1)
    assert tree_allclose(f2[1], True)
    assert tree_allclose(f2[2], general_tree[2])
    assert jnp.all(f2[3]["a"] == a + 1)
    assert tree_allclose(f2[3]["tuple"][0], 2.0)
    assert jnp.all(f2[3]["tuple"][1] == b + 1)
    assert jnp.all(f2[4] == c + 1)
    assert tree_allclose(f2[5], mlp_add)


def test_num_traces():
    num_traces = 0

    def f(x):
        nonlocal num_traces
        num_traces += 1

    f = eqx.filter_jit(f)

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

    g = eqx.filter_jit(g)

    g(jnp.zeros(2), True)
    g(jnp.zeros(2), False)
    assert num_traces == 2

    num_traces = 0

    def h(x, y, z, w):
        nonlocal num_traces
        num_traces += 1

    h = eqx.filter_jit(h)

    h(True, {"a": 1, "b": 1}, True, True)
    h(False, {"a": 1, "b": 1}, True, True)
    h(True, {"a": 1, "b": 0}, True, True)
    h(True, {"a": 1, "b": 1}, True, 2)
    h(True, {"a": 1, "b": 1}, 5, True)
    assert num_traces == 5
    h(True, {"a": 1, "b": 1}, True, True)
    assert num_traces == 5
    h(True, {"a": 2, "b": 1}, True, True)
    assert num_traces == 6


@pytest.mark.parametrize("call", [False, True])
@pytest.mark.parametrize("outer", [False, True])
def test_methods(call, outer):
    num_traces = 0

    class M(eqx.Module):
        increment: Union[int, jax.Array]

        if call:

            def __call__(self, x):
                nonlocal num_traces
                num_traces += 1
                return x + self.increment

            if not outer:
                __call__ = eqx.filter_jit(__call__)
        else:

            def method(self, x):
                nonlocal num_traces
                num_traces += 1
                return x + self.increment

            if not outer:
                method = eqx.filter_jit(method)

    def run(_m):
        y = jnp.array(1.0)
        if call:
            if outer:
                return eqx.filter_jit(_m)(y)
            else:
                return _m(y)
        else:
            if outer:
                return eqx.filter_jit(_m.method)(y)
            else:
                return _m.method(y)

    m = M(1)
    assert run(m) == 2
    assert run(m) == 2
    assert num_traces == 1
    n = M(2)
    assert run(n) == 3
    assert run(n) == 3
    assert num_traces == 2
    o = M(jnp.array(1))
    p = M(jnp.array(2))
    assert run(o) == 2
    assert run(p) == 3
    assert num_traces == 3


def test_args_kwargs():
    num_traces = 0

    @eqx.filter_jit
    def f(*args, **kwargs):
        nonlocal num_traces
        num_traces += 1
        return kwargs["x"]

    assert f(x=jnp.array(2)) == 2
    assert f(x=jnp.array(3)) == 3
    assert num_traces == 1

    assert f(x=jnp.array(3), y=jnp.array(4)) == 3
    assert num_traces == 2

    @eqx.filter_jit
    def h(*args, **kwargs):
        nonlocal num_traces
        num_traces += 1
        return args[0]

    assert h(1, 2) == 1  # check we can use other args


def test_jit_jit():
    num_traces = 0

    @eqx.filter_jit
    @eqx.filter_jit
    def f(x):
        nonlocal num_traces
        num_traces += 1
        return x + 1

    assert f(jnp.array(1)) == 2
    assert f(jnp.array(2)) == 3
    assert num_traces == 1

    @eqx.filter_jit
    def g(x):
        nonlocal num_traces
        num_traces += 1
        return x + 1

    assert eqx.filter_jit(g)(jnp.array(1)) == 2
    assert eqx.filter_jit(g)(jnp.array(2)) == 3
    assert num_traces == 2


def test_jit_grad():
    num_traces = 0

    def f(x):
        nonlocal num_traces
        num_traces += 1
        return x + 1

    assert eqx.filter_jit(eqx.filter_grad(f))(jnp.array(1.0)) == 1
    assert eqx.filter_jit(eqx.filter_grad(f))(jnp.array(2.0)) == 1
    assert num_traces == 1

    assert eqx.filter_jit(eqx.filter_value_and_grad(f))(jnp.array(1.0)) == (2, 1)
    assert eqx.filter_jit(eqx.filter_value_and_grad(f))(jnp.array(2.0)) == (3, 1)
    assert num_traces == 2


def test_jit_vmap():
    num_traces = 0

    def f(x):
        nonlocal num_traces
        num_traces += 1
        return x + 1

    out = eqx.filter_jit(eqx.filter_vmap(f))(jnp.array([1, 2]))
    assert tree_allclose(out, jnp.array([2, 3]))
    assert num_traces == 1

    out = eqx.filter_jit(eqx.filter_vmap(f))(jnp.array([2, 3]))
    assert tree_allclose(out, jnp.array([3, 4]))
    assert num_traces == 1


@pytest.fixture
def log_compiles_config():
    """Setup and teardown of jax_log_compiles flag"""
    with jax.log_compiles(True):
        yield


def test_wrap_jax_partial(getkey):
    def f(x, y):
        return x + y

    g = jtu.Partial(f, jrandom.normal(getkey(), ()))
    eqx.filter_jit(g)


def test_buffer_donation_function():
    @eqx.filter_jit(donate="all")
    def f(x):
        return x + 1

    x = jnp.array(0.0)
    old_p = x.unsafe_buffer_pointer()
    new_x = f(x)
    assert tree_allclose(new_x, jnp.array(1.0))
    assert x.is_deleted()
    assert new_x.unsafe_buffer_pointer() == old_p


def test_buffer_donation_function_except_first():
    @eqx.filter_jit(donate="warn-except-first")
    def f(x, y):
        return x + y

    x = jnp.array(0.0)
    y = jnp.array(1.0)
    old_p = y.unsafe_buffer_pointer()
    new_x = f(x, y)
    assert tree_allclose(new_x, jnp.array(1.0))
    assert not x.is_deleted()
    assert y.is_deleted()
    assert new_x.unsafe_buffer_pointer() == old_p


def test_buffer_donation_method(getkey):
    num_traces = 0

    class M(eqx.Module):
        buffer: jax.Array
        mlp: eqx.nn.MLP

        def __init__(self, width: int, depth: int):
            self.mlp = eqx.nn.MLP(width, width, width, depth, key=getkey())
            self.buffer = jnp.zeros((100000,))

        @eqx.filter_jit(donate="all")
        def __call__(self, x):
            nonlocal num_traces
            num_traces += 1

            return self.mlp(x), eqx.tree_at(
                lambda s: s.buffer, self, self.buffer.at[0].add(1)
            )

    m = M(10, 3)
    old_m_p = jtu.tree_map(
        lambda x: x.unsafe_buffer_pointer()
        if hasattr(x, "unsafe_buffer_pointer")
        else None,
        m,
    )

    _, new_m = m(jnp.ones((10,)))
    _, new_m = new_m(jnp.ones((10,)))
    _, new_m = new_m(jnp.ones((10,)))
    assert num_traces == 1

    assert tree_allclose(new_m.buffer[0], jnp.array(3.0))
    assert m.buffer.is_deleted()
    assert new_m.buffer.unsafe_buffer_pointer() == old_m_p.buffer

    old_m_deleted_flag = jtu.tree_map(
        lambda x: x.is_deleted() if hasattr(x, "is_deleted") else None, m
    )
    for flag in jtu.tree_leaves(old_m_deleted_flag):
        assert flag is True or flag is None


def test_buffer_donation_instance(getkey):
    num_traces = 0

    class F(eqx.Module):
        buffer: jax.Array
        mlp: eqx.nn.MLP

        def __init__(self, width: int, depth: int):
            self.mlp = eqx.nn.MLP(width, width, width, depth, key=getkey())
            self.buffer = jnp.zeros((100000,))

        def __call__(self, x):
            nonlocal num_traces
            num_traces += 1
            return self.mlp(x), eqx.tree_at(
                lambda s: s.buffer, self, self.buffer.at[0].add(1)
            )

    f = F(10, 3)
    jit_f = eqx.filter_jit(f, donate="all")
    _, new_f = jit_f(jnp.ones((10,)))
    assert num_traces == 1
    assert f.buffer.is_deleted()

    jit_f = eqx.filter_jit(new_f, donate="all")
    _, new_f_2 = jit_f(jnp.ones((10,)))

    assert num_traces == 1
    assert new_f.buffer.is_deleted()


def test_donation_warning():
    """Test the warning when compiling a function decorated with `filter_jit`"""

    @eqx.filter_jit
    def f(x, y):
        return x + jnp.sum(y)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f(jnp.array(1.0), jnp.ones((10,)))

    @eqx.filter_jit(donate="warn")
    def g(x, y):
        return x + jnp.sum(y)

    with pytest.warns(
        UserWarning, match=r"Some donated buffers were not usable*"
    ) as record:
        g(jnp.array(1.0), jnp.ones((10,)))

    assert len(record) == 1


# Issue 325
@pytest.mark.parametrize("donate", ("all", "all-except-first", "none"))
def test_aot_compilation(donate):
    def f(x, y):
        return 2 * x + y

    x, y = jnp.array(3), 4
    lowered = eqx.filter_jit(f, donate=donate).lower(x, y)
    lowered.as_text()
    compiled = lowered.compile()
    compiled(x, y)


# Issue 625
@pytest.mark.parametrize("donate", ("all", "all-except-first", "none"))
def test_aot_compilation_kwargs(donate):
    def f(x, y, **kwargs):
        return 2 * x + y

    x, y = jnp.array(3), 4
    lowered = eqx.filter_jit(f, donate=donate).lower(x, y, test=123)
    lowered.as_text()
    compiled = lowered.compile()
    compiled(x, y, test=123)
