import functools as ft
import typing

import jax
import jax.numpy as jnp
import numpy as np

import equinox as eqx


def test_tuple():
    assert eqx.tree_pformat((1, 2)) == "(1, 2)"
    assert eqx.tree_pformat((1,)) == "(1,)"
    assert eqx.tree_pformat(()) == "()"


def test_list():
    assert eqx.tree_pformat([1, 2]) == ("[1, 2]")
    assert eqx.tree_pformat([1]) == "[1]"
    assert eqx.tree_pformat([]) == "[]"


def test_dict():
    assert eqx.tree_pformat({"a": 1, "b": 2}) == "{'a': 1, 'b': 2}"
    assert eqx.tree_pformat({"a": 1}) == "{'a': 1}"
    assert eqx.tree_pformat(dict()) == "{}"


def test_module(getkey):
    lin = eqx.nn.Linear(2, 2, key=getkey())
    mlp = eqx.nn.MLP(2, 2, 2, 2, key=getkey())
    eqx.tree_pformat(lin)
    eqx.tree_pformat(mlp)


def test_named_tuple():
    class M(typing.NamedTuple):
        a: int

    assert eqx.tree_pformat(M(1)) == "M(a=1)"


def test_jax_array():
    assert eqx.tree_pformat(jnp.array(1)) == "i32[]"
    assert eqx.tree_pformat(jnp.arange(12).reshape(3, 4)) == "i32[3,4]"
    assert (
        eqx.tree_pformat(jnp.array(1), short_arrays=False)
        == "DeviceArray(1, dtype=int32, weak_type=True)"
    )


def test_numpy_array():
    assert eqx.tree_pformat(np.array(1)) == "i64[](numpy)"
    assert eqx.tree_pformat(np.arange(12).reshape(3, 4)) == "i64[3,4](numpy)"
    assert eqx.tree_pformat(np.array(1), short_arrays=False) == "array(1)"


def test_builtins():
    assert eqx.tree_pformat(1) == "1"
    assert eqx.tree_pformat(0.1) == "0.1"
    assert eqx.tree_pformat(True) == "True"
    assert eqx.tree_pformat(1 + 1j) == "(1+1j)"
    assert eqx.tree_pformat("hi") == "'hi'"


def test_function():
    def f():
        pass

    @ft.wraps(f)
    def g():
        pass

    h = ft.partial(f)

    i = jax.custom_vjp(f)
    j = jax.custom_jvp(f)

    assert eqx.tree_pformat(f) == "<function f>"
    assert eqx.tree_pformat(g) == "<wrapped function f>"
    assert eqx.tree_pformat(h) == "<wrapped function f>"
    assert eqx.tree_pformat(i) == "<function f>"
    assert eqx.tree_pformat(j) == "<function f>"
