import functools as ft

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def test_module(getkey):
    lin = eqx.nn.Linear(2, 2, key=getkey())
    mlp = eqx.nn.MLP(2, 2, 2, 2, key=getkey())
    eqx.tree_pformat(lin)
    eqx.tree_pformat(mlp)


def test_jax_array():
    assert eqx.tree_pformat(jnp.array(1)) == "weak_i32[]"
    assert eqx.tree_pformat(jnp.arange(12).reshape(3, 4)) == "i32[3,4]"
    array = "Array(1, dtype=int32, weak_type=True)"
    device_array = "DeviceArray(1, dtype=int32, weak_type=True)"
    assert eqx.tree_pformat(jnp.array(1), short_arrays=False) in (array, device_array)


def test_numpy_array():
    assert eqx.tree_pformat(np.array(1)) == "i64[](numpy)"
    assert eqx.tree_pformat(np.arange(12).reshape(3, 4)) == "i64[3,4](numpy)"
    assert eqx.tree_pformat(np.array(1), short_arrays=False) == "array(1)"


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
    assert eqx.tree_pformat(h) == "partial(<function f>)"
    assert eqx.tree_pformat(i) == "<function f>"
    assert eqx.tree_pformat(j) == "<function f>"


def test_struct_as_array():
    x = jax.ShapeDtypeStruct((2, 3), jax.numpy.float32)
    assert eqx.tree_pformat(x, struct_as_array=True) == "f32[2,3]"


def test_truncate_leaf():
    class Foo:
        pass

    foo = Foo()
    truncate_leaf = lambda x: x is foo
    assert eqx.tree_pformat([foo, 1], truncate_leaf=truncate_leaf) == "[Foo(...), 1]"
