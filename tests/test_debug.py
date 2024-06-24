import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest


def test_backward_nan(capfd):
    @eqx.filter_custom_vjp
    def backward_nan(x):
        return x

    @backward_nan.def_fwd
    def backward_nan_fwd(perturbed, x):
        del perturbed
        return backward_nan(x), None

    @backward_nan.def_bwd
    def backward_nan_bwd(residual, grad_x, perturbed, x):
        del residual, grad_x, perturbed, x
        return jnp.nan

    @eqx.filter_jit
    @jax.grad
    def f(x, terminate):
        y = eqx.debug.backward_nan(x, name="foo", terminate=terminate)
        return backward_nan(y)

    capfd.readouterr()
    f(jnp.array(1.0), terminate=False)
    jax.effects_barrier()
    text, _ = capfd.readouterr()
    out_text1 = "foo:\n   primals=Array(1., dtype=float32)\ncotangents=Array(nan, dtype=float32)\n"  # noqa: E501
    out_text2 = "foo:\n   primals=array(1., dtype=float32)\ncotangents=array(nan, dtype=float32)\n"  # noqa: E501
    assert text in (out_text1, out_text2)

    with pytest.raises(Exception):
        f(jnp.array(1.0), terminate=True)


def test_check_dce(capfd):
    @jax.jit
    def f(x):
        a, _, _ = eqx.debug.store_dce((x**2, x + 1, "foobar"))
        return a

    f(1)
    capfd.readouterr()
    eqx.debug.inspect_dce()
    text, _ = capfd.readouterr()
    assert "(i32[], <DCE'd>, 'foobar')" in text


def test_max_traces():
    @jax.jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def f(x):
        return x + 1

    f(1)
    f(2)

    with pytest.raises(RuntimeError, match="can only be traced 1 times"):
        f(3.0)


def test_max_traces_clone(getkey):
    lin = eqx.nn.Linear(3, 4, key=getkey())
    lin = eqx.filter_jit(eqx.debug.assert_max_traces(lin, max_traces=2))
    leaves, treedef = jtu.tree_flatten(lin)
    lin2 = jtu.tree_unflatten(treedef, leaves)

    lin(jnp.array([1.0, 2.0, 3.0]))
    assert eqx.debug.get_num_traces(lin) == 1
    assert eqx.debug.get_num_traces(lin2) == 1
    with jax.numpy_dtype_promotion("standard"):
        lin2(jnp.array([1, 2, 3]))
        assert eqx.debug.get_num_traces(lin) == 2
        assert eqx.debug.get_num_traces(lin2) == 2

        with pytest.raises(RuntimeError, match="can only be traced 2 times"):
            lin(jnp.array([False, False, False]))
