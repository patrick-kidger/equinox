import jax
import jax.numpy as jnp
import pytest

import equinox as eqx


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
    assert (
        text
        == "foo:\n   primals=array(1., dtype=float32)\ncotangents=array(nan, dtype=float32)\n"  # noqa: E501
    )

    with pytest.raises(Exception):
        f(jnp.array(1.0), terminate=True)


def test_check_dce(capfd):
    @jax.jit
    def f(x):
        a, _ = eqx.debug.store_dce((x**2, x + 1))
        return a

    f(1)
    capfd.readouterr()
    eqx.debug.inspect_dce()
    text, _ = capfd.readouterr()
    assert "(i32[], <DCE'd>)" in text
