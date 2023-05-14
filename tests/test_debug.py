import jax
import jax.numpy as jnp
import pytest

import equinox as eqx
import equinox.internal as eqxi


def test_backward_nan(capfd):
    @eqx.filter_custom_vjp
    def backward_nan(x):
        return x

    def backward_nan_fwd(x):
        return backward_nan(x), None

    def backward_nan_bwd(_, x, grad_x):
        return jnp.nan

    backward_nan.defvjp(backward_nan_fwd, backward_nan_bwd)

    @eqx.filter_jit
    @jax.grad
    def f(x, terminate):
        y = eqxi.debug_backward_nan(x, name="foo", terminate=terminate)
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
