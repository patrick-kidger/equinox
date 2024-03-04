import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import pytest


# Currently failing due to a bug in tf2onnx
@pytest.mark.skip
def test_onnx_export():
    @jax.vmap
    def fn(x, y):
        x = x + 1
        y = eqxi.unvmap_any(y)
        return jnp.where(y, x, 1)

    onnx_fn = eqxi.to_onnx(fn)
    args = jnp.array([1, 2]), jnp.array([True, False])
    onnx_fn(*args)
