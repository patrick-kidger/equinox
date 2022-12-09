import jax
import jax.numpy as jnp

import equinox.internal as eqxi


def test_onnx_export():
    @jax.vmap
    def fn(x, y):
        x = x + 1
        y = eqxi.unvmap_any(y)
        return jnp.where(y, x, 1)

    onnx_fn = eqxi.to_onnx(fn, wrapper_prims=[eqxi.unvmap_any_p])
    args = jnp.array([1, 2]), jnp.array([True, False])
    onnx_fn(*args)
