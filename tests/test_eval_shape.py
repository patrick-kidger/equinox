import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp


def test_eval_shape(getkey):
    mlp = eqx.nn.MLP(1, 2, 5, 2, key=getkey())
    sentinel1 = object()
    sentinel2 = object()
    sentinel3 = object()

    def call(fn1, fn2, x, flag):
        if flag is sentinel1:
            return fn1(x), sentinel2
        else:
            return fn2(x), sentinel3

    y = jnp.array([1.0])
    z = jax.ShapeDtypeStruct(shape=y.shape, dtype=y.dtype)
    out1 = eqx.filter_eval_shape(call, mlp, jnn.relu, y, sentinel1)
    out2 = eqx.filter_eval_shape(call, mlp, jnn.relu, y, object())
    assert out1 == (jax.ShapeDtypeStruct(shape=(2,), dtype=y.dtype), sentinel2)
    assert out2 == (jax.ShapeDtypeStruct(shape=(1,), dtype=y.dtype), sentinel3)

    out3 = eqx.filter_eval_shape(call, mlp, jnn.relu, z, sentinel1)
    out4 = eqx.filter_eval_shape(call, mlp, jnn.relu, z, object())
    assert out3 == (jax.ShapeDtypeStruct(shape=(2,), dtype=y.dtype), sentinel2)
    assert out4 == (jax.ShapeDtypeStruct(shape=(1,), dtype=y.dtype), sentinel3)
