import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .helpers import tree_allclose


def test_callback():
    sentinel1 = object()
    sentinel2 = object()

    def f(x, y):
        assert y is sentinel1
        return (x + 1).astype(np.float32), sentinel2

    out_struct = (jax.ShapeDtypeStruct((), jnp.float32), sentinel2)

    out = eqx.filter_pure_callback(
        f, jnp.array(1.0), sentinel1, result_shape_dtypes=out_struct
    )
    assert tree_allclose(out, (jnp.array(2.0), sentinel2))

    @eqx.filter_jit
    def g(x):
        return eqx.filter_pure_callback(f, x, sentinel1, result_shape_dtypes=out_struct)

    out = g(jnp.array(2.0))
    assert tree_allclose(out, (jnp.array(3.0), sentinel2))


def test_wrong():
    sentinel1 = object()
    sentinel2 = object()

    def f(x, y):
        assert y is sentinel1
        return x + 1, sentinel1

    out_struct = (jax.ShapeDtypeStruct((), jnp.float32), sentinel2)

    with pytest.raises(RuntimeError):
        eqx.filter_pure_callback(
            f, jnp.array(1.0), sentinel1, result_shape_dtypes=out_struct
        )

    @eqx.filter_jit
    def g(x):
        return eqx.filter_pure_callback(f, x, sentinel1, result_shape_dtypes=out_struct)

    with pytest.raises(RuntimeError):
        g(jnp.array(2.0))
