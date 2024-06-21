from typing import Any

import jax
import jax.core
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray


def left_broadcast_to(arr: Array, shape: tuple[int, ...]) -> Array:
    arr = arr.reshape(arr.shape + (1,) * (len(shape) - arr.ndim))
    return jnp.broadcast_to(arr, shape)


def currently_jitting():
    return isinstance(jnp.array(1) + 1, jax.core.Tracer)


def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32


def default_init(
    key: PRNGKeyArray, shape: tuple[int, ...], dtype: Any, lim: float
) -> jax.Array:
    if jnp.issubdtype(dtype, jnp.complexfloating):
        # only two possible complex dtypes, jnp.complex64 or jnp.complex128
        real_dtype = jnp.float32 if dtype == jnp.complex64 else jnp.float64
        rkey, ikey = jrandom.split(key, 2)
        real = jrandom.uniform(rkey, shape, real_dtype, minval=-lim, maxval=lim)
        imag = jrandom.uniform(ikey, shape, real_dtype, minval=-lim, maxval=lim)
        return real.astype(dtype) + 1j * imag.astype(dtype)
    else:
        return jrandom.uniform(key, shape, dtype, minval=-lim, maxval=lim)
