import jax
import jax.core
import jax.numpy as jnp
from jaxtyping import Array


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
