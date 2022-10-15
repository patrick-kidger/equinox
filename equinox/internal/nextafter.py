import jax
import jax.numpy as jnp
from jaxtyping import Array


@jax.custom_jvp
def nextafter(x: Array) -> Array:
    """Returns the floating point number after `x`."""
    y = jnp.nextafter(x, jnp.inf)
    # Flush denormal to normal.
    # Our use for these is to handle jumps in the vector field. Typically that means
    # there will be an "if x > cond" condition somewhere. However JAX uses DAZ
    # (denormals-are-zero), which will cause this check to fail near zero:
    # `jnp.nextafter(0, jnp.inf) > 0` gives `False`.
    return jnp.where(x == 0, jnp.finfo(x.dtype).tiny, y)


nextafter.defjvps(lambda x_dot, _, __: x_dot)


@jax.custom_jvp
def prevbefore(x: Array) -> Array:
    """Returns the floating point number before `x`."""
    y = jnp.nextafter(x, jnp.NINF)
    return jnp.where(x == 0, -jnp.finfo(x.dtype).tiny, y)


prevbefore.defjvps(lambda x_dot, _, __: x_dot)
