from typing import Tuple

import jax.numpy as jnp

from ..custom_types import Array


def left_broadcast_to(arr: Array, shape: Tuple[int]):
    arr = arr.reshape(arr.shape + (1,) * (len(shape) - arr.ndim))
    return jnp.broadcast_to(arr, shape)
