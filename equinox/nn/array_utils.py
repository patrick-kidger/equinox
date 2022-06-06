from typing import Tuple

from ..custom_types import Array


def left_broadcast_to(arr: Array, shape: Tuple[int]):
    return arr.reshape(shape + (1,) * (len(shape) - arr.ndim))
