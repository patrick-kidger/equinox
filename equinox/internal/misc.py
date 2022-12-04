from typing import Tuple

import jax.numpy as jnp
from jaxtyping import Array


def left_broadcast_to(arr: Array, shape: Tuple[int, ...]) -> Array:
    arr = arr.reshape(arr.shape + (1,) * (len(shape) - arr.ndim))
    return jnp.broadcast_to(arr, shape)


class ContainerMeta(type):
    def __new__(cls, name, bases, dict):
        assert "reverse_lookup" not in dict
        _dict = {}
        reverse_lookup = []
        i = 0
        for key, value in dict.items():
            if key.startswith("__") and key.endswith("__"):
                _dict[key] = value
            else:
                _dict[key] = i
                reverse_lookup.append(value)
                i += 1
        _dict["reverse_lookup"] = reverse_lookup
        return super().__new__(cls, name, bases, _dict)

    def __instancecheck__(cls, instance):
        return isinstance(instance, int) or super().__instancecheck__(instance)

    def __getitem__(cls, item):
        return cls.reverse_lookup[item]

    def __len__(cls):
        return len(cls.reverse_lookup)
