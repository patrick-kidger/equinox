from typing import Tuple, Union

import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Int, PyTree, Shaped


def at_set(
    xs: PyTree[Shaped[Array, " dim *_rest"]],
    i: Union[int, Int[Array, ""]],
    x: PyTree[Shaped[Array, " *_rest"]],
):
    """Like `xs.at[i].set(x)`. Used for updating state during a loop.

    Differences to using `xs.at[i].step(x)`:
    - It uses `lax.dynamic_update_index_in_dim` rather than `scatter`. This will be
        lowered to a true in-place update when a scatter sometimes isn't.
    - It operates on PyTrees for `x` and `xs`.
    """
    return jtu.tree_map(
        lambda y, ys: lax.dynamic_update_index_in_dim(ys, y, i, 0), x, xs
    )


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
