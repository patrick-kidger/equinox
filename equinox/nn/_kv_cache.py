from collections.abc import Callable

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .._misc import default_floating_dtype
from .._module import field, Module
from ._stateful import State, StateIndex


KVCacheCallable = Callable[
    [
        Float[Array, "seq_length num_heads qk_size"],
        Float[Array, "seq_length num_heads vo_size"],
        Int[Array, ""],
        State,
    ],
    tuple[
        Float[Array, "state_length num_heads qk_size"],
        Float[Array, "state_length num_heads vo_size"],
        State,
    ],
]


class StandardKVCache(Module):
    """
    A class to manage the key and value caches for a transformer
    model with autoregressive decoding.
    """

    key_shape: tuple[int, int, int] = field(static=True)
    value_shape: tuple[int, int, int] = field(static=True)

    autoregressive_index: StateIndex

    def __init__(
        self,
        state_length: int,
        num_heads: int,
        key_size: int,
        value_size: int,
        dtype=None,
    ):
        r"""**Arguments:**

        - `state_length`: Refers to the maximum sequence length
        - `num_heads`: Number of parallel attention heads $h$.
        - `key_size`: Number of input channels for key $K$.
        - `value_size`: Number of input channels for value $V$.
        - `dtype` (optional): The data type of the KV caches.

        """
        dtype = default_floating_dtype() if dtype is None else dtype
        self.key_shape = state_length, num_heads, key_size
        self.value_shape = state_length, num_heads, value_size

        self.autoregressive_index = StateIndex(
            (
                jnp.empty(self.key_shape, dtype=dtype),
                jnp.empty(self.value_shape, dtype=dtype),
            ),
        )

    @jax.named_scope("eqx.nn.StandardKVCache")
    def __call__(
        self,
        key_heads: Float[Array, "seq_length num_heads qk_size"],
        value_heads: Float[Array, "seq_length num_heads vo_size"],
        index: Int[Array, ""],
        state: State,
    ) -> tuple[
        Float[Array, "state_length num_heads qk_size"],
        Float[Array, "state_length num_heads vo_size"],
        State,
    ]:
        """**Arguments:**

        - `key_heads`: The new key heads to be added to the cache
        - `value_heads`: The new value heads to be added to the cache
        - `state`: The current state containing the index for autoregressive decoding

        **Returns:**

        A tuple (key_state, value_state, index, state) containing the updated keys
        and values as well as the index and the new state.
        The shape of `key_state` is `(state_length num_heads qk_size)`
        and the shape of `value_state` is `(state_length num_heads vo_size)`.

        """
        kv_seq_length, _, _ = key_heads.shape
        key_state, value_state = state.get(self.autoregressive_index)
        key_state = lax.dynamic_update_slice_in_dim(key_state, key_heads, index, axis=0)
        value_state = lax.dynamic_update_slice_in_dim(
            value_state, value_heads, index, axis=0
        )
        state = state.set(self.autoregressive_index, (key_state, value_state))

        return key_state, value_state, state
