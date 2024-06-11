from collections.abc import Callable

import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .._misc import default_int_dtype
from .._module import field, Module
from ._stateful import State, StateIndex


KVCacheCallable = Callable[
    [
        Float[Array, "seq_length num_heads qk_size"],
        Float[Array, "seq_length num_heads vo_size"],
        State,
    ],
    tuple[
        Float[Array, "state_length num_heads qk_size"],
        Float[Array, "state_length num_heads vo_size"],
        Int[Array, ""],
        State,
    ],
]


class StandardKVCache(Module):
    key_shape: tuple[int, int, int] = field(static=True)
    value_shape: tuple[int, int, int] = field(static=True)

    autoregressive_index: StateIndex

    def __init__(
        self,
        state_length: int,
        num_heads: int,
        key_size: int,
        value_size: int,
    ):
        self.key_shape = state_length, num_heads, key_size
        self.value_shape = state_length, num_heads, value_size

        self.autoregressive_index = StateIndex(
            (
                lambda _: jnp.empty(self.key_shape),
                jnp.empty(self.value_shape),
                jnp.zeros((), default_int_dtype()),
            ),
        )

    def __call__(
        self,
        key_heads: Float[Array, "seq_length num_heads qk_size"],
        value_heads: Float[Array, "seq_length num_heads vo_size"],
        state: State,
    ) -> tuple[
        Float[Array, "state_length num_heads qk_size"],
        Float[Array, "state_length num_heads vo_size"],
        Int[Array, ""],
        State,
    ]:
        kv_seq_length, _, _ = key_heads.shape
        key_state, value_state, index = state.get(self.autoregressive_index)
        key_state = lax.dynamic_update_slice_in_dim(key_state, key_heads, index, axis=0)
        value_state = lax.dynamic_update_slice_in_dim(
            value_state, value_heads, index, axis=0
        )
        index = index + kv_seq_length
        state = state.set(
            self.autoregressive_index, (key_state, value_state, index + kv_seq_length)
        )

        return key_state, value_state, index, state
