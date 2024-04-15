import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .._misc import default_int_dtype
from .._module import field, Module
from ._stateful import State, StateIndex


"""        key_state, value_state, index, state = self._get_cache(
            state, key_heads, value_heads
        )
        key_shape = key_state.shape
        value_shape = value_state.shape
        if key_shape != self.key_shape:
            raise ValueError(f"Expected key shape {self.key_shape}, got {key_shape}")
        if value_shape != self.value_shape:
            raise ValueError(
                f"Expected value shape {self.value_shape}, got {value_shape}"
            )
        return key_state, value_state, index, state
"""


class StandardKVCache(Module):
    key_shape: tuple[int, int, int] = field(static=True)
    value_shape: tuple[int, int, int] = field(static=True)

    autoregressive_index: StateIndex

    def __init__(
        self,
        key_shape: tuple[int, int, int],
        value_shape: tuple[int, int, int],
    ):
        self.key_shape = key_shape
        self.value_shape = value_shape

        def _make_cache(**_):
            _int = default_int_dtype()
            return jnp.empty(key_shape), jnp.empty(value_shape), jnp.zeros((), _int)

        self.autoregressive_index = StateIndex(_make_cache())

    def __call__(
        self,
        state: State,
        key_heads: Float[Array, "seq_length num_heads qk_size"],
        value_heads: Float[Array, "seq_length num_heads vo_size"],
    ) -> tuple[
        Float[Array, "state_length num_heads qk_size"],
        Float[Array, "state_length num_heads vo_size"],
        Int[Array, ""],
        State,
    ]:
        kv_seq_length = key_heads.shape[0]
        key_state, value_state, index = state.get(self.autoregressive_index)
        key_state = lax.dynamic_update_slice_in_dim(key_state, key_heads, index, axis=0)
        value_state = lax.dynamic_update_slice_in_dim(
            value_state, value_heads, index, axis=0
        )
        index = index + kv_seq_length
        state = state.set(self.autoregressive_index, (key_state, value_state, index))

        return key_state, value_state, index, state
