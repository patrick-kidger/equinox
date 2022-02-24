from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from ..custom_types import Array
from ..module import Module, static_field


_shape_t = Union[int, List[int]]


class LayerNorm(Module):
    """Layer Normalization as described in https://arxiv.org/abs/1607.06450"""

    normalized_shape: Union[int, Tuple[int], List[int]] = static_field()
    eps: float = static_field()
    elementwise_affine: bool = static_field()
    weight: Array
    bias: Array

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        *,
        key: "jax.random.PRNGKey"
    ):
        """**Arguments:**
        - `normalized_shape`: Input shape.
        - `eps`: Value added to denominator for numerical stability. Default: `1e-5`.
        - `elementwise_affine`: Whether the module has learnable affine parameters. Default: `True`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        """
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = jnp.ones(self.normalized_shape)
        self.bias = jnp.zeros(self.normalized_shape)

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**

        - `x`: A JAX array of shape `normalized_shape`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of shape `normalized_shape`.
        """
        mean = jnp.mean(x, keepdims=True)
        variance = jnp.var(x, keepdims=True)
        if self.elementwise_affine:
            scale = self.weight
            offset = self.bias
        else:
            scale = jnp.array(1.0, dtype=x.dtype)
            offset = jnp.array(0.0, dtype=x.dtype)
        scale = jnp.broadcast_to(scale, x.shape)
        offset = jnp.broadcast_to(offset, x.shape)
        mean = jnp.broadcast_to(mean, x.shape)
        eps = jax.lax.convert_element_type(self.eps, variance.dtype)
        inv = scale * jax.lax.rsqrt(variance + eps)
        return inv * (x - mean) + offset
