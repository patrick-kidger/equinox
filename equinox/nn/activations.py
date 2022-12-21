from typing import Optional, Union

import jax.numpy as jnp
from jaxtyping import Array

from ..module import Module


class PReLU(Module):
    """PReLU activation function."""

    negative_slope: Array

    def __init__(
        self,
        init_alpha: Optional[Union[float, Array]] = 0.25,
    ):
        r"""**Arguments:**

        - `init_alpha`: The initial value $\alpha$ of negative slope.
            Value can accept a single float value $0.25$ by default or
            a JAX array of $\alpha_i$ values. The shape of JAX array is
            expected to be `(1,)` or `(in_channels,)`, where `in_channels`
            is the number of input channels.
        """

        if init_alpha is float:
            init_alpha = jnp.array((init_alpha,))

        self.negative_slope = init_alpha

    def __call__(self, x) -> Array:
        r"""**Arguments:**

        - `x`: The input.

        **Returns:**

        A JAX array of the input shape.
        """
        return jnp.where(x >= 0, x, self.negative_slope * x)
