from typing import Optional, Union

import jax.numpy as jnp
from jaxtyping import Array

from .._module import Module


class PReLU(Module):
    """PReLU activation function.

    This is the elementwise function `x -> max(x, 0) + α * min(x, 0)`.
    This can be thought of as a leaky ReLU, with a learnt leak `α`.
    """

    negative_slope: Array

    def __init__(
        self,
        init_alpha: Optional[Union[float, Array]] = 0.25,
    ):
        r"""**Arguments:**

        - `init_alpha`: The initial value $\alpha$ of the negative slope.
            This should either be a `float` (default value is $0.25$), or
            a JAX array of $\alpha_i$ values. The shape of such a JAX array
            should be broadcastable to the input.
        """

        self.negative_slope = jnp.asarray(init_alpha)

    def __call__(self, x: Array) -> Array:
        r"""**Arguments:**

        - `x`: The input.

        **Returns:**

        A JAX array of the same shape as the input.
        """
        return jnp.where(x >= 0, x, self.negative_slope * x)
