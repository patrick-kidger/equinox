from typing import Optional, Sequence, Union

import jax
import jax.numpy as jnp

from ..custom_types import Array
from ..module import Module, static_field


class LayerNorm(Module):
    r"""
    Computes a mean and standard deviation over the whole input array, and uses these
    to normalise the whole array. Optionally applies an elementwise affine
    transformation afterwards.

    Given an input array $x$, this layer computes

    $$\frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \varepsilon}} * \gamma + \beta$$

    where $\gamma$, $\beta$ have the same shape as $x$ if `elementwise_affine=True`,
    and $\gamma = 1$, $\beta = 0$ if `elementwise_affine=False`.

    ??? cite
        [Layer Normalization](https://arxiv.org/abs/1607.06450)

        ```bibtex
        @article{ba2016layer,
            author={Jimmy Lei Ba, Jamie Ryan Kriso, Geoffrey E. Hinton},
            title={Layer Normalization},
            year={2016},
            journal={arXiv:1607.06450},
        }
        ```

    !!! faq "FAQ"

        If you need to normalise over only some input dimensions, then this can be
        achieved by vmap'ing. For example the following will compute statistics over
        every dimension *except* the first:
        ```python
        layer = LayerNorm(...)
        array = jax.vmap(layer)(array)
        ```

    """

    shape: Union[None, int, Sequence[int]] = static_field()
    eps: float = static_field()
    elementwise_affine: bool = static_field()
    weight: Array
    bias: Array

    def __init__(
        self,
        shape: Union[None, int, Sequence[int]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        **kwargs,
    ):
        """**Arguments:**

        - `shape`: Input shape. May be left unspecified (e.g. just `None`) if
            `elementwise_affine=False`.
        - `eps`: Value added to denominator for numerical stability.
        - `elementwise_affine`: Whether the module has learnable affine parameters.
        """
        super().__init__(**kwargs)
        self.weight = jnp.ones(shape) if elementwise_affine else None
        self.bias = jnp.zeros(shape) if elementwise_affine else None
        self.shape = shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**

        - `x`: A JAX array whose shape is given by `shape`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of shape `shape`.
        """
        mean = jnp.mean(x, keepdims=True)
        variance = jnp.var(x, keepdims=True)
        inv = jax.lax.rsqrt(variance + self.eps)
        out = (x - mean) * inv
        if self.elementwise_affine:
            out = self.weight * out + self.bias
        return out
