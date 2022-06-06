import functools as ft
from typing import Optional, Sequence, Union

import jax
import jax.numpy as jnp

from ..custom_types import Array
from ..module import Module, static_field
from .array_utils import left_broadcast_to


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
        self.shape = shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = jnp.ones(shape) if elementwise_affine else None
        self.bias = jnp.zeros(shape) if elementwise_affine else None

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


class GroupNorm(Module):
    r"""
    Splits the first dimension ("channels") into groups of fixed size. Computes a mean
    and standard deviation over the contents of each group, and uses these to normalise
    the group. Optionally applies a channel-wise affine transformation afterwards.

    Given an input array $x$ of shape `(channels, ...)`, this layer splits this up into
    `groups`-many arrays $x_i$ each of shape `(channels/groups, ...)`, and for each one
    computes

    $$\frac{x_i - \mathbb{E}[x_i]}{\sqrt{\text{Var}[x_i] + \varepsilon}} * \gamma_i + \beta_i$$

    where $\gamma_i$, $\beta_i$ have shape `(channels/groups,)` if
    `channelwise_affine=True`, and $\gamma = 1$, $\beta = 0$ if
    `channelwise_affine=False`.

    ??? cite
        [Group Normalization](https://arxiv.org/abs/1803.08494)

        ```bibtex
        @article{wu2018group,
            author={Yuxin Wu and Kaiming He},
            title={Group Normalization},
            year={2018},
            journal={arXiv:1803.08494},
        }
        ```
    """

    groups: int = static_field()
    channels: int = static_field()
    eps: float = static_field()
    channelwise_affine: bool = static_field()
    weight: Array
    bias: Array

    def __init__(
        self,
        groups: int,
        channels: int,
        eps: float = 1e-5,
        channelwise_affine: bool = True,
        **kwargs,
    ):
        """**Arguments:**

        - `shape`: Input shape. May be left unspecified (e.g. just `None`) if
            `elementwise_affine=False`.
        - `groups`: The number of groups to split the input into.
        - `channels`: The number of input channels. May be left unspecified (e.g. just
            `None`) if `channelwise_affine=False`.
        - `eps`: Value added to denominator for numerical stability.
        - `channelwise_affine`: Whether the module has learnable affine parameters.
        """
        if channels % groups != 0:
            raise ValueError("The number of groups must divide the number of channels.")
        super().__init__(**kwargs)
        self.groups = groups
        self.channels = channels
        self.eps = eps
        self.channelwise_affine = channelwise_affine
        self.weight = jnp.ones(channels) if channelwise_affine else None
        self.bias = jnp.zeros(channels) if channelwise_affine else None

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**

        - `x`: A JAX array of shape `(channels, ...)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of shape `(channels, ...)`.
        """
        y = x.reshape(self.groups, self.channels // self.groups, *x.shape[1:])
        mean = jax.vmap(ft.partial(jnp.mean, keepdims=True))(y)
        variance = jax.vmap(ft.partial(jnp.var, keepdims=True))(y)
        inv = jax.lax.rsqrt(variance + self.eps)
        out = (y - mean) * inv
        out = out.reshape(x.shape)
        if self.channelwise_affine:
            weight = left_broadcast_to(self.weight, out.shape)
            bias = left_broadcast_to(self.bias, out.shape)
            out = weight * out + bias
        return out
