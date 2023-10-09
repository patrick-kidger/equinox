import functools as ft
import warnings
from collections.abc import Sequence
from typing import Optional, overload, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .._custom_types import sentinel
from .._misc import left_broadcast_to
from .._module import field, Module
from ._stateful import State


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

    shape: tuple[int, ...] = field(static=True)
    eps: float = field(static=True)
    use_weight: bool = field(static=True)
    use_bias: bool = field(static=True)
    weight: Optional[Float[Array, "*shape"]]
    bias: Optional[Float[Array, "*shape"]]

    def __init__(
        self,
        shape: Union[int, Sequence[int]],
        eps: float = 1e-5,
        use_weight: bool = True,
        use_bias: bool = True,
        *,
        elementwise_affine: Optional[bool] = None,
        **kwargs,
    ):
        """**Arguments:**

        - `shape`: Shape of the input.
        - `eps`: Value added to denominator for numerical stability.
        - `use_weight`: Whether the module has learnable affine weights.
        - `use_bias`: Whether the module has learnable affine biases.
        - `elementwise_affine`: Deprecated alternative to `use_weight` and `use_bias`.
        """
        super().__init__(**kwargs)
        if isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple(shape)
        self.shape = shape
        self.eps = eps
        if elementwise_affine is not None:
            use_weight = elementwise_affine
            use_bias = elementwise_affine
            warnings.warn(
                "LayerNorm(elementwise_affine=...) is deprecated "
                "in favour of LayerNorm(use_weight=...) and LayerNorm(use_bias=...)"
            )
        self.use_weight = use_weight
        self.use_bias = use_bias
        self.weight = jnp.ones(shape) if use_weight else None
        self.bias = jnp.zeros(shape) if use_bias else None

    @overload
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        ...

    @overload
    def __call__(
        self, x: Array, state: State, *, key: Optional[PRNGKeyArray] = None
    ) -> tuple[Array, State]:
        ...

    @jax.named_scope("eqx.nn.LayerNorm")
    def __call__(
        self,
        x: Float[Array, "*shape"],
        state: State = sentinel,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Union[Array, tuple[Array, State]]:
        """**Arguments:**

        - `x`: A JAX array, with the same shape as the `shape` passed to `__init__`.
        - `state`: Ignored; provided for interchangability with the
            [`equinox.nn.BatchNorm`][] API.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        The output is a JAX array of the same shape as `x`.

        If `state` is passed, then a 2-tuple of `(output, state)` is returned. The state
        is passed through unchanged. If `state` is not passed, then just the output is
        returned.
        """
        if x.shape != self.shape:
            raise ValueError(
                "`LayerNorm(shape)(x)` must satisfy the invariant `shape == x.shape`"
                f"Received `shape={self.shape} and `x.shape={x.shape}`. You might need "
                "to replace `layer_norm(x)` with `jax.vmap(layer_norm)(x)`.\n"
                "\n"
                "If this is a new error for you, it might be because this became "
                "stricter in Equinox v0.11.0. Previously all that was required is that "
                "`x.shape` ended with `shape`. However, this turned out to be a "
                "frequent source of bugs, so we made the check stricter!"
            )
        mean = jnp.mean(x, keepdims=True)
        variance = jnp.var(x, keepdims=True)
        variance = jnp.maximum(0.0, variance)
        inv = jax.lax.rsqrt(variance + self.eps)
        out = (x - mean) * inv
        if self.use_weight:
            out = self.weight * out
        if self.use_bias:
            out = out + self.bias
        if state is sentinel:
            return out
        else:
            return out, state


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
    """  # noqa: E501

    groups: int = field(static=True)
    channels: Optional[int] = field(static=True)
    eps: float = field(static=True)
    channelwise_affine: bool = field(static=True)
    weight: Optional[Array]
    bias: Optional[Array]

    def __init__(
        self,
        groups: int,
        channels: Optional[int] = None,
        eps: float = 1e-5,
        channelwise_affine: bool = True,
        **kwargs,
    ):
        """**Arguments:**

        - `groups`: The number of groups to split the input into.
        - `channels`: The number of input channels. May be left unspecified (e.g. just
            `None`) if `channelwise_affine=False`.
        - `eps`: Value added to denominator for numerical stability.
        - `channelwise_affine`: Whether the module has learnable affine parameters.
        """
        if (channels is not None) and (channels % groups != 0):
            raise ValueError("The number of groups must divide the number of channels.")
        if (channels is None) and channelwise_affine:
            raise ValueError(
                "The number of channels should be specified if "
                "`channelwise_affine=True`"
            )
        super().__init__(**kwargs)
        self.groups = groups
        self.channels = channels
        self.eps = eps
        self.channelwise_affine = channelwise_affine
        self.weight = jnp.ones(channels) if channelwise_affine else None
        self.bias = jnp.zeros(channels) if channelwise_affine else None

    @overload
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        ...

    @overload
    def __call__(
        self, x: Array, state: State, *, key: Optional[PRNGKeyArray] = None
    ) -> tuple[Array, State]:
        ...

    @jax.named_scope("eqx.nn.GroupNorm")
    def __call__(
        self, x: Array, state: State = sentinel, *, key: Optional[PRNGKeyArray] = None
    ) -> Union[Array, tuple[Array, State]]:
        """**Arguments:**

        - `x`: A JAX array of shape `(channels, ...)`.
        - `state`: Ignored; provided for interchangability with the
            [`equinox.nn.BatchNorm`][] API.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        The output is a JAX array of shape `(channels, ...)`.

        If `state` is passed, then a 2-tuple of `(output, state)` is returned. The state
        is passed through unchanged. If `state` is not passed, then just the output is
        returned.
        """
        channels = x.shape[0]
        y = x.reshape(self.groups, channels // self.groups, *x.shape[1:])
        mean = jax.vmap(ft.partial(jnp.mean, keepdims=True))(y)
        variance = jax.vmap(ft.partial(jnp.var, keepdims=True))(y)
        variance = jnp.maximum(0.0, variance)
        inv = jax.lax.rsqrt(variance + self.eps)
        out = (y - mean) * inv
        out = out.reshape(x.shape)
        if self.channelwise_affine:
            weight = left_broadcast_to(self.weight, out.shape)  # pyright: ignore
            bias = left_broadcast_to(self.bias, out.shape)  # pyright: ignore
            out = weight * out + bias
        if state is sentinel:
            return out
        else:
            return out, state
