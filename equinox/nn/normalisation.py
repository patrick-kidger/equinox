from typing import Optional, Sequence, Union

import jax
import jax.lax as lax
import jax.numpy as jnp

from ..custom_types import Array
from ..module import Module, static_field
from ..stateful import get_state, set_state, StateIndex


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
        *,
        key: "jax.random.PRNGKey",
        **kwargs,
    ):
        """**Arguments:**

        - `shape`: Input shape. May be left unspecified (e.g. just `None`) if
            `elementwise_affine=False`.
        - `eps`: Value added to denominator for numerical stability.
        - `elementwise_affine`: Whether the module has learnable affine parameters.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
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


class BatchNorm(Module):
    r"""Computes a mean and standard deviation over the batch and spatial
    dimensions of an array, and uses these to normalise the whole array. Optionally
    applies a channelwise affine transformation afterwards.

    Given an input array $x = [x_1, ... x_C]$ with $C$ channels, this layer computes

    $$\frac{x_i - \mathbb{E}[x_i]}{\sqrt{\text{Var}[x_i] + \varepsilon}} * \gamma_i + \beta_i$$

    for all $i$. Here $\gamma$, $\beta$ have shape $(C,)$ if `channelwise_affine=True`,
    and $\gamma = 1$, $\beta = 0$ if `channelwise_affine=False`. Here $*$ denotes
    elementwise multiplication. Expectations are computed over all spatial dimensions
    *and* over the batch dimension.

    !!! warning

        This layer must be used inside of a `vmap` or `pmap` with a matching
        `axis_name`. Not doing so will raise an error.
    """  # noqa: E501
    weight: Optional[Array["channels"]]
    bias: Optional[Array["channels"]]
    first_time_index: StateIndex
    state_index: StateIndex
    axis_name: str
    update_stats: bool
    channels: int = static_field()
    eps: float = static_field()
    channelwise_affine: bool = static_field()
    momentum: float = static_field()

    def __init__(
        self,
        axis_name: str,
        channels: int,
        eps: float = 1e-5,
        channelwise_affine: bool = True,
        momentum: float = 0.99,
        update_stats: bool = True,
        *,
        key: "jax.random.PRNGKey",
        **kwargs,
    ):
        """**Arguments:**

        - `axis_name`: The name of the batch axis to compute statistics over, as passed
            to `axis_name` in `jax.vmap` or `jax.pmap`.
        - `channels`: The number of channels in the input array.
        - `eps`: Value added to the denominator for numerical stability.
        - `channelwise_affine`: Whether the module has learnable channel-wise affine
            parameters.
        - `momentum`: The rate at which to obtain the running statistics (used during
            inference).
        - `update_stats`: Whether or not to update the running statistics. (Set to
            `False` at inference time.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        """

        super().__init__(**kwargs)

        if channelwise_affine:
            self.weight = jnp.ones((channels,))
            self.bias = jnp.zeros((channels,))
        else:
            self.weight = None
            self.bias = None
        self.first_time_index = StateIndex()
        self.state_index = StateIndex()
        self.update_stats = update_stats
        self.axis_name = axis_name
        self.channels = channels
        self.eps = eps
        self.channelwise_affine = channelwise_affine
        self.momentum = momentum

        set_state(self.first_time_index, jnp.array(True))

    def __call__(
        self,
        x: Array,
        *,
        key: Optional["jax.random.PRNGKey"] = None,
        update_stats: Optional[bool] = None,
    ) -> Array:
        """**Arguments:**

        - `x`: A JAX array of shape `(channels, dim_1, ..., dim_N)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        - `update_stats`: As per [`equinox.nn.BatchNorm.__init__`][]. If `True` or
            `False` then it will take priority over `self.update_stats`. If `None` then
            the value from `self.update_stats` will be used.

        **Returns:**

        A JAX array of shape `(channels, dim_1, ..., dim_N)`.

        **Raises:**

        A `NameError` if no `vmap`s are placed around this operation, or if this vmap
        does not have a matching `axis_name`.
        """

        def _stats(y):
            mean = jnp.mean(y)
            mean = lax.pmean(mean, self.axis_name)
            var = jnp.mean((y - mean) ** 2)
            var = lax.pmean(var, self.axis_name)
            return mean, var

        batch_state = jax.vmap(_stats)(x)

        first_time = get_state(self.first_time_index, like=jnp.array(False))

        running_state = lax.cond(
            first_time,
            lambda: batch_state,
            lambda: get_state(self.state_index, like=batch_state),
        )
        set_state(self.first_time_index, jnp.array(False))

        batch_mean, batch_var = batch_state
        running_mean, running_var = running_state
        running_mean = (1 - self.momentum) * batch_mean + self.momentum * running_mean
        running_var = (1 - self.momentum) * batch_var + self.momentum * running_var
        if self.update_stats:
            set_state(self.state_index, (running_mean, running_var))

        def _norm(y, m, v, w, b):
            out = (y - m) / jnp.sqrt(v + self.eps)
            if self.channelwise_affine:
                out = out * w + b
            return out

        return jax.vmap(_norm)(x, running_mean, running_var, self.weight, self.bias)
