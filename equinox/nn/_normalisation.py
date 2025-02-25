import functools as ft
import warnings
from collections.abc import Sequence
from typing import Optional, overload, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .._custom_types import sentinel
from .._misc import default_floating_dtype, left_broadcast_to
from .._module import field, Module
from ._misc import named_scope
from ._stateful import State


class LayerNorm(Module, strict=True):
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
        dtype=None,
        *,
        elementwise_affine: Optional[bool] = None,
    ):
        """**Arguments:**

        - `shape`: Shape of the input.
        - `eps`: Value added to denominator for numerical stability.
        - `use_weight`: Whether the module has learnable affine weights.
        - `use_bias`: Whether the module has learnable affine biases.
        - `dtype`: The dtype to use for the weight and the bias in this layer if
            `use_weight` or `use_bias` is set to `True`.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `elementwise_affine`: Deprecated alternative to `use_weight` and `use_bias`.
        """
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
        self.weight = jnp.ones(shape, dtype=dtype) if use_weight else None
        self.bias = jnp.zeros(shape, dtype=dtype) if use_bias else None

    @overload
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array: ...

    @overload
    def __call__(
        self, x: Array, state: State, *, key: Optional[PRNGKeyArray] = None
    ) -> tuple[Array, State]: ...

    @named_scope("eqx.nn.LayerNorm")
    def __call__(
        self,
        x: Float[Array, "*shape"],
        state: State = sentinel,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Union[Array, tuple[Array, State]]:
        """**Arguments:**

        - `x`: A JAX array, with the same shape as the `shape` passed to `__init__`.
        - `state`: Ignored; provided for interchangeability with the
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
        orig_dtype = x.dtype
        with jax.numpy_dtype_promotion("standard"):
            dtype = jnp.result_type(x.dtype, jnp.float32)

        x = x.astype(dtype)
        mean = jnp.mean(x, keepdims=True)
        variance = jnp.var(x, keepdims=True)
        variance = jnp.maximum(0.0, variance)
        inv = jax.lax.rsqrt(variance + self.eps)
        out = (x - mean) * inv
        if self.use_weight:
            out = self.weight.astype(dtype) * out  # pyright: ignore
        if self.use_bias:
            out = out + self.bias.astype(dtype)  # pyright: ignore
        if state is sentinel:
            return out.astype(orig_dtype)
        else:
            return out.astype(orig_dtype), state


class GroupNorm(Module, strict=True):
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
        dtype=None,
    ):
        """**Arguments:**

        - `groups`: The number of groups to split the input into.
        - `channels`: The number of input channels. May be left unspecified (e.g. just
            `None`) if `channelwise_affine=False`.
        - `eps`: Value added to denominator for numerical stability.
        - `channelwise_affine`: Whether the module has learnable affine parameters.
        - `dtype`: The dtype to use for the weight and the bias in this layer if
            `channelwise_affine` is set to `True`.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        """
        if (channels is not None) and (channels % groups != 0):
            raise ValueError("The number of groups must divide the number of channels.")
        if (channels is None) and channelwise_affine:
            raise ValueError(
                "The number of channels should be specified if "
                "`channelwise_affine=True`"
            )
        self.groups = groups
        self.channels = channels
        self.eps = eps
        self.channelwise_affine = channelwise_affine
        self.weight = jnp.ones(channels, dtype=dtype) if channelwise_affine else None
        self.bias = jnp.zeros(channels, dtype=dtype) if channelwise_affine else None

    @overload
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array: ...

    @overload
    def __call__(
        self, x: Array, state: State, *, key: Optional[PRNGKeyArray] = None
    ) -> tuple[Array, State]: ...

    @named_scope("eqx.nn.GroupNorm")
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

        orig_dtype = x.dtype
        with jax.numpy_dtype_promotion("standard"):
            dtype = jnp.result_type(x.dtype, jnp.float32)

        x = x.astype(dtype)
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
            out = weight.astype(dtype) * out + bias.astype(dtype)
        if state is sentinel:
            return out.astype(orig_dtype)
        else:
            return out.astype(orig_dtype), state


class RMSNorm(Module, strict=True):
    r"""
    A simplified version of LayerNorm which rescales the inputs, but does not center
    them. Optionally applies a learned reweighting of the transformed array afterward.

    Given an input array $x$, this layer computes

    $$\frac{x}{\sqrt{\varepsilon + \frac{1}{n}\Vert x \Vert^2_2}} \gamma + \beta$$

    where $\Vert x \Vert^2_2 = \sum_{i=1}^n x_i^2$, $n = \dim(x)$, and $\gamma$ is a
    learned array with the same shape as $x$ if `use_weight=True`, or
    $\gamma = 1$ if `use_weight=False`, as proposed in
    [this paper](https://browse.arxiv.org/abs/2307.14995). `\beta` is an optional bias
    term.

    ??? cite

        [Root Mean Square Layer Normalization](https://browse.arxiv.org/abs/1910.07467)

        ```bibtex
        @article{zhang2019root,
            title={Root Mean Square Layer Normalization},
            author={Biao Zhang and Rico Sennrich},
            year={2019},
            journal={arXiv:1910.07467}
        }
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
        dtype=None,
    ):
        """**Arguments:**

        - `shape`: Shape of the input.
        - `eps`: Value added to denominator for numerical stability.
        - `use_weight`: Whether the module has learnable affine weights.
        - `use_bias`: Whether the module has learnable affine shift.
        - `dtype`: The dtype to use for the weight and the bias in this layer if
            `use_weight` or `use_bias` is set to `True`.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        if isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple(shape)
        self.shape = shape
        self.eps = eps
        self.use_weight = use_weight
        self.use_bias = use_bias
        self.weight = jnp.ones(shape, dtype=dtype) if use_weight else None
        self.bias = jnp.zeros(shape, dtype=dtype) if use_bias else None

    @overload
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array: ...

    @overload
    def __call__(
        self, x: Array, state: State, *, key: Optional[PRNGKeyArray] = None
    ) -> tuple[Array, State]: ...

    @named_scope("eqx.nn.RMSNorm")
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
                "`RMSNorm(shape)(x)` must satisfy the invariant `shape == x.shape`"
                f"Received `shape={self.shape} and `x.shape={x.shape}`. You might need "
                "to replace `rms_norm(x)` with `jax.vmap(rms_norm)(x)`.\n"
            )

        orig_dtype = x.dtype

        with jax.numpy_dtype_promotion("standard"):
            dtype = jnp.result_type(x.dtype, jnp.float32)

        x = x.astype(dtype)
        inv_rms = jax.lax.rsqrt(jnp.mean(x**2) + self.eps)
        out = inv_rms * x

        if self.use_weight:
            out = self.weight.astype(dtype) * out  # pyright: ignore
        if self.use_bias:
            out = out + self.bias.astype(dtype)  # pyright: ignore
        if state is sentinel:
            return out.astype(orig_dtype)
        else:
            return out.astype(orig_dtype), state
