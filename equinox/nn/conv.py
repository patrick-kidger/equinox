import itertools as it
from typing import Any, Optional, Sequence, Tuple, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from ..custom_types import Array
from ..module import Module, static_field


def _ntuple(n: int) -> callable:
    def parse(x: Any) -> tuple:
        if isinstance(x, Sequence):
            if len(x) == n:
                return tuple(x)
            else:
                raise ValueError(
                    f"Length of {x} (length = {len(x)}) is not equal to {n}"
                )
        else:
            return tuple(it.repeat(x, n))

    return parse


# TODO: remove this once JAX fixes the issue.
# Working around JAX not always properly respecting __jax_array__, see JAX issue #10065
# This comes up in particular when using SpectralNorm.
def _to_jax_array(weight):
    if hasattr(weight, "__jax_array__"):
        return weight.__jax_array__()
    else:
        return weight


class Conv(Module):
    """General N-dimensional convolution."""

    num_spatial_dims: int = static_field()
    weight: Array
    bias: Optional[Array]
    in_channels: int = static_field()
    out_channels: int = static_field()
    kernel_size: Tuple[int, ...] = static_field()
    stride: Tuple[int, ...] = static_field()
    padding: Tuple[Tuple[int, int], ...] = static_field()
    dilation: Tuple[int, ...] = static_field()
    groups: int = static_field()
    use_bias: bool = static_field()

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = True,
        *,
        key: "jax.random.PRNGKey",
        **kwargs,
    ):
        """**Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example traditional
            convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `kernel_size`: The size of the convolutional kernel.
        - `stride`: The stride of the convolution.
        - `padding`: The amount of padding to apply before and after each spatial
            dimension. The same amount of padding is applied both before and after.
        - `dilation`: The dilation of the convolution.
        - `groups`: The number of input channel groups. At `groups=1`,
            all input channels contribute to all output channels. Values
            higher than `1` are equivalent to running `groups` independent
            `Conv` operations side-by-side, each having access only to
            `in_channels` // `groups` input channels, and
            concatenating the results along the output channel dimension.
            `in_channels` must be divisible by `groups`.
        - `use_bias`: Whether to add on a bias after the convolution.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        !!! info

            All of `kernel_size`, `stride`, `padding`, `dilation` can be either an
            integer or a sequence of integers. If they are a sequence then the sequence
            should be of length equal to `num_spatial_dims`, and specify the value of
            each property down each spatial dimension in turn. If they are an integer
            then the same kernel size / stride / padding / dilation will be used along
            every spatial dimension.

        """
        super().__init__(**kwargs)
        wkey, bkey = jrandom.split(key, 2)

        parse = _ntuple(num_spatial_dims)
        kernel_size = parse(kernel_size)
        stride = parse(stride)
        dilation = parse(dilation)

        if in_channels % groups != 0:
            raise ValueError(
                f"`in_channels` (={in_channels}) must be divisible "
                f"by `groups` (={groups})."
            )

        grouped_in_channels = in_channels // groups
        lim = 1 / np.sqrt(grouped_in_channels * np.prod(kernel_size))
        self.weight = jrandom.uniform(
            wkey,
            (out_channels, grouped_in_channels) + kernel_size,
            minval=-lim,
            maxval=lim,
        )
        if use_bias:
            self.bias = jrandom.uniform(
                bkey,
                (out_channels,) + (1,) * num_spatial_dims,
                minval=-lim,
                maxval=lim,
            )
        else:
            self.bias = None

        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(padding, int):
            self.padding = tuple((padding, padding) for _ in range(num_spatial_dims))
        elif isinstance(padding, Sequence) and len(padding) == num_spatial_dims:
            self.padding = tuple((p, p) for p in padding)
        else:
            raise ValueError(
                "`padding` must either be an int or tuple of length "
                f"{num_spatial_dims}."
            )
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape `(in_channels, dim_1, ..., dim_N)`, where
            `N = num_spatial_dims`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of shape `(out_channels, new_dim_1, ..., new_dim_N)`.
        """

        unbatched_rank = self.num_spatial_dims + 1
        if x.ndim != unbatched_rank:
            raise ValueError(
                f"Input to `Conv` needs to have rank {unbatched_rank},",
                f" but input has shape {x.shape}.",
            )
        x = jnp.expand_dims(x, axis=0)
        x = lax.conv_general_dilated(
            lhs=x,
            rhs=_to_jax_array(self.weight),
            window_strides=self.stride,
            padding=self.padding,
            rhs_dilation=self.dilation,
            feature_group_count=self.groups,
        )
        if self.use_bias:
            x = x + self.bias
        x = jnp.squeeze(x, axis=0)
        return x


class Conv1d(Conv):
    """As [`equinox.nn.Conv`][] with `num_spatial_dims=1`."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        use_bias=True,
        *,
        key,
        **kwargs,
    ):
        super().__init__(
            num_spatial_dims=1,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            key=key,
            **kwargs,
        )


class Conv2d(Conv):
    """As [`equinox.nn.Conv`][] with `num_spatial_dims=2`."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        use_bias=True,
        *,
        key,
        **kwargs,
    ):
        super().__init__(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            key=key,
            **kwargs,
        )


class Conv3d(Conv):
    """As [`equinox.nn.Conv`][] with `num_spatial_dims=3`."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        dilation=(1, 1, 1),
        groups=1,
        use_bias=True,
        *,
        key,
        **kwargs,
    ):
        super().__init__(
            num_spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            key=key,
            **kwargs,
        )


class ConvTranspose(Module):
    """General N-dimensional transposed convolution."""

    num_spatial_dims: int = static_field()
    weight: Array
    bias: Optional[Array]
    in_channels: int = static_field()
    out_channels: int = static_field()
    kernel_size: Tuple[int, ...] = static_field()
    stride: Tuple[int, ...] = static_field()
    padding: Tuple[Tuple[int, int], ...] = static_field()
    output_padding: Tuple[int, ...] = static_field()
    dilation: Tuple[int, ...] = static_field()
    groups: int = static_field()
    use_bias: bool = static_field()

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int]] = 0,
        output_padding: Union[int, Sequence[int]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = True,
        *,
        key: "jax.random.PRNGKey",
        **kwargs,
    ):
        """**Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example traditional
            convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `kernel_size`: The size of the transposed convolutional kernel.
        - `stride`: The stride used on the equivalent [`equinox.nn.Conv`][].
        - `padding`: The amount of padding used on the equivalent [`equinox.nn.Conv`][].
        - `output_padding`: Additional padding for the output shape.
        - `dilation`: The spacing between kernel points.
        - `groups`: The number of input channel groups. At `groups=1`,
            all input channels contribute to all output channels. Values
            higher than 1 are equivalent to running `groups` independent
            `ConvTranspose` operations side-by-side, each having access only to
            `in_channels` // `groups` input channels, and
            concatenating the results along the output channel dimension.
            `in_channels` must be divisible by `groups`.
        - `use_bias`: Whether to add on a bias after the transposed convolution.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        !!! info

            All of `kernel_size`, `stride`, `padding`, `output_padding`, `dilation` can
            be either an integer or a sequence of integers. If they are a sequence then
            the sequence should be of length equal to `num_spatial_dims`, and specify
            the value of each property down each spatial dimension in turn.. If they
            are an integer then the same kernel size / stride / padding / dilation will
            be used along every spatial dimension.

        !!! tip

            Transposed convolutions are often used to go in the "opposite direction" to
            a normal convolution. That is, from something with the shape of the output
            of a convolution to something with the shape of the input to a convolution.
            Moreover, to do so with the same "connectivity", i.e. which inputs can
            affect which outputs.

            Relative to an [`equinox.nn.Conv`][] layer, this can be accomplished by
            switching the values of `in_channels` and `out_channels`, whilst keeping
            `kernel_size`, `stride`, `padding`, `dilation`, and `groups` the same.

            When `stride > 1` then [`equinox.nn.Conv`][] maps multiple input shapes to the
            same output shape. `output_padding` is provided to resolve this ambiguity,
            by adding a little extra padding to just the bottom/right edges of the
            input.

            See [these animations](https://github.com/vdumoulin/conv_arithmetic/blob/af6f818b0bb396c26da79899554682a8a499101d/README.md#transposed-convolution-animations)
            and [this report](https://arxiv.org/abs/1603.07285) for a nice reference.
        """  # noqa: E501

        super().__init__(**kwargs)
        wkey, bkey = jrandom.split(key, 2)

        parse = _ntuple(num_spatial_dims)
        kernel_size = parse(kernel_size)
        stride = parse(stride)
        output_padding = parse(output_padding)
        dilation = parse(dilation)

        for s, o in zip(stride, output_padding):
            if output_padding >= stride:
                raise ValueError("Must have `output_padding < stride` (elementwise).")

        grouped_in_channels = in_channels // groups
        lim = 1 / np.sqrt(grouped_in_channels * np.prod(kernel_size))
        self.weight = jrandom.uniform(
            wkey,
            (out_channels, grouped_in_channels) + kernel_size,
            minval=-lim,
            maxval=lim,
        )
        if use_bias:
            self.bias = jrandom.uniform(
                bkey,
                (out_channels,) + (1,) * num_spatial_dims,
                minval=-lim,
                maxval=lim,
            )
        else:
            self.bias = None

        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(padding, int):
            self.padding = tuple((padding, padding) for _ in range(num_spatial_dims))
        elif isinstance(padding, Sequence) and len(padding) == num_spatial_dims:
            self.padding = tuple((p, p) for p in padding)
        else:
            raise ValueError(
                "`padding` must either be an int or tuple of length "
                f"{num_spatial_dims}."
            )
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape `(in_channels, dim_1, ..., dim_N)`, where
            `N = num_spatial_dims`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of shape `(out_channels, new_dim_1, ..., new_dim_N)`.
        """
        unbatched_rank = self.num_spatial_dims + 1
        if x.ndim != unbatched_rank:
            raise ValueError(
                f"Input to `ConvTranspose` needs to have rank {unbatched_rank},",
                f" but input has shape {x.shape}.",
            )
        x = jnp.expand_dims(x, axis=0)
        # Given by Relationship 14 of https://arxiv.org/abs/1603.07285
        padding = tuple(
            (d * (k - 1) - p0, d * (k - 1) - p1 + o)
            for k, (p0, p1), o, d in zip(
                self.kernel_size, self.padding, self.output_padding, self.dilation
            )
        )
        x = lax.conv_general_dilated(
            lhs=x,
            rhs=_to_jax_array(self.weight),
            window_strides=(1,) * self.num_spatial_dims,
            padding=padding,
            lhs_dilation=self.stride,
            rhs_dilation=self.dilation,
            feature_group_count=self.groups,
        )
        if self.use_bias:
            x = x + self.bias
        x = jnp.squeeze(x, axis=0)
        return x


class ConvTranspose1d(ConvTranspose):
    """As [`equinox.nn.ConvTranspose`][] with `num_spatial_dims=1`."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        output_padding=0,
        padding=0,
        dilation=1,
        groups=1,
        use_bias=True,
        *,
        key,
        **kwargs,
    ):
        super().__init__(
            num_spatial_dims=1,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=output_padding,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            key=key,
            **kwargs,
        )


class ConvTranspose2d(ConvTranspose):
    """As [`equinox.nn.ConvTranspose`][] with `num_spatial_dims=2`."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1),
        output_padding=(0, 0),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        use_bias=True,
        *,
        key,
        **kwargs,
    ):
        super().__init__(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=output_padding,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            key=key,
            **kwargs,
        )


class ConvTranspose3d(ConvTranspose):
    """As [`equinox.nn.ConvTranspose`][] with `num_spatial_dims=3`."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1, 1),
        output_padding=(0, 0, 0),
        padding=(0, 0, 0),
        dilation=(1, 1, 1),
        groups=1,
        use_bias=True,
        *,
        key,
        **kwargs,
    ):
        super().__init__(
            num_spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=output_padding,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            key=key,
            **kwargs,
        )
