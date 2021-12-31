import collections
from itertools import repeat
from typing import Optional, Sequence, Union

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from jax.lax import conv_general_dilated

from ..custom_types import Array
from ..module import Module, static_field


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            if len(x) == n:
                return tuple(x)
            else:
                raise ValueError(
                    f"Length of {x} (length = {len(x)}) is not equal to {n}"
                )
        return tuple(repeat(x, n))

    return parse


class Conv(Module):
    num_spatial_dims: int = static_field()
    weight: Array
    bias: Optional[Array]
    in_channels: int = static_field()
    out_channels: int = static_field()
    kernel_size: Union[int, Sequence[int]] = static_field()
    stride: Union[int, Sequence[int]] = static_field()
    padding: Union[int, Sequence[int]] = static_field()
    dilation: Union[int, Sequence[int]] = static_field()
    use_bias: bool = static_field()

    def __init__(
        self,
        num_spatial_dims,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        use_bias=True,
        *,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_spatial_dims = num_spatial_dims
        parse = _ntuple(self.num_spatial_dims)
        wkey, bkey = jrandom.split(key, 2)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = parse(kernel_size)
        self.use_bias = use_bias
        lim = 1 / np.sqrt(self.in_channels * np.prod(self.kernel_size))

        self.weight = jrandom.uniform(
            wkey,
            (
                self.out_channels,
                self.in_channels,
            )
            + self.kernel_size,
            minval=-lim,
            maxval=lim,
        )
        if self.use_bias:
            self.bias = jrandom.uniform(
                bkey,
                (self.out_channels,) + (1,) * self.num_spatial_dims,
                minval=-lim,
                maxval=lim,
            )
        else:
            self.bias = None

        self.stride = parse(stride)
        if isinstance(padding, int):
            self.padding = tuple(
                (padding, padding) for _ in range(self.num_spatial_dims)
            )
        elif isinstance(padding, Sequence) and len(padding) == self.num_spatial_dims:
            self.padding = tuple((p, p) for p in padding)
        else:
            raise ValueError(
                "`padding` must either be an int or tuple of length "
                f"{self.num_spatial_dims}."
            )
        self.dilation = parse(dilation)

    def __call__(self, x, *, key=None):
        unbatched_rank = self.num_spatial_dims + 1
        if x.ndim != unbatched_rank:
            raise ValueError(
                f"Input to `Conv` needs to have rank {unbatched_rank},",
                f" but input has shape {x.shape}.",
            )
        x = jnp.expand_dims(x, axis=0)
        x = conv_general_dilated(
            lhs=x,
            rhs=self.weight,
            window_strides=self.stride,
            padding=self.padding,
            rhs_dilation=self.dilation,
        )
        if self.use_bias:
            x += jnp.broadcast_to(self.bias, x.shape)
        x = jnp.squeeze(x, axis=0)
        return x


class Conv1d(Conv):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
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
            use_bias=use_bias,
            key=key,
            **kwargs,
        )


class Conv2d(Conv):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
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
            use_bias=use_bias,
            key=key,
            **kwargs,
        )


class Conv3d(Conv):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        dilation=(1, 1, 1),
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
            use_bias=use_bias,
            key=key,
            **kwargs,
        )
