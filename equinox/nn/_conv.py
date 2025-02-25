import itertools as it
import math
from collections.abc import Callable, Sequence
from typing import Optional, TypeVar, Union

import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from .._misc import default_floating_dtype
from .._module import field, Module
from ._misc import all_sequences, default_init, named_scope


_T = TypeVar("_T")


def _ntuple(n: int) -> Callable[[Union[_T, Sequence[_T]]], tuple[_T, ...]]:
    def parse(x: Union[_T, Sequence[_T]]) -> tuple[_T, ...]:
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


def _padding_init(
    padding: Union[str, int, Sequence[int], Sequence[tuple[int, int]]],
    num_spatial_dims: int,
) -> Union[str, tuple[tuple[int, int], ...]]:
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ("SAME", "SAME_LOWER", "VALID"):
            raise ValueError(
                "`padding` string must be `'SAME'`, `'SAME_LOWER'`, or `'VALID'`."
            )
    elif isinstance(padding, int):
        padding = tuple((padding, padding) for _ in range(num_spatial_dims))
    elif isinstance(padding, Sequence) and len(padding) == num_spatial_dims:
        if all_sequences(padding):
            padding = tuple(padding)  # pyright: ignore
        else:
            padding = tuple((p, p) for p in padding)
    else:
        raise ValueError(
            "`padding` must either be a string, an int, or tuple of length "
            f"{num_spatial_dims} containing ints or tuples of length 2."
        )
    return padding  # pyright: ignore


def _padding_mode_init(padding_mode: str) -> str:
    padding_mode = padding_mode.upper()
    if padding_mode not in ("ZEROS", "REFLECT", "REPLICATE", "CIRCULAR"):
        raise ValueError(
            "`padding_mode` must be `'ZEROS'`, `'REFLECT'`, `'REPLICATE'`, or "
            "`'CIRCULAR'`."
        )
    return padding_mode


class Conv(Module, strict=True):
    """General N-dimensional convolution."""

    num_spatial_dims: int = field(static=True)
    weight: Array
    bias: Optional[Array]
    in_channels: int = field(static=True)
    out_channels: int = field(static=True)
    kernel_size: tuple[int, ...] = field(static=True)
    stride: tuple[int, ...] = field(static=True)
    padding: Union[str, tuple[tuple[int, int], ...]] = field(static=True)
    dilation: tuple[int, ...] = field(static=True)
    groups: int = field(static=True)
    use_bias: bool = field(static=True)
    padding_mode: str = field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int], Sequence[tuple[int, int]]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = "ZEROS",
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example traditional
            convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `kernel_size`: The size of the convolutional kernel.
        - `stride`: The stride of the convolution.
        - `padding`: The padding of the convolution.
        - `dilation`: The dilation of the convolution.
        - `groups`: The number of input channel groups. At `groups=1`,
            all input channels contribute to all output channels. Values
            higher than `1` are equivalent to running `groups` independent
            `Conv` operations side-by-side, each having access only to
            `in_channels` // `groups` input channels, and
            concatenating the results along the output channel dimension.
            `in_channels` must be divisible by `groups`.
        - `use_bias`: Whether to add on a bias after the convolution.
        - `padding_mode`: One of the following strings specifying the padding values.
            - `'ZEROS'` (default): pads with zeros, `1234 -> 00123400`.
            - `'REFLECT'`: pads with the reflection on boundary, `1234 -> 32123432`.
            - `'REPLICATE'`: pads with the replication of edge values,
                `1234 -> 11123444`.
            - `'CIRCULAR'`: pads with circular values, `1234 -> 34123412`.
        - `dtype`: The dtype to use for the weight and the bias in this layer.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        !!! info

            All of `kernel_size`, `stride`, `padding`, `dilation` can be either an
            integer or a sequence of integers.

            If they are an integer then the same kernel size / stride / padding /
            dilation will be used along every spatial dimension.

            If they are a sequence then the sequence should be of length equal to
            `num_spatial_dims`, and specify the value of each property down each spatial
            dimension in turn.

            In addition, `padding` can be:

            - a sequence of 2-element tuples, each representing the padding to apply
                before and after each spatial dimension.
            - the string `'VALID'`, which is the same as zero padding.
            - one of the strings `'SAME'` or `'SAME_LOWER'`. This will apply padding to
                produce an output with the same size spatial dimensions as the input.
                The padding is split between the two sides equally or almost equally. In
                case the padding is an odd number, then the extra padding is added at
                the end for `'SAME'` and at the beginning for `'SAME_LOWER'`.
        """

        parse = _ntuple(num_spatial_dims)
        kernel_size = parse(kernel_size)
        stride = parse(stride)
        dilation = parse(dilation)

        if in_channels % groups != 0:
            raise ValueError(
                f"`in_channels` (={in_channels}) must be divisible "
                f"by `groups` (={groups})."
            )

        dtype = default_floating_dtype() if dtype is None else dtype
        wkey, bkey = jrandom.split(key, 2)
        grouped_in_channels = in_channels // groups
        lim = 1 / math.sqrt(grouped_in_channels * math.prod(kernel_size))
        wshape = (out_channels, grouped_in_channels) + kernel_size
        self.weight = default_init(wkey, wshape, dtype, lim)
        bshape = (out_channels,) + (1,) * num_spatial_dims
        self.bias = default_init(bkey, bshape, dtype, lim) if use_bias else None

        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = _padding_init(padding, num_spatial_dims)
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias
        self.padding_mode = _padding_mode_init(padding_mode)

    def _nonzero_pad(self, x: Array) -> Array:
        if isinstance(self.padding, str):
            rhs_shape = tuple(
                d * (k - 1) + 1 for k, d in zip(self.kernel_size, self.dilation)
            )
            padding = lax.padtype_to_pads(
                x.shape[1:], rhs_shape, self.stride, self.padding
            )
        else:
            padding = list(self.padding)

        if self.padding_mode == "REFLECT":
            mode = "reflect"
        elif self.padding_mode == "REPLICATE":
            mode = "edge"
        elif self.padding_mode == "CIRCULAR":
            mode = "wrap"
        else:
            raise ValueError("Invalid padding mode")

        x = jnp.pad(x, [(0, 0)] + padding, mode)
        return x

    @named_scope("eqx.nn.Conv")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape
            `(in_channels, dim_1, ..., dim_N)`, where `N = num_spatial_dims`.
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

        if self.padding_mode != "ZEROS":
            x = self._nonzero_pad(x)
            padding = "VALID"
        else:
            padding = self.padding

        x = jnp.expand_dims(x, axis=0)
        x = lax.conv_general_dilated(
            lhs=x,
            rhs=self.weight,
            window_strides=self.stride,
            padding=padding,
            rhs_dilation=self.dilation,
            feature_group_count=self.groups,
        )
        x = jnp.squeeze(x, axis=0)

        if self.use_bias:
            x = x + self.bias
        return x


class Conv1d(Conv):
    """As [`equinox.nn.Conv`][] with `num_spatial_dims=1`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int], Sequence[tuple[int, int]]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = "ZEROS",
        dtype=None,
        *,
        key: PRNGKeyArray,
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
            padding_mode=padding_mode,
            dtype=dtype,
            key=key,
        )


class Conv2d(Conv):
    """As [`equinox.nn.Conv`][] with `num_spatial_dims=2`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = (1, 1),
        padding: Union[str, int, Sequence[int], Sequence[tuple[int, int]]] = (0, 0),
        dilation: Union[int, Sequence[int]] = (1, 1),
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = "ZEROS",
        dtype=None,
        *,
        key: PRNGKeyArray,
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
            padding_mode=padding_mode,
            dtype=dtype,
            key=key,
        )


class Conv3d(Conv):
    """As [`equinox.nn.Conv`][] with `num_spatial_dims=3`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = (1, 1, 1),
        padding: Union[str, int, Sequence[int], Sequence[tuple[int, int]]] = (0, 0, 0),
        dilation: Union[int, Sequence[int]] = (1, 1, 1),
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = "ZEROS",
        dtype=None,
        *,
        key: PRNGKeyArray,
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
            padding_mode=padding_mode,
            dtype=dtype,
            key=key,
        )


class ConvTranspose(Module, strict=True):
    """General N-dimensional transposed convolution."""

    num_spatial_dims: int = field(static=True)
    weight: Array
    bias: Optional[Array]
    in_channels: int = field(static=True)
    out_channels: int = field(static=True)
    kernel_size: tuple[int, ...] = field(static=True)
    stride: tuple[int, ...] = field(static=True)
    padding: Union[str, tuple[tuple[int, int], ...]] = field(static=True)
    output_padding: tuple[int, ...] = field(static=True)
    dilation: tuple[int, ...] = field(static=True)
    groups: int = field(static=True)
    use_bias: bool = field(static=True)
    padding_mode: str = field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int], Sequence[tuple[int, int]]] = 0,
        output_padding: Union[int, Sequence[int]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = "ZEROS",
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example traditional
            convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `kernel_size`: The size of the transposed convolutional kernel.
        - `stride`: The stride used on the equivalent [`equinox.nn.Conv`][].
        - `padding`: The padding used on the equivalent [`equinox.nn.Conv`][].
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
        - `padding_mode`: One of the following strings specifying the padding values
            used on the equivalent [`equinox.nn.Conv`][].
            - `'ZEROS'` (default): pads with zeros, no extra connectivity.
            - `'CIRCULAR'`: pads with circular values, extra connectivity (see the Tip
                below).
        - `dtype`: The dtype to use for the weight and the bias in this layer.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        !!! info

            All of `kernel_size`, `stride`, `padding`, `dilation` can be either an
            integer or a sequence of integers.

            If they are an integer then the same kernel size / stride / padding /
            dilation will be used along every spatial dimension.

            If they are a sequence then the sequence should be of length equal to
            `num_spatial_dims`, and specify the value of each property down each spatial
            dimension in turn.

            In addition, `padding` can be:

            - a sequence of 2-element tuples, each representing the padding to apply
                before and after each spatial dimension.
            - the string `'VALID'`, which is the same as zero padding.
            - one of the strings `'SAME'` or `'SAME_LOWER'`. This will apply padding to
                produce an output with the same size spatial dimensions as the input.
                The padding is split between the two sides equally or almost equally. In
                case the padding is an odd number, then the extra padding is added at
                the end for `'SAME'` and at the beginning for `'SAME_LOWER'`.

        !!! tip

            Transposed convolutions are often used to go in the "opposite direction" to
            a normal convolution. That is, from something with the shape of the output
            of a convolution to something with the shape of the input to a convolution.
            Moreover, to do so with the same "connectivity", i.e. which inputs can
            affect which outputs.

            Relative to an [`equinox.nn.Conv`][] layer, this can be accomplished by
            switching the values of `in_channels` and `out_channels`, whilst keeping
            `kernel_size`, `stride`, `padding`, `dilation`, and `groups` the same.

            When `stride > 1` then [`equinox.nn.Conv`][] maps multiple input shapes to
            the same output shape. `output_padding` is provided to resolve this
            ambiguity.

            - For `'SAME'` or `'SAME_LOWER'` padding, it reduces the calculated input
                shape.
            - For other cases, it adds a little extra padding to the bottom or right
                edges of the input.

            The extra connectivity created in 'CIRCULAR' padding is correctly taken into
            account. For instance, consider the equivalent
            [`equinox.nn.Conv`][] with kernel size 3. Then:

            - `Input 1234 --(zero padding)--> 012340 --(conv)--> Output abcd`
            - `Input 1234 --(circular padding)--> 412341 --(conv)--> Output abcd`

            so that `a` is connected with `1, 2` for zero padding, while connected with
            `1, 2, 4` for circular padding.

            See [these animations](https://github.com/vdumoulin/conv_arithmetic/blob/af6f818b0bb396c26da79899554682a8a499101d/README.md#transposed-convolution-animations)
            and [this report](https://arxiv.org/abs/1603.07285) for a nice reference.

        !!! faq "FAQ"

            If you need to exactly transpose a convolutional layer, i.e. not just create an
            operation with similar inductive biases but compute the actual linear transpose
            of a specific CNN you can reshape the weights of the forward convolution
            via the following:

            ```python
            cnn = eqx.Conv(...)
            cnn_t = eqx.ConvTranspose(...)
            cnn_t = eqx.tree_at(lambda x: x.weight, cnn_t, jnp.flip(cnn.weight,
                                axis=tuple(range(2, cnn.weight.ndim))).swapaxes(0, 1))
            ```

        !!! warning

            `padding_mode='CIRCULAR'` is only implemented for `output_padding=0` and
            `padding='SAME'` or `'SAME_LOWER'`.
        """  # noqa: E501
        dtype = default_floating_dtype() if dtype is None else dtype
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
        lim = 1 / math.sqrt(grouped_in_channels * math.prod(kernel_size))
        wshape = (out_channels, grouped_in_channels) + kernel_size
        self.weight = default_init(wkey, wshape, dtype, lim)
        bshape = (out_channels,) + (1,) * num_spatial_dims
        self.bias = default_init(bkey, bshape, dtype, lim) if use_bias else None

        padding = _padding_init(padding, num_spatial_dims)
        padding_mode = _padding_mode_init(padding_mode)
        if padding_mode in ("REFLECT", "REPLICATE"):
            raise NotImplementedError(
                "'REFLECT' or 'REPLICATE' padding mode is not implemented"
            )
        elif padding_mode == "CIRCULAR":
            if any(o != 0 for o in output_padding):
                raise NotImplementedError(
                    "`padding_mode == 'CIRCULAR'` with non-zero `output_padding` is not"
                    "implemented."
                )
            if padding not in ("SAME", "SAME_LOWER"):
                raise NotImplementedError(
                    "`padding_mode == 'CIRCULAR'` is only implemented for 'SAME' or"
                    "'SAME_LOWER' padding."
                )

        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias
        self.padding_mode = padding_mode

    def _padding_transpose(self) -> tuple[tuple[int, int], ...]:
        # Notations follow https://arxiv.org/abs/1603.07285
        k = np.asarray(self.kernel_size)
        s = np.asarray(self.stride)
        a = np.asarray(self.output_padding)
        d = np.asarray(self.dilation)

        if isinstance(self.padding, str):
            if self.padding == "VALID":
                p0 = np.zeros(self.num_spatial_dims, np.int64)
                p1 = np.zeros(self.num_spatial_dims, np.int64)
            else:
                p_sum = d * (k - 1) - s + a + 1
                a = np.where(p_sum < 0, -p_sum, 0)
                p_sum = np.where(p_sum < 0, 0, p_sum)
                lower = 1 if self.padding == "SAME_LOWER" else 0
                p0 = (p_sum + lower) // 2
                p1 = p_sum - p0
        else:
            p0 = np.asarray(self.padding)[:, 0]
            p1 = np.asarray(self.padding)[:, 1]

        # Given by Relationship 14 of https://arxiv.org/abs/1603.07285
        p0t = d * (k - 1) - p0
        p1t = d * (k - 1) - p1 + a
        padding_t = tuple((x.item(), y.item()) for x, y in zip(p0t, p1t))
        return padding_t

    def _circular_pad(
        self, x: Array, padding_t: tuple[tuple[int, int], ...]
    ) -> tuple[Array, tuple[tuple[int, int], ...]]:
        stride = np.expand_dims(self.stride, axis=1)
        pad_width = np.insert(padding_t // stride, 0, 0, axis=0)
        x = jnp.pad(x, pad_width, mode="wrap")
        padding_t = tuple((p[0].item(), p[1].item()) for p in padding_t % stride)
        return x, padding_t

    @named_scope("eqx.nn.ConvTranspose")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape
            `(in_channels, dim_1, ..., dim_N)`, where `N = num_spatial_dims`.
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

        padding_t = self._padding_transpose()
        if self.padding_mode == "CIRCULAR":
            x, padding_t = self._circular_pad(x, padding_t)

        x = jnp.expand_dims(x, axis=0)
        x = lax.conv_general_dilated(
            lhs=x,
            rhs=self.weight,
            window_strides=(1,) * self.num_spatial_dims,
            padding=padding_t,
            lhs_dilation=self.stride,
            rhs_dilation=self.dilation,
            feature_group_count=self.groups,
        )
        x = jnp.squeeze(x, axis=0)

        if self.use_bias:
            x = x + self.bias
        return x


class ConvTranspose1d(ConvTranspose):
    """As [`equinox.nn.ConvTranspose`][] with `num_spatial_dims=1`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        output_padding: Union[int, Sequence[int]] = 0,
        padding: Union[str, int, Sequence[int], Sequence[tuple[int, int]]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = "ZEROS",
        dtype=None,
        *,
        key: PRNGKeyArray,
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
            padding_mode=padding_mode,
            dtype=dtype,
            key=key,
        )


class ConvTranspose2d(ConvTranspose):
    """As [`equinox.nn.ConvTranspose`][] with `num_spatial_dims=2`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = (1, 1),
        output_padding: Union[int, Sequence[int]] = (0, 0),
        padding: Union[str, int, Sequence[int], Sequence[tuple[int, int]]] = (0, 0),
        dilation: Union[int, Sequence[int]] = (1, 1),
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = "ZEROS",
        dtype=None,
        *,
        key: PRNGKeyArray,
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
            padding_mode=padding_mode,
            dtype=dtype,
            key=key,
        )


class ConvTranspose3d(ConvTranspose):
    """As [`equinox.nn.ConvTranspose`][] with `num_spatial_dims=3`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = (1, 1, 1),
        output_padding: Union[int, Sequence[int]] = (0, 0, 0),
        padding: Union[str, int, Sequence[int], Sequence[tuple[int, int]]] = (0, 0, 0),
        dilation: Union[int, Sequence[int]] = (1, 1, 1),
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = "ZEROS",
        dtype=None,
        *,
        key: PRNGKeyArray,
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
            padding_mode=padding_mode,
            key=key,
        )
