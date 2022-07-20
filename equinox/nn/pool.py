from typing import Callable, Optional, Sequence, Tuple, Union

import jax.lax as lax
import jax.numpy as jnp
import jax.random
import numpy as np

from ..custom_types import Array
from ..module import Module, static_field


class Pool(Module):
    """General N-dimensional downsampling over a sliding window."""

    init: Union[int, float, Array]
    operation: Callable[[Array, Array], Array]
    num_spatial_dims: int = static_field()
    kernel_size: Union[int, Sequence[int]] = static_field()
    stride: Union[int, Sequence[int]] = static_field()
    padding: Union[int, Sequence[int], Sequence[Tuple[int, int]]] = static_field()

    def __init__(
        self,
        init: Union[int, float, Array],
        operation: Callable[[Array, Array], Array],
        num_spatial_dims: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int], Sequence[Tuple[int, int]]] = 0,
        **kwargs,
    ):
        """**Arguments:**

        - `init`: The initial value for the reduction.
        - `operation`: The operation applied to the inputs of each window.
        - `num_spatial_dims`: The number of spatial dimensions.
        - `kernel_size`: The size of the convolutional kernel.
        - `stride`: The stride of the convolution.
        - `padding`: The amount of padding to apply before and after each
            spatial dimension.

        !!! info

            In order for `Pool` to be differentiable, `operation(init, x) == x` needs to
            be true for all finite `x`. For further details see
            [https://www.tensorflow.org/xla/operation_semantics#reducewindow](https://www.tensorflow.org/xla/operation_semantics#reducewindow)
            and [https://github.com/google/jax/issues/7718](https://github.com/google/jax/issues/7718).
        """
        super().__init__(**kwargs)

        self.operation = operation
        self.init = init
        self.num_spatial_dims = num_spatial_dims

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size,) * num_spatial_dims
        elif isinstance(kernel_size, Sequence):
            self.kernel_size = kernel_size
        else:
            raise ValueError(
                "`kernel_size` must either be an int or tuple of length "
                f"{num_spatial_dims} containing ints."
            )

        if isinstance(stride, int):
            self.stride = (stride,) * num_spatial_dims
        elif isinstance(stride, Sequence):
            self.stride = stride
        else:
            raise ValueError(
                "`stride` must either be an int or tuple of length "
                f"{num_spatial_dims} containing ints."
            )

        if isinstance(padding, int):
            self.padding = tuple((padding, padding) for _ in range(num_spatial_dims))
        elif isinstance(padding, Sequence) and len(padding) == num_spatial_dims:
            if all(isinstance(element, Sequence) for element in padding):
                self.padding = padding
            else:
                self.padding = tuple((p, p) for p in padding)
        else:
            raise ValueError(
                "`padding` must either be an int or tuple of length "
                f"{num_spatial_dims} containing ints or tuples of length 2."
            )

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape `(channels, dim_1, ..., dim_N)`, where
            `N = num_spatial_dims`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of shape `(channels, new_dim_1, ..., new_dim_N)`.
        """
        assert len(x.shape) == self.num_spatial_dims + 1, (
            f"Input should have {self.num_spatial_dims} spatial dimensions, "
            f"but input has shape {x.shape}"
        )

        x = jnp.moveaxis(x, 0, -1)
        x = jnp.expand_dims(x, axis=0)
        x = lax.reduce_window(
            x,
            self.init,
            self.operation,
            (1,) + self.kernel_size + (1,),
            (1,) + self.stride + (1,),
            ((0, 0),) + self.padding + ((0, 0),),
        )

        x = jnp.squeeze(x, axis=0)
        x = jnp.moveaxis(x, -1, 0)
        return x


class AvgPool1d(Pool):
    """One-dimensional downsample using an average over a sliding window."""

    def __init__(
        self,
        kernel_size,
        stride,
        padding=0,
        **kwargs,
    ):
        """**Arguments:**

        - `kernel_size`: The size of the convolutional kernel.
        - `stride`: The stride of the convolution.
        - `padding`: The amount of padding to apply before and after each
            spatial dimension.
        """

        super().__init__(
            init=0,
            operation=lax.add,
            num_spatial_dims=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs,
        )

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape `(channels, dim)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of shape `(channels, new_dim)`.
        """

        return super().__call__(x) / np.prod(self.kernel_size)


class MaxPool1d(Pool):
    """One-dimensional downsample using the maximum over a sliding window."""

    def __init__(
        self,
        kernel_size,
        stride,
        padding=0,
        **kwargs,
    ):
        """**Arguments:**

        - `kernel_size`: The size of the convolutional kernel.
        - `stride`: The stride of the convolution.
        - `padding`: The amount of padding to apply before and after each
            spatial dimension.
        """

        super().__init__(
            init=-jnp.inf,
            operation=lax.max,
            num_spatial_dims=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs,
        )

    # Redefined to get them in the right order in docs
    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape `(channels, dim)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of shape `(channels, new_dim)`.
        """

        return super().__call__(x)


class AvgPool2d(Pool):
    """Two-dimensional downsample using an average over a sliding window."""

    def __init__(
        self,
        kernel_size,
        stride,
        padding=0,
        **kwargs,
    ):
        """**Arguments:**

        - `kernel_size`: The size of the convolutional kernel.
        - `stride`: The stride of the convolution.
        - `padding`: The amount of padding to apply before and after each
            spatial dimension.
        """

        super().__init__(
            init=0,
            operation=lax.add,
            num_spatial_dims=2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs,
        )

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape `(channels, dim_1, dim_2)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of shape `(channels, new_dim_1, new_dim_2)`.
        """

        return super().__call__(x) / np.prod(self.kernel_size)


class MaxPool2d(Pool):
    """Two-dimensional downsample using the maximum over a sliding window."""

    def __init__(
        self,
        kernel_size,
        stride,
        padding=0,
        **kwargs,
    ):
        """**Arguments:**

        - `kernel_size`: The size of the convolutional kernel.
        - `stride`: The stride of the convolution.
        - `padding`: The amount of padding to apply before and after each
            spatial dimension.
        """

        super().__init__(
            init=-jnp.inf,
            operation=lax.max,
            num_spatial_dims=2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs,
        )

    # Redefined to get them in the right order in docs
    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape `(channels, dim_1, dim_2)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of shape `(channels, new_dim_1, new_dim_2)`.
        """

        return super().__call__(x)


class AvgPool3d(Pool):
    """Three-dimensional downsample using an average over a sliding window."""

    def __init__(
        self,
        kernel_size,
        stride,
        padding=0,
        **kwargs,
    ):
        """**Arguments:**

        - `kernel_size`: The size of the convolutional kernel.
        - `stride`: The stride of the convolution.
        - `padding`: The amount of padding to apply before and after each
            spatial dimension.
        """

        super().__init__(
            init=0,
            operation=lax.add,
            num_spatial_dims=3,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs,
        )

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape
            `(channels, dim_1, dim_2, dim_3)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of shape `(channels, new_dim_1, new_dim_2, new_dim_3)`.
        """

        return super().__call__(x) / np.prod(self.kernel_size)


class MaxPool3d(Pool):
    """Three-dimensional downsample using the maximum over a sliding window."""

    def __init__(
        self,
        kernel_size,
        stride,
        padding=0,
        **kwargs,
    ):
        """**Arguments:**

        - `kernel_size`: The size of the convolutional kernel.
        - `stride`: The stride of the convolution.
        - `padding`: The amount of padding to apply before and after each
            spatial dimension.
        """

        super().__init__(
            init=-jnp.inf,
            operation=lax.max,
            num_spatial_dims=3,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs,
        )

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape
            `(channels, dim_1, dim_2, dim_3)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of shape `(channels, new_dim_1, new_dim_2, new_dim_3)`.
        """

        return super().__call__(x)


# Backward compatability: these were originally misnamed.
AvgPool1D = AvgPool1d
AvgPool2D = AvgPool2d
AvgPool3D = AvgPool3d
MaxPool1D = MaxPool1d
MaxPool2D = MaxPool2d
MaxPool3D = MaxPool3d


def _adaptive_pool1d(x: Array, target_size: int, operation: Callable) -> Array:
    """**Arguments:**

    - `x`: The input. Should be a JAX array of shape `(dim,)`.
    - `target_size`: The shape of the output after the pooling operation `(target_size,)`.
    - `operation`: The pooling operation to be performed on the input array.

    **Returns:**

    A JAX array of shape `(1, target_shape)`.
    """
    dims = jnp.size(x)
    num_head_arrays = dims % target_size
    if num_head_arrays != 0:
        head_end_index = num_head_arrays * (dims // target_size + 1)
        head_op = jax.vmap(operation)(x[:head_end_index].reshape(num_head_arrays, -1))
        tail_op = jax.vmap(operation)(
            x[head_end_index:].reshape(-1, dims // target_size)
        )
        outputs = jnp.concatenate([head_op, tail_op])
    else:
        outputs = jax.vmap(operation)(
            jax.vmap(operation)(x.reshape(-1, dims // target_size))
        )
    return outputs


class AdaptivePool(Module):
    """General N dimensional Adaptive downsampling for the target shape."""

    target_shape: Sequence[int] = static_field()
    operation: Callable

    def __init__(
        self,
        target_shape: Union[int, Sequence[int]],
        num_spatial_dims: int,
        operation: Callable,
        **kwargs,
    ):
        """**Arguments:**

        - `target_size`: The target output size.
        - `num_spatial_dims`: The number of spatial dimensions.
        - `operation`: The operation applied for downsample.
        """
        super().__init__(**kwargs)
        self.operation = operation
        if isinstance(target_shape, int):
            self.target_shape = (target_shape,) * num_spatial_dims
        elif (
            isinstance(target_shape, Sequence) and len(target_shape) == num_spatial_dims
        ):
            self.target_shape = target_shape
        else:
            raise ValueError(
                "`target_size` must either be an int or tuple of length "
                f"{num_spatial_dims} containing ints."
            )

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape
            `(channels, dim_1, dim_2, ... )`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of shape `(channels, target_shape)`.
        """
        if x.ndim - 1 != len(self.target_shape):
            raise ValueError(
                f"Expected input with {len(self.target_shape)} dimensions, "
                f"received {x.ndim-1} instead."
            )
        for i in range(1, x.ndim):
            op = jax.vmap(
                _adaptive_pool1d, (0, None, None), 0
            )  # batching over channels by default
            for j in range(1, x.ndim):
                if i == j:
                    continue
                op = jax.vmap(op, in_axes=(j, None, None), out_axes=j)
            x = op(x, self.target_shape[i - 1], self.operation)
        return x


class AdaptiveAvgPool1d(AdaptivePool):
    """Adaptive one-dimensional downsampling using average for the target shape."""

    def __init__(self, target_shape: Union[int, Sequence[int]], **kwargs):
        """**Arguments:**

        - `target_shape`: The target output shape.
        """
        super().__init__(target_shape, num_spatial_dims=1, operation=jnp.mean, **kwargs)


class AdaptiveAvgPool2d(AdaptivePool):
    """Adaptive two-dimensional downsampling using average for the target shape."""

    def __init__(self, target_shape: Union[int, Sequence[int]], **kwargs):
        """**Arguments:**

        - `target_shape`: The target output shape.
        """
        super().__init__(target_shape, num_spatial_dims=2, operation=jnp.mean, **kwargs)


class AdaptiveAvgPool3d(AdaptivePool):
    """Adaptive three-dimensional downsampling using average for the target shape."""

    def __init__(self, target_shape: Union[int, Sequence[int]], **kwargs):
        """**Arguments:**

        - `target_shape`: The target output shape.
        """
        super().__init__(target_shape, num_spatial_dims=3, operation=jnp.mean, **kwargs)


class AdaptiveMaxPool1d(AdaptivePool):
    """Adaptive one-dimensional downsampling using maximum for the target shape."""

    def __init__(self, target_shape: Union[int, Sequence[int]], **kwargs):
        """**Arguments:**

        - `target_shape`: The target output shape.
        """
        super().__init__(target_shape, num_spatial_dims=1, operation=jnp.max, **kwargs)


class AdaptiveMaxPool2d(AdaptivePool):
    """Adaptive two-dimensional downsampling using maximum for the target shape."""

    def __init__(self, target_shape: Union[int, Sequence[int]], **kwargs):
        """**Arguments:**

        - `target_shape`: The target output shape.
        """
        super().__init__(target_shape, num_spatial_dims=2, operation=jnp.max, **kwargs)


class AdaptiveMaxPool3d(AdaptivePool):
    """Adaptive three-dimensional downsampling using maximum for the target shape."""

    def __init__(self, target_shape: Union[int, Sequence[int]], **kwargs):
        """**Arguments:**

        - `target_shape`: The target output shape.
        """
        super().__init__(target_shape, num_spatial_dims=3, operation=jnp.max, **kwargs)
