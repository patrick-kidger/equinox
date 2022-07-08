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


class AvgPool1D(Pool):
    """One-dimensional downsample using an average over a sliding window."""

    def __init__(
        self,
        kernel_size,
        stride=None,
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


class MaxPool1D(Pool):
    """One-dimensional downsample using the maximum over a sliding window."""

    def __init__(
        self,
        kernel_size,
        stride=None,
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


class AvgPool2D(Pool):
    """Two-dimensional downsample using an average over a sliding window."""

    def __init__(
        self,
        kernel_size,
        stride=None,
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


class MaxPool2D(Pool):
    """Two-dimensional downsample using the maximum over a sliding window."""

    def __init__(
        self,
        kernel_size,
        stride=None,
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


class AvgPool3D(Pool):
    """Three-dimensional downsample using an average over a sliding window."""

    def __init__(
        self,
        kernel_size,
        stride=None,
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


class MaxPool3D(Pool):
    """Three-dimensional downsample using the maximum over a sliding window."""

    def __init__(
        self,
        kernel_size,
        stride=None,
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
