from typing import Sequence, Tuple, Union

import jax.lax as lax
import numpy as np

from ..custom_types import Array
from ..module import Module, static_field


class Pool(Module):
    operation: str = static_field()
    num_spatial_dims: int = static_field()
    kernel_size: Union[int, Sequence[int]] = static_field()
    stride: Sequence[int] = static_field()
    padding: Union[str, Sequence[Tuple[int, int]]] = static_field()

    def __init__(
        self,
        operation: str,
        num_spatial_dims: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Sequence[int] = None,
        padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.operation = operation
        self.num_spatial_dims = num_spatial_dims
        self.kernel_size = kernel_size
        self.stride = stride
        if stride is None:
            self.stride = self.kernel_size
        self.padding = padding

    def __call__(self, x: Array) -> Array:
        if self.operation == "max":
            output = lax.reduce_window(
                x,
                0.0,
                lax.max,
                (1,) + self.kernel_size,
                (1,) + self.stride,
                self.padding,
            )
            # print(x)
        elif self.operation == "avg":
            output = lax.reduce_window(
                x,
                0.0,
                lax.add,
                (1,) + self.kernel_size,
                (1,) + self.stride,
                self.padding,
            ) / np.prod(self.kernel_size)

        return output


class Pool2d_Avg(Pool):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
    ):
        super().__init__(
            num_spatial_dims=2,
            kernel_size=kernel_size,
            operation="avg",
            padding=padding,
            stride=stride,
        )


class Pool2d_Max(Pool):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
    ):
        super().__init__(
            num_spatial_dims=2,
            kernel_size=kernel_size,
            operation="max",
            padding=padding,
            stride=stride,
        )
