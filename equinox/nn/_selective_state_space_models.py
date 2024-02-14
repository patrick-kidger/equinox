import math
from typing import Literal, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from .._module import field, Module
from ._conv import Conv1d
from ._linear import Linear


class SelectiveStateSpaceModel(Module, strict=True):
    """
    State Space Model with Selective Scan. This is the implementation of the
    Mamba Block from the paper
    "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" [1].

    [1] Albert Gu and Tri Dao, Mamba: Linear-Time Sequence Modeling
    with Selective State Spaces, 2023
    """

    n_input_dims: int = field(static=True)
    state_space_dims: int = field(static=True)

    d_inner: int = field(static=True)
    d_conv: int = field(static=True)

    expand: int = field(static=True)
    dt_rank: int = field(static=True)
    pad_vocab_size_multiple: int = field(static=True)

    in_proj: Linear
    conv1d: Conv1d

    x_proj: Linear
    dt_proj: Linear

    A_log: Array
    D: Array

    out_proj: Linear

    def __init__(
        self,
        n_input_dims: int,
        state_space_dims: int,
        expand: int,
        d_conv: int,
        dt_rank: Union[int, Literal["auto"]],
        pad_vocab_size_multiple: int = 8,
        use_bias_in_proj: bool = True,
        use_bias_conv1d: bool = True,
        use_bias_out_proj: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        """
        Args:
            n_input_dims: The dimension of the input.
            state_space_dims: The dimension of the SSM (refers to 'N' in [1]).
            expand: The expansion factor of the inner dimension (refers to 'E' in [1]).
            d_conv: The kernel size of the convolutional layer
            dt_rank: The rank of delta. If "auto", it will be
                set to ceil(n_input_dims / state_space_dims).
            pad_vocab_size_multiple: The multiple of the vocabulary size

        """
        self.n_input_dims = n_input_dims
        self.state_space_dims = state_space_dims

        self.d_conv = d_conv
        self.expand = expand

        self.d_inner = int(self.expand * self.n_input_dims)

        self.pad_vocab_size_multiple = pad_vocab_size_multiple

        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.n_input_dims / self.state_space_dims)

        (
            key,
            linear_key,
            conv1d_key,
            x_proj_key,
            dt_proj_key,
            out_proj_key,
        ) = jax.random.split(key, 6)

        self.in_proj = Linear(
            n_input_dims,
            self.d_inner * 2,
            use_bias=use_bias_in_proj,
            key=linear_key,
        )

        self.conv1d = Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            use_bias=use_bias_conv1d,
            groups=self.d_inner,
            padding=d_conv - 1,
            key=conv1d_key,
        )

        self.x_proj = Linear(
            self.d_inner,
            self.dt_rank + state_space_dims * 2,
            use_bias=False,
            key=x_proj_key,
        )

        self.dt_proj = Linear(
            self.dt_rank, self.d_inner, use_bias=True, key=dt_proj_key
        )

        A = jnp.repeat(jnp.arange(1, self.state_space_dims + 1), self.d_inner).reshape(
            self.d_inner, self.state_space_dims
        )
        self.A_log = jnp.log(A)
        self.D = jnp.ones(self.d_inner)
        self.out_proj = Linear(
            self.d_inner,
            self.n_input_dims,
            use_bias=use_bias_out_proj,
            key=x_proj_key,
        )

    @jax.named_scope("eqx.nn.StateSpaceModel")
    def __call__(self) -> Array:
        raise NotImplementedError
