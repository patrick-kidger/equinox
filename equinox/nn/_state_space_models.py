import math
from typing import Literal, Union

import jax
from jaxtyping import Array, PRNGKeyArray

from .._module import field, Module
from ._conv import Conv1d
from ._linear import Linear


class StateSpaceModel(Module, strict=True):
    d_inner: int = field(static=True)
    d_state: int = field(static=True)
    d_conv: int = field(static=True)
    n_embd: int = field(static=True)
    n_dims: int = field(static=True)
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
        n_embd: int,
        expand: int,
        d_state: int,
        d_conv: int,
        dt_rank: Union[int, Literal["auto"]],
        pad_vocab_size_multiple: int = 8,
        n_dims: int = 256,
        use_bias_in_proj: bool = True,
        use_bias_conv1d: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        self.d_state = d_state
        self.d_conv = d_conv
        self.n_embd = n_embd
        self.expand = expand
        self.n_dims = n_dims
        self.d_inner = int(self.expand * self.n_embd)
        self.pad_vocab_size_multiple = pad_vocab_size_multiple

        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.n_embd / self.d_state)

        if self.n_dims % self.pad_vocab_size_multiple != 0:
            self.n_dims += (
                self.pad_vocab_size_multiple
                - self.n_dims % self.pad_vocab_size_multiple
            )

        (
            key,
            linear_key,
            conv1d_key,
            x_proj_key,
            dt_proj_key,
            out_proj_key,
        ) = jax.random.split(key, 6)

        self.in_proj = Linear(
            n_embd,
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
            self.dt_rank + d_state * 2,
            use_bias=False,
            key=x_proj_key,
        )

    @jax.named_scope("eqx.nn.StateSpaceModel")
    def __call__(self) -> Array:
        raise NotImplementedError
