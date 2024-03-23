import math
from typing import Literal, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .._module import field, Module
from ._conv import Conv1d
from ._linear import Linear


def _selective_scan(
    u: Float[Array, "seq_len d_inner"],
    delta: Float[Array, "seq_len d_inner"],
    A: Float[Array, "d_inner state_space_dims"],
    B: Float[Array, "seq_len state_space_dims"],
    C: Float[Array, "seq_len state_space_dims"],
    D: Float[Array, " d_inner"],
):
    seq_len, _ = u.shape
    d_inner, state_space_dims = A.shape

    delta_A = jnp.exp(jnp.einsum("l d,d n -> l d n", delta, A))
    delta_B_u = jnp.einsum("l d,l n,l d -> l d n", delta, B, u)

    x_res = jnp.zeros(shape=(d_inner, state_space_dims))

    def step(x, i):
        x = delta_A[i] * x + delta_B_u[i]

        y = jnp.einsum("d n,n -> d", x, C[i, :])
        return x, y

    _, ys = jax.lax.scan(step, x_res, jnp.arange(seq_len))

    ys = ys + u * D
    return ys


class SelectiveStateSpaceModel(Module, strict=True):
    r"""
    State Space Model with Selective Scan. This is the implementation of the
    Mamba Block from the paper
    "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" [1].


    ??? cite
        [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
        ```bibtex
            @misc{
            gu2023mamba,
            title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
            author={Albert Gu and Tri Dao},
            year={2023},
            eprint={2312.00752},
            archivePrefix={arXiv},
            primaryClass={cs.LG}
        }
        ```
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
        r"""**Arguments:**

        - `n_input_dims`: The dimension of the input.
        - `state_space_dims`: The dimension of the SSM (refers to $N$ in [1]).
        - `expand`: The expansion factor of the inner dimension (refers to $E$ in [1]).
        - `d_conv`: The kernel size of the convolutional layer
        - `dt_rank`: The rank of delta. If "auto", it will be set to
            ceil(n_input_dims / state_space_dims).
        - `pad_vocab_size_multiple`: The multiple of the vocabulary size
        - `use_bias_in_proj`: Whether to use bias in the input projection layer.
        - `use_bias_conv1d`: Whether to use bias in the convolutional layer.
        - `use_bias_out_proj`: Whether to use bias in the output projection layer.
        - `key`: The PRNG key.

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

        A = (
            jnp.repeat(jnp.arange(1, self.state_space_dims + 1), self.d_inner)
            .reshape(self.state_space_dims, self.d_inner)
            .transpose()
        )
        self.A_log = jnp.log(A)
        self.D = jnp.ones(self.d_inner)
        self.out_proj = Linear(
            self.d_inner,
            self.n_input_dims,
            use_bias=use_bias_out_proj,
            key=x_proj_key,
        )

    @jax.named_scope("eqx.nn.SelectiveStateSpaceModel")
    def __call__(self, x: Float[Array, "seq_len n_input_dims"]) -> Array:
        r"""**Arguments:**

        - `x`: The input sequence. Should be a JAX array of
                shape `(seq_len, n_input_dims)`.

        **Returns:**

        - A JAX array of shape `(seq_len, n_input_dims)`.

        """
        seq_len, d = x.shape
        if d != self.n_input_dims:
            raise ValueError(
                f"Input dimension mismatch: expected {self.n_input_dims}, got {d}"
            )
        x_and_res = jax.vmap(self.in_proj)(x)
        (x, res) = jnp.split(x_and_res, 2, axis=-1)

        x = jnp.transpose(x)
        x = self.conv1d(x)[:, :seq_len]
        x = jnp.transpose(x)
        x = jax.nn.silu(x)

        y = self._ssm(x)
        y = y * jax.nn.silu(res)

        output = jax.vmap(self.out_proj)(y)
        return output

    def _ssm(self, x: Float[Array, "seq_len d_inner"]) -> Array:
        A = -jnp.exp(self.A_log)
        D = self.D

        x_delta_b_c = jax.vmap(self.x_proj)(x)

        split_indices = [
            self.dt_rank,
            self.dt_rank + self.state_space_dims,
        ]
        delta, B, C = jnp.split(x_delta_b_c, split_indices, axis=-1)
        delta = jax.nn.softplus(jax.vmap(self.dt_proj)(delta))

        y = _selective_scan(x, delta, A, B, C, D)
        return y
