from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom

from ..custom_types import Array
from ..module import Module, static_field
from .linear import Dropout, Linear


class MultiheadAttention(Module):
    """Multihead Attention layer from 'Attention Is All You Need' (https://arxiv.org/abs/1706.03762)"""

    embed_dim: int = static_field()
    num_heads: int = static_field()
    kdim: int = static_field()
    vdim: int = static_field()
    _qkv_same_embed_dim: bool = static_field()
    head_dim: int = static_field()
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    out_proj: Linear
    dropout: Dropout

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_bias: bool = True,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        add_bias_kv: bool = False,
        *,
        key: "jax.random.PRNGKey",
    ):
        """**Arguments:**

        - `embed_dim`: Dimension of the model.
        - `num_heads`: Number of parallel attention heads.
        - `dropout`: Dropout probability on attention matrix. Default: `0.0`.
        - `use_bias`: Whether to use a bias term on the output projection. Default: `True`.
        - `kdim`: Total number of features for keys. Default: `None` (use `kdim=embed_dim`).
        - `vdim`: Total number of features for values. Default: `None` (use `vdim=embed_dim`).
        - `add_bias_kv`: Whether to use bias term for value and key projections. Default: `False`.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        """
        super().__init__()
        key1, key2, key3, key4 = jrandom.split(key, 4)
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if kdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        if dropout == 0.0:
            self.dropout = Dropout(dropout, deterministic=True)
        else:
            self.dropout = Dropout(dropout)
        self.q_proj = Linear(
            self.embed_dim, self.num_heads * self.embed_dim, use_bias=False, key=key1
        )
        self.k_proj = Linear(
            self.kdim, self.num_heads * embed_dim, use_bias=add_bias_kv, key=key2
        )
        self.v_proj = Linear(
            self.vdim, self.num_heads * embed_dim, use_bias=add_bias_kv, key=key3
        )
        self.out_proj = Linear(
            embed_dim * num_heads, embed_dim, use_bias=use_bias, key=key4
        )

    def __call__(
        self,
        query: Array,
        key: Array,
        value: Array,
        attn_mask: Optional[Array] = None,
        *,
        key_: Optional["jax.random.PRNGKey"] = None,
    ) -> Array:
        """**Arguments:**

        - `query`: Query embedding. Should be a JAX array of shape `(sequence_lenth, embed_dim)`.
        - `key`: Key embedding. Should be a JAX array of shape `(sequence_lenth, embed_dim)`.
        - `value`: Value embedding. Should be a JAX array of shape `(sequence_lenth, embed_dim)`.
        - `attn_mask`: A mask preventing attention to certain positions.
        - `key_`: A PRNGKey used for dropout.

        **Returns:**

        A JAX array of shape `(sequence_lenth, embed_dim)`.
        """
        d1, d2 = query.shape
        query_heads = jax.vmap(self.q_proj)(query).reshape(
            self.embed_dim, self.num_heads, d1
        )
        key_heads = jax.vmap(self.k_proj)(key).reshape(self.kdim, self.num_heads, d1)
        value_heads = jax.vmap(self.v_proj)(value).reshape(
            self.vdim, self.num_heads, d1
        )

        attn_logits = jnp.einsum("...dhs,...dhS->...hsS", query_heads, key_heads)
        sqrt_key_size = jnp.sqrt(self.kdim).astype(key.dtype)
        attn_logits = attn_logits / sqrt_key_size
        attn_logits = self.dropout(attn_logits, key=key_)

        if attn_mask is not None:
            if attn_mask.ndim != attn_logits.ndim:
                raise ValueError(
                    f"Mask dimensionality {attn_mask.ndim} must match logits "
                    f"{attn_logits.ndim}."
                )
            attn_logits = jnp.where(attn_mask, attn_logits, -1e30)

        attn_weights = jax.nn.softmax(attn_logits)
        attn = jnp.einsum("...hsS,...dhS->...hsd", attn_weights, value_heads)
        attn_vec = jnp.reshape(attn, (*query.shape[:-1], -1))

        return jax.vmap(self.out_proj)(attn_vec)
