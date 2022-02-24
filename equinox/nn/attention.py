from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from ..custom_types import Array
from ..module import Module, static_field
from .dropout import Dropout
from .linear import Linear


class MultiheadAttention(Module):
    """
    Multihead Attention layer from 'Attention Is All You Need' (https://arxiv.org/abs/1706.03762)
    """

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
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim = {self.embed_dim}"
                f" and num_heads = {self.num_heads})"
            )
        if self.kdim % num_heads != 0:
            raise ValueError(
                f"kdim must be divisible by num_heads (got kdim = {self.kdim} and "
                f"num_heads = {self.num_heads})"
            )
        if self.vdim % num_heads != 0:
            raise ValueError(
                f"vdim must be divisible by num_heads (got vdim = {self.vdim} and "
                f"num_heads = {self.num_heads})"
            )
        if dropout == 0.0:
            self.dropout = Dropout(dropout, deterministic=True)
        else:
            self.dropout = Dropout(dropout)
        self.q_proj = Linear(self.embed_dim, self.embed_dim, use_bias=False, key=key1)
        self.k_proj = Linear(self.kdim, self.embed_dim, use_bias=add_bias_kv, key=key2)
        self.v_proj = Linear(self.vdim, self.embed_dim, use_bias=add_bias_kv, key=key3)
        self.out_proj = Linear(embed_dim, embed_dim, use_bias=use_bias, key=key4)

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

        - `query`: Query embedding. Should be a JAX array of shape `(sequence_length, embed_dim)`.
        - `key`: Key embedding. Should be a JAX array of shape `(sequence_length, embed_dim)`.
        - `value`: Value embedding. Should be a JAX array of shape `(sequence_length, embed_dim)`.
        - `attn_mask`: A mask preventing attention to certain positions.
        - `key_`: A PRNGKey used for dropout.

        **Returns:**

        A JAX array of shape `(sequence_length, embed_dim)`.
        """
        d1, _ = query.shape
        query_heads = self._project(self.q_proj, query)
        key_heads = self._project(self.k_proj, key)
        value_heads = self._project(self.v_proj, value)
        attn_logits = jnp.einsum("...shd,...Shd->...hsS", query_heads, key_heads)
        sqrt_key_size = np.sqrt(self.kdim // self.num_heads).astype(key.dtype)
        attn_logits = attn_logits / sqrt_key_size
        attn_logits = self.dropout(attn_logits, key=key_)

        if attn_mask is not None:
            if attn_mask.ndim != attn_logits.ndim:
                raise ValueError(
                    f"Mask dimensionality {attn_mask.ndim} must match logits "
                    f"{attn_logits.ndim}."
                )
            attn_logits = jnp.where(attn_mask, attn_logits, -1e30)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        attn = jnp.einsum("...hsS,...Shd->...shd", attn_weights, value_heads)
        attn_vec = jnp.reshape(attn, (*query.shape[:-1], -1))

        return jax.vmap(self.out_proj)(attn_vec)

    def _project(self, proj, x):
        d1, _ = x.shape
        projection = jax.vmap(proj)(x).reshape(
            d1, self.num_heads, self.embed_dim // self.num_heads
        )
        return projection
