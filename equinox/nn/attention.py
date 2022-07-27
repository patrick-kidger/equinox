import math
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom

from ..custom_types import Array
from ..module import Module, static_field
from .dropout import Dropout
from .linear import Linear


def dot_product_attention_weights(
    query: Array,
    key_: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    dropout_p: float = 0.0,
    *,
    dropout: Optional[Dropout] = None,
    key: Optional["jax.random.PRNGKey"] = None,
    inference: Optional[bool] = None,
    deterministic: Optional[bool] = None,
):

    r"""Computes dot-product attention weights given query and key.
    Used by :func:`dot_product_attention`, which is what you'll most likely use.
    But if you want access to the attention weights for introspection, then
    you can directly call this function and call einsum yourself.
    This also supports multi-query attention (https://arxiv.org/pdf/1911.02150).
    **Arguments:**

        - `query`: Query vectors. Should be a JAX array of shape
            `(batch ... query_seq_length, num_heads, qk_size)`.
        - `key_`: Key vectors. Should be a JAX array of shape
            `(batch ... kv_seq_length, num_heads, qk_size)`.
        - `mask`: Optional mask preventing attention to certain positions. Should be a
            JAX array of shape `(num_heads, query_seq_length, kv_seq_length)`.
        - `dropout_p`: Dropout probability on attention weights.
        - `dropout`: Already initialized Dropout module. Unused if `dropout_p != 0.`
          (Keyword only argument).
        - `key`: A `jax.random.PRNGKey` used for dropout. Unused if `dropout_p = 0.`.
            (Keyword only argument.)
        - `inference`: As [`equinox.nn.Dropout.__call__`][]. (Keyword only
            argument.)
        - `deterministic`: (Deprecated in favour of `inference`.)

        **Returns:**

        A JAX array of shape `(batch..., num_heads, query_seq_length, kv_seq_length)`.
    """
    if query.ndim == key_.ndim:
        assert query.shape[:-3] == key_.shape[:-3], "q, k batch dims must match."
        assert query.shape[-2] == key_.shape[-2], "q, k num_heads must match."
        assert query.shape[-1] == key_.shape[-1], "q, k depths must match."
    elif query.ndim > key_.ndim:
        # support for multi-query attention
        assert query.shape[:-3] == key_.shape[:-2], "q, k batch dims must match."
        assert query.shape[-1] == key_.shape[-1], "q, k depths must match."
    else:
        raise ValueError("q must have equal or more dimensions than k.")

    query_seq_length, num_heads, depth = (
        query.shape[-3],
        query.shape[-2],
        query.shape[-1],
    )
    kv_seq_length = key_.shape[-2] if key_.ndim < query.ndim else key_.shape[-3]

    query = query / jnp.sqrt(depth)
    if query.ndim == key_.ndim:
        attn_weights = jnp.einsum("...qhd,...khd->...hqk", query, key_)
    else:
        # multi-query attention
        attn_weights = jnp.einsum("...qhd,...kd->...hqk", query, key_)

    # apply attention bias
    if bias is not None:
        attn_weights = attn_weights + bias

    # apply attention mask
    if mask is not None:
        if mask.shape != attn_weights.shape:
            raise ValueError(
                f"mask must have shape (num_heads, query_seq_length, "
                f"kv_seq_length)=({num_heads}, {query_seq_length}, "
                f"{kv_seq_length}). Got {mask.shape}."
            )
        attn_weights = jnp.where(mask, attn_weights, -jnp.inf)

    # apply softmax to normalize
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)

    # apply dropout
    if dropout_p > 0.0:
        dropout = Dropout(dropout_p, inference=inference)

    if dropout is not None:
        attn_weights = dropout(
            attn_weights, key=key, inference=inference, deterministic=deterministic
        )

    return attn_weights


def dot_product_attention(
    query: Array,
    key_: Array,
    value: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    dropout_p: float = 0.0,
    *,
    dropout: Optional[Dropout] = None,
    key: Optional["jax.random.PRNGKey"] = None,
    inference: Optional[bool] = None,
    deterministic: Optional[bool] = None,
):

    r"""Computes dot-product attention given query, key, and value.
    This is the core function for applying attention based on
    https://arxiv.org/abs/1706.03762.

    - $\text{Attention}$ is defined as
      $\text{Attention}(\widetilde{Q}, \widetilde{K}, \widetilde{V})
       = \text{softmax}(\frac{\widetilde{Q}\widetilde{K}^\intercal}
                             {\sqrt{d_\text{qk}}})\widetilde{V}$.

    This also supports multi-query attention (https://arxiv.org/pdf/1911.02150).
    **Arguments:**

    - `query`: Query vectors. Should be a JAX array of shape
            `(batch ... query_seq_length, num_heads, qk_size)`.
    - `key_`: Key vectors. Should be a JAX array of shape
            `(batch ... kv_seq_length, num_heads, qk_size)`.
    - `value`: Value vectors. Should be a JAX array of shape
            `(batch ... kv_seq_length, num_heads, vo_size)`.
    - `mask`: Optional mask preventing attention to certain positions. Should be a
            JAX array of shape `(num_heads, query_seq_length, kv_seq_length)`.
    - `dropout_p`: Dropout probability on attention weights.
    - `dropout`: Already initialized Dropout module. Unused if `dropout_p != 0.`
          (Keyword only argument).
    - `key`: A `jax.random.PRNGKey` used for dropout. Unused if `dropout_p = None`.
            (Keyword only argument.)
    - `inference`: As [`equinox.nn.Dropout.__call__`][]. (Keyword only
            argument.)
    - `deterministic`: (Deprecated in favour of `inference`.)

    **Returns:**

    A JAX array of shape `(batch..., query_seq_length, num_heads, vo_size)`.
    """
    assert key_.ndim == value.ndim, "k, v must have same rank."
    if query.ndim == key_.ndim == value.ndim:
        assert (
            query.shape[:-3] == key_.shape[:-3] == value.shape[:-3]
        ), "q, k, v batch dims must match."
        assert (
            query.shape[-2] == key_.shape[-2] == value.shape[-2]
        ), "q, k, v num_heads must match."
        assert key_.shape[-3] == value.shape[-3], "k, v lengths must match"
    elif query.ndim > key_.ndim and query.ndim > value.ndim:
        # support for multi-query attention
        assert (
            query.shape[:-3] == key_.shape[:-2] == value.shape[:-2]
        ), "q, k batch dims must match."
        assert key_.shape[-2] == value.shape[-2], "k, v lengths must match"
    else:
        raise ValueError("q must have equal or more dimensions than k, v.")

    attn_weights = dot_product_attention_weights(
        query=query,
        key_=key_,
        bias=bias,
        mask=mask,
        dropout_p=dropout_p,
        dropout=dropout,
        key=key,
        inference=inference,
        deterministic=deterministic,
    )

    if query.ndim == value.ndim:
        return jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
    else:
        # multi-query attention
        return jnp.einsum("...hqk,...kd->...qhd", attn_weights, value)


class MultiheadAttention(Module):
    r"""
    Computes

    $$\text{MultiheadAttention}(Q, K, V)
      = \sum_i \text{Attention}\left(QW^Q_i, KW^K_i, VW^V_i\right)W^O_i$$

    where:

    - The inputs are
      $Q \in \mathbb{R}^{d_\text{seq} \times d_\text{query}}$,
      $K \in \mathbb{R}^{d_\text{seq} \times d_\text{key}}$,
      $V \in \mathbb{R}^{d_\text{seq} \times d_\text{value}}$.
      These are referred to as query, key, and value respectively. Meanwhile
      $d_\text{seq}$ is the sequence length, and $d_\text{query}$, $d_\text{key}$,
      $d_\text{value}$ are numbers of channels.

    - The trainable weights are
    $W^Q_i \in \mathbb{R}^{d_\text{query} \times d_\text{qk}}$,
    $W^K_i \in \mathbb{R}^{d_\text{key} \times d_\text{qk}}$,
    $W^V_i \in \mathbb{R}^{d_\text{value} \times d_\text{vo}}$,
    $W^O_i \in \mathbb{R}^{d_\text{vo} \times d_\text{output}}$,
    with $i \in \{1, \ldots, h\}$, where $h$ is the number of heads, and $d_\text{qk}$,
    $d_\text{vo}$, $d_\text{output}$ are hyperparameters.

    - $\text{Attention}$ is defined as
      $\text{Attention}(\widetilde{Q}, \widetilde{K}, \widetilde{V})
       = \text{softmax}(\frac{\widetilde{Q}\widetilde{K}^\intercal}
                             {\sqrt{d_\text{qk}}})\widetilde{V}$.

    ??? cite

        [Attention is All You Need](https://arxiv.org/abs/1706.03762)

        ```bibtex
        @inproceedings{vaswani2017attention,
            author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and
                    Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and
                    Kaiser, {\L}ukasz and Polosukhin, Illia},
            booktitle={Advances in Neural Information Processing Systems},
            publisher={Curran Associates, Inc.},
            title={Attention is All You Need},
            volume={30},
            year={2017}
        }
        ```

    !!! faq "FAQ"

        Different software libraries often implement multihead attention in slightly
        different ways. Some of them will or won't add on biases by default. Most of
        them will fix the values of $d_\text{qk}, d_\text{vo}, d_\text{output}$ in
        terms of $d_\text{query}$ or $d_\text{key}$ or $d_\text{value}$. Equinox
        chooses to expose all of these as options.

        Relative to the original
        [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper: our
        $d_\text{qk}$ is their "$d_k$". Our $d_\text{vo}$ is their "$d_\text{v}$". They
        fix $d_\text{query} = d_\text{key} = d_\text{value} = d_\text{output}$ and
        refer to it as "$d_\text{model}$".
    """

    query_proj: Linear
    key_proj: Linear
    value_proj: Linear
    output_proj: Linear
    dropout: Dropout

    num_heads: int = static_field()
    query_size: int = static_field()
    key_size: int = static_field()
    value_size: int = static_field()
    output_size: int = static_field()
    qk_size: int = static_field()
    vo_size: int = static_field()
    use_query_bias: bool = static_field()
    use_key_bias: bool = static_field()
    use_value_bias: bool = static_field()
    use_output_bias: bool = static_field()

    def __init__(
        self,
        num_heads: int,
        query_size: int,
        key_size: Optional[int] = None,
        value_size: Optional[int] = None,
        output_size: Optional[int] = None,
        qk_size: Optional[int] = None,
        vo_size: Optional[int] = None,
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_output_bias: bool = False,
        dropout_p: float = 0.0,
        inference: bool = False,
        *,
        key: "jax.random.PRNGKey",
        **kwargs,
    ):
        r"""**Arguments:**

        - `num_heads`: Number of parallel attention heads $h$.
        - `query_size`: Number of input channels for query $Q$.
        - `key_size`: Number of input channels for key $K$. Defaults to `query_size`.
        - `value_size`: Number of input channels for value $V$. Defaults to
            `query_size`.
        - `output_size`: Number of output channels. Defaults to `query_size`.
        - `qk_size`: Number of channels to compare query and key over, per head.
            Defaults to `query_size // num_heads`.
        - `vo_size`: Number of channels to compare attention-weighted value and output
            over, per head. Defaults to `query_size // num_heads`.
        - `use_query_bias`: Whether to use a bias term in the query projections.
        - `use_key_bias`: Whether to use a bias term in the key projections.
        - `use_value_bias`: Whether to use a bias term in the value projections.
        - `use_output_bias`: Whether to use a bias term in the output projection.
        - `dropout_p`: Dropout probability on attention weights.
        - `inference`: Whether to actually apply dropout at all. If `True` then dropout
            is not applied. If `False` then dropout is applied. This may be toggled
            with [`equinox.tree_inference`][] or overridden during
            [`equinox.nn.MultiheadAttention.__call__`][].
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__(**kwargs)
        qkey, kkey, vkey, okey = jrandom.split(key, 4)

        if key_size is None:
            key_size = query_size
        if value_size is None:
            value_size = query_size
        if qk_size is None:
            qk_size = query_size // num_heads
        if vo_size is None:
            vo_size = query_size // num_heads
        if output_size is None:
            output_size = query_size

        self.query_proj = Linear(
            query_size, num_heads * qk_size, use_bias=use_query_bias, key=qkey
        )
        self.key_proj = Linear(
            key_size, num_heads * qk_size, use_bias=use_key_bias, key=kkey
        )
        self.value_proj = Linear(
            value_size, num_heads * vo_size, use_bias=use_value_bias, key=vkey
        )
        self.output_proj = Linear(
            num_heads * vo_size, output_size, use_bias=use_output_bias, key=okey
        )

        self.dropout = Dropout(dropout_p, inference=inference)

        self.num_heads = num_heads
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.output_size = output_size
        self.qk_size = qk_size
        self.vo_size = vo_size
        self.use_query_bias = use_query_bias
        self.use_key_bias = use_key_bias
        self.use_value_bias = use_value_bias
        self.use_output_bias = use_output_bias

    def __call__(
        self,
        query: Array["query_seq_length", "query_size"],  # noqa: F821
        key_: Array["kv_seq_length", "key_size"],  # noqa: F821
        value: Array["kv_seq_length", "value_size"],  # noqa: F821
        mask: Optional[
            Array["num_heads", "query_seq_length", "kv_seq_length"]  # noqa: F821
        ] = None,
        *,
        key: Optional["jax.random.PRNGKey"] = None,
        inference: Optional[bool] = None,
        deterministic: Optional[bool] = None,
    ) -> Array["query_seq_length", "output_size"]:  # noqa: F821
        """**Arguments:**

        - `query`: Query embedding. Should be a JAX array of shape
            `(query_seq_length, query_size)`.
        - `key_`: Key embedding. Should be a JAX array of shape
            `(kv_seq_length, key_size)`.
        - `value`: Value embedding. Should be a JAX array of shape
            `(kv_seq_length, value_size)`.
        - `mask`: Optional mask preventing attention to certain positions. Should be a
            JAX array of shape `(num_heads, query_seq_length, kv_seq_length)`.
        - `key`: A `jax.random.PRNGKey` used for dropout. Unused if `dropout = 0`.
            (Keyword only argument.)
        - `inference`: As [`equinox.nn.Dropout.__call__`][]. (Keyword only
            argument.)
        - `deterministic`: (Deprecated in favour of `inference`.)

        **Returns:**

        A JAX array of shape `(query_seq_length, output_size)`.
        """

        query_seq_length, _ = query.shape
        kv_seq_length, _ = key_.shape
        kv_seq_length2, _ = value.shape
        if kv_seq_length != kv_seq_length2:
            # query length can be different
            raise ValueError("key and value must both be sequences of equal length.")

        query_heads = self._project(self.query_proj, query)
        key_heads = self._project(self.key_proj, key_)
        value_heads = self._project(self.value_proj, value)

        attn = dot_product_attention(
            query=query_heads,
            key_=key_heads,
            value=value_heads,
            mask=mask,
            dropout=self.dropout,
            key=key,
            inference=inference,
            deterministic=deterministic,
        )

        attn = attn.reshape(query_seq_length, -1)

        return jax.vmap(self.output_proj)(attn)

    def _project(self, proj, x):
        seq_length, _ = x.shape
        projection = jax.vmap(proj)(x)
        return projection.reshape(seq_length, self.num_heads, -1)
