import functools as ft
import math
import warnings
from collections.abc import Callable
from functools import partial
from typing import cast, Optional, Union

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from .._misc import default_floating_dtype
from .._module import field, Module
from ._dropout import Dropout
from ._linear import Linear
from ._misc import named_scope


def dot_product_attention_weights(
    query: Float[Array, "q_seq qk_size"],
    key: Float[Array, "kv_seq qk_size"],
    mask: Optional[Bool[Array, "q_seq kv_seq"]] = None,
) -> Float[Array, "q_seq kv_seq"]:
    query = query / math.sqrt(query.shape[-1])
    logits = jnp.einsum("sd,Sd->sS", query, key)
    if mask is not None:
        if mask.shape != logits.shape:
            raise ValueError(
                f"mask must have shape (query_seq_length, "
                f"kv_seq_length)=({query.shape[0]}, "
                f"{key.shape[0]}). Got {mask.shape}."
            )
        logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)
        logits = cast(Array, logits)

    with jax.numpy_dtype_promotion("standard"):
        dtype = jnp.result_type(logits.dtype, jnp.float32)
    weights = jax.nn.softmax(logits.astype(dtype)).astype(logits.dtype)
    return weights


def dot_product_attention(
    query: Float[Array, "q_seq qk_size"],
    key_: Float[Array, "kv_seq qk_size"],
    value: Float[Array, "kv_seq v_size"],
    mask: Optional[Bool[Array, "q_seq kv_seq"]] = None,
    dropout: Optional[Dropout] = None,
    *,
    key: Optional[PRNGKeyArray] = None,
    inference: Optional[bool] = None,
) -> Float[Array, "q_seq v_size"]:
    weights = dot_product_attention_weights(query, key_, mask)
    if dropout is not None:
        weights = dropout(weights, key=key, inference=inference)
    attn = jnp.einsum("sS,Sd->sd", weights, value)
    return attn


class MultiheadAttention(Module, strict=True):
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

    num_heads: int = field(static=True)
    query_size: int = field(static=True)
    key_size: int = field(static=True)
    value_size: int = field(static=True)
    output_size: int = field(static=True)
    qk_size: int = field(static=True)
    vo_size: int = field(static=True)
    use_query_bias: bool = field(static=True)
    use_key_bias: bool = field(static=True)
    use_value_bias: bool = field(static=True)
    use_output_bias: bool = field(static=True)

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
        dtype=None,
        *,
        key: PRNGKeyArray,
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
            with [`equinox.nn.inference_mode`][] or overridden during
            [`equinox.nn.MultiheadAttention.__call__`][].
        - `dtype`: The dtype to use for all trainable parameters in this layer.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        dtype = default_floating_dtype() if dtype is None else dtype
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
            query_size,
            num_heads * qk_size,
            use_bias=use_query_bias,
            dtype=dtype,
            key=qkey,
        )
        self.key_proj = Linear(
            key_size, num_heads * qk_size, use_bias=use_key_bias, dtype=dtype, key=kkey
        )
        self.value_proj = Linear(
            value_size,
            num_heads * vo_size,
            use_bias=use_value_bias,
            dtype=dtype,
            key=vkey,
        )
        self.output_proj = Linear(
            num_heads * vo_size,
            output_size,
            use_bias=use_output_bias,
            dtype=dtype,
            key=okey,
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

    @named_scope("eqx.nn.MultiheadAttention")
    def __call__(
        self,
        query: Float[Array, "q_seq q_size"],
        key_: Float[Array, "kv_seq k_size"],
        value: Float[Array, "kv_seq v_size"],
        mask: Union[
            None, Bool[Array, "q_seq kv_seq"], Bool[Array, "num_heads q_seq kv_seq"]
        ] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
        inference: Optional[bool] = None,
        deterministic: Optional[bool] = None,
        process_heads: Optional[
            Callable[
                [
                    Float[Array, "seq_length num_heads qk_size"],
                    Float[Array, "seq_length num_heads qk_size"],
                    Float[Array, "seq_length num_heads vo_size"],
                ],
                tuple[
                    Float[Array, "seq_length num_heads qk_size"],
                    Float[Array, "seq_length num_heads qk_size"],
                    Float[Array, "seq_length num_heads vo_size"],
                ],
            ]
        ] = None,
    ) -> Float[Array, "q_seq o_size"]:
        """**Arguments:**

        - `query`: Query embedding. Should be a JAX array of shape
            `(query_seq_length, query_size)`.
        - `key_`: Key embedding. Should be a JAX array of shape
            `(kv_seq_length, key_size)`.
        - `value`: Value embedding. Should be a JAX array of shape
            `(kv_seq_length, value_size)`.
        - `mask`: Optional mask preventing attention to certain positions. Should either
            be a JAX array of shape `(query_seq_length, kv_seq_length)`, or (for custom
            per-head masking) `(num_heads, query_seq_length, kv_seq_length)`. A value of
            `False` at a position indicates that position should be ignored.
        - `key`: A `jax.random.PRNGKey` used for dropout. Unused if `dropout = 0`.
            (Keyword only argument.)
        - `inference`: As [`equinox.nn.Dropout.__call__`][]. (Keyword only
            argument.)
        - `deterministic`: (Deprecated in favour of `inference`.)
        - `process_heads`: A function that takes in the query, key, and value heads and
            returns new query, key, and value heads. For example, this can be
            used to implement relative positional embeddings -
            see e.g. `RotaryPositionalEmbedding`for an example. (Keyword only argument.)

        **Returns:**

        A JAX array of shape `(query_seq_length, output_size)`.
        """

        if deterministic is not None:
            inference = deterministic
            warnings.warn(
                "MultiheadAttention()(deterministic=...) is deprecated "
                "in favour of MultiheadAttention()(inference=...)"
            )

        query_seq_length, _ = query.shape
        kv_seq_length, _ = key_.shape
        kv_seq_length2, _ = value.shape
        if kv_seq_length != kv_seq_length2:
            # query length can be different
            raise ValueError("key and value must both be sequences of equal length.")

        query_heads = self._project(self.query_proj, query)
        key_heads = self._project(self.key_proj, key_)
        value_heads = self._project(self.value_proj, value)

        if process_heads is not None:
            q_shape, k_shape, v_shape = (
                query_heads.shape,
                key_heads.shape,
                value_heads.shape,
            )
            query_heads, key_heads, value_heads = process_heads(
                query_heads, key_heads, value_heads
            )

            if (
                query_heads.shape != q_shape
                or key_heads.shape != k_shape
                or value_heads.shape != v_shape
            ):
                raise ValueError(
                    "process_heads must not change the shape of the heads."
                )

        attn_fn = partial(
            dot_product_attention, dropout=self.dropout, inference=inference
        )
        keys = None if key is None else jax.random.split(key, query_heads.shape[1])
        if mask is not None and mask.ndim == 3:
            # Batch `mask` and `keys` down their 0-th dimension.
            attn = jax.vmap(attn_fn, in_axes=1, out_axes=1)(
                query_heads, key_heads, value_heads, mask=mask, key=keys
            )
        else:
            # Batch `keys` down its 0-th dimension.
            attn = jax.vmap(ft.partial(attn_fn, mask=mask), in_axes=1, out_axes=1)(
                query_heads, key_heads, value_heads, key=keys
            )
        attn = attn.reshape(query_seq_length, -1)

        return jax.vmap(self.output_proj)(attn)

    def _project(self, proj, x):
        seq_length, _ = x.shape
        projection = jax.vmap(proj)(x)
        return projection.reshape(seq_length, self.num_heads, -1)
