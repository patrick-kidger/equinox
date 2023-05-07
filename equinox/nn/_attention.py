import functools as ft
import math
import warnings
from typing import Literal, Optional, overload, Tuple, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Bool, Float

from .._custom_types import PRNGKey, sentinel
from .._module import Module, static_field
from ._dropout import Dropout
from ._linear import Linear
from ._stateful import State, StateIndex


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

    return jax.nn.softmax(logits, axis=-1)


def dot_product_attention(
    query: Float[Array, "q_seq qk_size"],
    key_: Float[Array, "kv_seq qk_size"],
    value: Float[Array, "kv_seq v_size"],
    mask: Optional[Bool[Array, "q_seq kv_seq"]] = None,
    dropout: Optional[Dropout] = None,
    *,
    key: Optional[PRNGKey] = None,
    inference: Optional[bool] = None,
) -> Float[Array, "q_seq v_size"]:
    weights = dot_product_attention_weights(query, key_, mask)
    if dropout is not None:
        weights = dropout(weights, key=key, inference=inference)
    attn = jnp.einsum("sS,Sd->sd", weights, value)
    return attn


class MultiheadAttention(Module):
    r"""Computes multi-head or multi-query attention. Also supports autoregressive
    decoding.

    !!! tip

        See [`equinox.nn.self_attention`][] for a convenience wrapper if you get lost in
        the following very general discussion!

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

    One common variant is multi-query attention, in which $W^K_i$ are the same for all
    $i$, and $W^V_i$ are the same for all $i$. This can help when limited by memory
    bandwidth.

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

        Multi-query attention is from [Fast Transformer Decoding: One Write-Head is
        All You Need](https://arxiv.org/abs/1911.02150)
        ```bibtex
        @article{
            author={Noam Shazeer},
            title={Fast Transformer Decoding: One Write-Head is All You Need},
            year={2019},
            journal={arXiv:1911.02150},
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
    autoregressive_index: StateIndex

    num_heads: int = static_field()
    query_size: int = static_field()
    key_size: int = static_field()
    value_size: int = static_field()
    output_size: int = static_field()
    key_multihead: bool = static_field()
    value_multihead: bool = static_field()
    query_multihead: bool = static_field()
    state_length: Optional[int] = static_field()
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
        *,
        key_size: Optional[int] = None,
        value_size: Optional[int] = None,
        output_size: Optional[int] = None,
        key_multihead: bool = True,
        value_multihead: bool = True,
        query_multihead: bool = True,
        state_length: Optional[int] = None,
        qk_size: Optional[int] = None,
        vo_size: Optional[int] = None,
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_output_bias: bool = False,
        dropout_p: float = 0.0,
        inference: bool = False,
        key: PRNGKey,
        **kwargs,
    ):
        r"""**Arguments:**

        - `num_heads`: Number of parallel attention heads $h$.
        - `query_size`: Number of input channels for query $Q$.

        **Keyword-only arguments:**

        - `key_size`: Number of input channels for key $K$. Defaults to `query_size`.
        - `value_size`: Number of input channels for value $V$. Defaults to
            `query_size`.
        - `output_size`: Number of output channels. Defaults to `query_size`.

        - `key_multihead`: if `False`, then share the key projections across all heads.
        - `value_multihead`: if `False`, then share the value projections across all
            heads.
        - `query_multihead`: if `False`, then share the query projections across all
            heads.

        - `state_length`: Used when autoregressively decoding. This is the size of the
            key and value buffers that are updated each time the module is called.

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
            initialisation.

        !!! tip "Common variants"

            Gosh, that's rather a lot of arguments. Here's how to set up some common
            versions of attention.

            If you're not performing self-attention, and have differingly-sized
            query/key/value, then you'll need to specify `key_size` and `value_size`.

            If you want to perform multi-query attention, then this can be done by
            passing `key_multihead=False` and `value_multihead=False`.

            If you want to perform autoregressive decoding, then you'll need to specify
            `state_length`, and must pass in the `state` argument at call-time. This
            will append the new key and value to those currently seen, up to the maximum
            length of `state_length`. Unlike some attention implementations, you may
            pass in arbitrarily many key and value tokens in one go (not just one at a
            time). This means for example that you can use the same code between
            non-autoregressive training and autoregressive inference: in training, just
            pass in the full key/value, and then just discard the updated state.
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

        def _make_autoregressive_cache(**_):
            if state_length is None:
                raise ValueError(
                    "Cannot use autoregressive decoding without specifying "
                    "`MultiheadAttention(..., state_length=...)`."
                )
            if key_multihead:
                key_shape = state_length, num_heads, qk_size
            else:
                key_shape = state_length, qk_size
            if value_multihead:
                value_shape = state_length, num_heads, vo_size
            else:
                value_shape = state_length, vo_size
            if jax.config.jax_enable_x64:  # pyright: ignore
                _int = jnp.int64
            else:
                _int = jnp.int32
            return jnp.empty(key_shape), jnp.empty(value_shape), jnp.zeros((), _int)

        query_proj_out_size = qk_size
        key_proj_out_size = qk_size
        value_proj_out_size = vo_size
        if query_multihead:
            query_proj_out_size = query_proj_out_size * num_heads
        if key_multihead:
            key_proj_out_size = key_proj_out_size * num_heads
        if value_multihead:
            value_proj_out_size = value_proj_out_size * num_heads
        self.query_proj = Linear(
            query_size, query_proj_out_size, use_bias=use_query_bias, key=qkey
        )
        self.key_proj = Linear(
            key_size, key_proj_out_size, use_bias=use_key_bias, key=kkey
        )
        self.value_proj = Linear(
            value_size, value_proj_out_size, use_bias=use_value_bias, key=vkey
        )
        self.output_proj = Linear(
            vo_size * num_heads, output_size, use_bias=use_output_bias, key=okey
        )
        self.dropout = Dropout(dropout_p, inference=inference)
        self.autoregressive_index = StateIndex(_make_autoregressive_cache)

        self.num_heads = num_heads
        self.query_size = query_size
        self.query_multihead = query_multihead
        self.key_multihead = key_multihead
        self.value_multihead = value_multihead
        self.key_size = key_size
        self.value_size = value_size
        self.output_size = output_size
        self.qk_size = qk_size
        self.vo_size = vo_size
        self.use_query_bias = use_query_bias
        self.use_key_bias = use_key_bias
        self.use_value_bias = use_value_bias
        self.use_output_bias = use_output_bias
        self.state_length = state_length

    @overload
    def __call__(
        self,
        query: Float[Array, "q_seq q_size"],
        key_: Float[Array, "kv_seq k_size"],
        value: Float[Array, "kv_seq v_size"],
        mask: Union[
            None,
            Bool[Array, "q_seq kv_seq"],
            Bool[Array, "num_heads q_seq kv_seq"],
            Literal["causal"],
        ] = None,
        *,
        key: Optional[PRNGKey] = None,
        inference: Optional[bool] = None,
        deterministic: Optional[bool] = None,
    ) -> Float[Array, "q_seq o_size"]:
        ...

    @overload
    def __call__(
        self,
        query: Float[Array, "q_seq q_size"],
        key_: Float[Array, "kv_seq k_size"],
        value: Float[Array, "kv_seq v_size"],
        mask: Union[
            None,
            Bool[Array, "q_seq kv_seq"],
            Bool[Array, "num_heads q_seq kv_seq"],
            Literal["causal"],
        ],
        state: State,
        *,
        key: Optional[PRNGKey] = None,
        inference: Optional[bool] = None,
        deterministic: Optional[bool] = None,
    ) -> Tuple[Float[Array, "q_seq o_size"], State]:
        ...

    @overload
    def __call__(
        self,
        query: Float[Array, "q_seq q_size"],
        key_: Float[Array, "kv_seq k_size"],
        value: Float[Array, "kv_seq v_size"],
        *,
        state: State,
        key: Optional[PRNGKey] = None,
        inference: Optional[bool] = None,
        deterministic: Optional[bool] = None,
    ) -> Tuple[Float[Array, "q_seq o_size"], State]:
        ...

    def __call__(
        self,
        query: Float[Array, "q_seq q_size"],
        key_: Float[Array, "kv_seq k_size"],
        value: Float[Array, "kv_seq v_size"],
        mask: Union[
            None,
            Bool[Array, "q_seq kv_seq"],
            Bool[Array, "num_heads q_seq kv_seq"],
            Literal["causal"],
        ] = None,
        state: State = sentinel,
        *,
        key: Optional[PRNGKey] = None,
        inference: Optional[bool] = None,
        deterministic: Optional[bool] = None,
    ) -> Union[
        Float[Array, "q_seq o_size"], Tuple[Float[Array, "q_seq o_size"], State]
    ]:
        """**Arguments:**

        - `query`: Query embedding. Should be a JAX array of shape
            `(query_seq_length, query_size)`.
        - `key_`: Key embedding. Should be a JAX array of shape
            `(kv_seq_length, key_size)`.
        - `value`: Value embedding. Should be a JAX array of shape
            `(kv_seq_length, value_size)`.
        - `mask`: Optional mask preventing attention to certain positions. Should either
            be:
            - a JAX array of shape `(query_seq_length, kv_seq_length)`;
            - a JAX array of shape `(num_heads, query_seq_length, kv_seq_length)` (for
                custom per-head masking);
            - the string `"causal"`, to automatically build a causal attention mask.
        - `state`: Optional state for the keys and values. If passed then `key_` and
            `value` will be appended to all currently seen keys and values before
            performing attention. This is commonly known as autoregressive decoding.
            This should typically be used in conjunction with `mask="causal"`.
        - `key`: A `jax.random.PRNGKey` used for dropout. Unused if `dropout = 0`.
            (Keyword only argument.)
        - `inference`: As [`equinox.nn.Dropout.__call__`][]. (Keyword only
            argument.)
        - `deterministic`: (Deprecated in favour of `inference`.)

        **Returns:**

        The output is a JAX array of shape `(query_seq_length, output_size)`.

        If `state` is not passed then just this output is returned. If `state` is passed
        then a 2-tuple of `(output, updated_state)` is returned.
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
        del kv_seq_length2

        query_heads = self._project(self.query_proj, self.query_multihead, query)
        key_heads = self._project(self.key_proj, self.key_multihead, key_)
        value_heads = self._project(self.value_proj, self.value_multihead, value)

        if state is sentinel:
            causal_mask_offset = 0
        else:
            key_state, value_state, index = state.get(self.autoregressive_index)
            key_state = lax.dynamic_update_slice_in_dim(
                key_state, key_heads, index, axis=0
            )
            value_state = lax.dynamic_update_slice_in_dim(
                value_state, value_heads, index, axis=0
            )
            causal_mask_offset = index
            index = index + kv_seq_length
            state = state.set(
                self.autoregressive_index, (key_state, value_state, index)
            )
            key_heads = key_state
            value_heads = value_state
            kv_seq_length = self.state_length

        if mask == "causal":
            query_indices = jnp.arange(query_seq_length)[:, None]
            kv_indices = jnp.arange(kv_seq_length)[None, :]
            mask = kv_indices <= query_indices + causal_mask_offset
        if state is not sentinel:
            # Also mask out the latter parts of the state we haven't written into yet.
            unwritten_mask = jnp.arange(self.state_length) < index  # pyright: ignore
            if mask is None:
                mask = jnp.broadcast_to(
                    unwritten_mask, (query_seq_length, self.state_length)
                )
            else:
                mask = mask & unwritten_mask

        attn_fn = ft.partial(
            dot_product_attention, dropout=self.dropout, inference=inference
        )
        keys = None if key is None else jax.random.split(key, self.num_heads)
        in_axes = (
            1 if self.query_multihead else None,
            1 if self.key_multihead else None,
            1 if self.value_multihead else None,
            0 if mask is not None and mask.ndim == 3 else None,
        )
        # Batch `keys` down its first axis as it is passed as a keyword argument.
        attn = jax.vmap(attn_fn, in_axes=in_axes, out_axes=1, axis_size=self.num_heads)(
            query_heads, key_heads, value_heads, mask, key=keys
        )

        attn = attn.reshape(query_seq_length, self.num_heads * self.vo_size)
        out = jax.vmap(self.output_proj)(attn)

        if state is sentinel:
            return out
        else:
            return out, state

    def _project(self, proj, multihead, x):
        seq_length, _ = x.shape
        projection = jax.vmap(proj)(x)
        if multihead:
            _, projection_size = projection.shape
            size_per_head = projection_size // self.num_heads
            projection = projection.reshape(seq_length, self.num_heads, size_per_head)
        return projection


def self_attention(
    num_heads: int,
    size: int,
    *,
    multiquery: bool = False,
    state_length: Optional[int] = None,
    key: PRNGKey,
):
    """Multi-head or multi-query attention. Also supports autoregressive decoding.

    This function is just a convenience wrapper for creating
    [`equinox.nn.MultiheadAttention`][] instances, as the full API has a great many
    options.

    **Arguments:**

    - `num_heads`: Number of parallel attention heads.
    - `size`: Number of input channels in the key, value, and query, and the number of
        channels in the output.
    - `multiquery`: if `True`, then compute multi-query rather than full multi-head
        attention. (Keyword only argument.)
    - `state_length`: Used when autoregressively decoding. This is the size of the
        key and value buffers that are updated each time the module is called. (Keyword
        only argument.)
    - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation. (Keyword only argument.)

    **Returns:**

    An [`equinox.nn.MultiheadAttention`][] instance.
    """
    return MultiheadAttention(
        num_heads=num_heads,
        query_size=size,
        state_length=state_length,
        key_multihead=not multiquery,
        value_multihead=not multiquery,
        key=key,
    )
