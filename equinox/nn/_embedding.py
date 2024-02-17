from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, ArrayLike, Complex, Float, Int, PRNGKeyArray

from .._caches import cache_clears
from .._filters import is_array_like
from .._module import field, Module


internal_rope_embedding_cache = {}
internal_sinusoidal_positional_encoding_cache = {}
cache_clears.append(internal_rope_embedding_cache.clear)
cache_clears.append(internal_sinusoidal_positional_encoding_cache.clear)


class Embedding(Module, strict=True):
    """A simple lookup table that stores embeddings of a fixed size."""

    num_embeddings: int = field(static=True)
    embedding_size: int = field(static=True)
    weight: Array

    def __init__(
        self,
        num_embeddings: Optional[int] = None,  # pyright: ignore
        embedding_size: Optional[int] = None,  # pyright: ignore
        weight: Optional[Float[Array, "num_embeddings embedding_size"]] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        """**Arguments:**

        `Embedding` should be initialised with either:

        - `num_embeddings`: Size of embedding dictionary. Must be non-negative.
        - `embedding_size`: Size of each embedding vector. Must be non-negative.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for initialisation
            of the embedding lookup table. (Keyword only argument.)

        Or:

        - `weight`: The embedding lookup table, of shape
            `(num_embeddings, embedding_size)`.
        """
        if weight is None:
            if num_embeddings is None or embedding_size is None or key is None:
                raise ValueError(
                    "Must provide `eqx.nn.Embedding(num_embeddings=..., "
                    "embedding_size=..., key=...)` if not providing the weight "
                    "directly."
                )
            if num_embeddings < 0:
                raise ValueError("num_embeddings must not be negative.")
            if embedding_size < 0:
                raise ValueError("embedding_size must not be negative.")
            self.weight = jrandom.normal(key, (num_embeddings, embedding_size))
        else:
            if weight.ndim != 2:
                raise ValueError(
                    "weight must have shape (num_embeddings, embedding_size)."
                )
            if num_embeddings is None:
                num_embeddings: int = weight.shape[0]
            if embedding_size is None:
                embedding_size: int = weight.shape[1]
            if weight.shape != (num_embeddings, embedding_size):
                raise ValueError(
                    "weight must have shape (num_embeddings, embedding_size)."
                )
            self.weight = weight
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size

    @jax.named_scope("eqx.nn.Embedding")
    def __call__(
        self, x: Int[ArrayLike, ""], *, key: Optional[PRNGKeyArray] = None
    ) -> Array:
        """**Arguments:**

        - `x`: The table index. Should be a scalar integer array.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of shape `(embedding_size,)`, from the x-th index of the embedding
        table.
        """
        if is_array_like(x) and jnp.shape(x) == ():
            return self.weight[x]
        else:
            raise ValueError(
                "`eqx.nn.Embedding()(x)` should be called with a scalar index `x`. "
                "Use `jax.vmap` if you would like to index with multiple values."
            )


class RotaryPositionalEmbedding(Module, strict=True):
    """
    A rotary positional encoding module, as described in the paper
    "RoFormer: Enhanced Transformer with Rotary Position Embedding". While
    this module can be used in any context, it is particularly useful for
    providing positional information to transformer models.

    !!! example
        The following example demonstrates how to use `RotaryPositionalEmbedding` in
        a simple transformer model.
        ```python

        class TransformerBlock(eqx.nn.StatefulLayer):
            ...
            key_rope_embeddings: RotaryPositionalEmbedding
            query_rope_embeddings: RotaryPositionalEmbedding

            def __init__(...):
                ...
                self.query_rope_embeddings = RotaryPositionalEmbedding(
                    embedding_size=n_embd, max_seq_len=max_seq_len
                )
                self.key_rope_embeddings = RotaryPositionalEmbedding(
                    embedding_size=n_embd, max_seq_len=max_seq_len
                )
                ...

            def __call__(...):
                def process_heads(query_heads, key_heads, value_heads):
                    query_heads = jax.vmap(self.query_rope_embeddings,
                                           in_axes=1,
                                           out_axes=1)(query_heads)
                    key_heads = jax.vmap(self.key_rope_embeddings,
                                         in_axes=1,
                                         out_axes=1)(key_heads)

                    return query_heads, key_heads, value_heads

                mha_output = self.mha_attention(
                    process_heads=process_heads,
                    query=jax.vmap(self.rms_norm)(x),
                    key_=jax.vmap(self.rms_norm)(x),
                    value=jax.vmap(self.rms_norm)(x),
                    mask=mask,
                )
        ```

    ??? cite

        [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864)

        ```bibtex
            @misc{su2023roformer,
              title={RoFormer: Enhanced Transformer with Rotary Position Embedding},
              author={Jianlin Su and Yu Lu and Shengfeng Pan and Ahmed Murtadha and
              Bo Wen and Yunfeng Liu},
              year={2023},
              eprint={2104.09864},
              archivePrefix={arXiv},
              primaryClass={cs.CL}
            }
        ```
    """

    embedding_size: int = field(static=True)
    max_seq_len: Optional[int] = field(static=True)

    def __init__(
        self,
        embedding_size: int,
        max_seq_len: Optional[int] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
        **kwargs,
    ):
        """**Arguments:**
        `RotaryPositionalEmbedding` requires:

        - `embedding_size`: Size of the token embeddings. Must be non-negative.
        - `max_seq_len`: The maximum sequence length. Must be non-negative if provided.
        - `key`: Not used; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        """
        if embedding_size < 0:
            raise ValueError("embedding_size must not be negative.")
        if max_seq_len is not None and max_seq_len < 0:
            raise ValueError("max_seq_len must not be negative.")
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len

    @staticmethod
    def negate_half(x: Float[Array, "max_seq_len embedding_size"]):
        d_2 = x.shape[-1] // 2
        return jnp.concatenate([x[..., :d_2], -x[..., d_2:]], axis=-1)

    @staticmethod
    def precompute_freqs_cis(
        embedding_size: int, end: int, theta: float = 10000.0
    ) -> Complex[Array, "end embedding_size/2"]:
        freqs = 1.0 / (
            theta ** (jnp.arange(0, embedding_size, 2)[jnp.newaxis, :] / embedding_size)
        )
        t = jnp.arange(end)
        freqs_outer = jnp.outer(t, freqs)
        freqs_cis = jnp.cos(freqs_outer) + jnp.sin(freqs_outer) * 1j
        return freqs_cis

    @jax.named_scope("eqx.nn.RotaryPositionalEmbedding")
    def __call__(
        self,
        x: Float[Array, "seq_len embedding_size"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "max_seq_len embedding_size"]:
        """**Arguments:**

        - `x`: A JAX array of shape `(seq_len, embedding_size)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of shape `(seq_len, embedding_size)`, with the rotary positional
        encoding applied to the input.
        """

        seq_len, embedding_size = x.shape
        if embedding_size != self.embedding_size:
            raise ValueError(
                f"x.shape[-1] must match self.embedding_size, "
                f"but {x.shape[-1]} != {self.embedding_size}"
            )
        if embedding_size % 2 != 0:
            raise ValueError(
                f"x.shape[-1] must be even, but {x.shape[-1]} is not even."
            )

        if self.max_seq_len is not None and seq_len > self.max_seq_len:
            raise ValueError(
                f"x.shape[0] must be <= self.max_seq_len, "
                f"but {x.shape[0]} > {self.max_seq_len}"
            )

        neg_half_x = self.negate_half(x)

        if (embedding_size, seq_len) in internal_rope_embedding_cache:
            freqs_cis = internal_rope_embedding_cache[(embedding_size, seq_len)]
        else:
            freqs_cis = self.precompute_freqs_cis(embedding_size, seq_len)
            internal_rope_embedding_cache[(embedding_size, seq_len)] = freqs_cis

        assert freqs_cis is not None, "freqs_cis must not be None."
        freqs_cis = jax.lax.stop_gradient(freqs_cis)
        freqs_real = jnp.tile(freqs_cis.real, (1, 2))
        freqs_imag = jnp.tile(freqs_cis.imag, (1, 2))

        x_rope = (x * freqs_real) + (neg_half_x * freqs_imag)
        return x_rope


class SinusoidalPositionalEmbedding(Module):
    r"""
    A sinusoidal positional encoding module, as described in the paper
    "Attention is All You Need". While this module can be used in any context, it is
    particularly useful for providing positional information to transformer models.

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
    """

    embedding_size: int = field(static=True)
    max_seq_len: Optional[int] = field(static=True)
    theta: float = field(static=True)

    def __init__(
        self,
        embedding_size: int,
        max_seq_len: Optional[int] = None,
        *,
        theta: float = 10000.0,
        key: Optional[PRNGKeyArray] = None,
        **kwargs,
    ):
        """**Arguments:**
        `SinusoidalPositionalEmbedding` requires:

        - `embedding_size`: Size of the token embeddings. Must be non-negative.
        - `max_seq_len`: The maximum sequence length. Must be non-negative if provided.
        - `theta`: The frequency of the sinusoidal positional encoding.
            Must be positive. Defaults to 10000.0.
        - `key`: Not used; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        """

        if embedding_size % 2 != 0:
            raise ValueError(
                f"embedding_size must be even, but {embedding_size} is not even."
            )

        if max_seq_len is not None and max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, but {max_seq_len} <= 0.")

        if embedding_size <= 0:
            raise ValueError(
                f"embedding_size must be positive, but {embedding_size} <= 0."
            )

        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len
        self.theta = theta

    @staticmethod
    def get_positional_encoding(
        embedding_size: int, max_seq_len: int, theta: float = 10000.0
    ) -> Float[Array, "max_seq_len embedding_size"]:
        pos = jnp.arange(max_seq_len)[:, jnp.newaxis]
        div_term = jnp.exp(
            jnp.arange(0, embedding_size, 2) * -(jnp.log(theta) / embedding_size)
        )
        # the following expression is closer to the actual notation they used.
        # div_term = 1 / 10000 ** (jnp.arange(0, embedding_size, 2) / embedding_size)
        pos_enc = jnp.zeros((max_seq_len, embedding_size))
        pos_enc = pos_enc.at[:, 0::2].set(jnp.sin(pos * div_term))
        pos_enc = pos_enc.at[:, 1::2].set(jnp.cos(pos * div_term))
        return pos_enc

    @jax.named_scope("eqx.nn.SinusoidalPositionalEmbedding")
    def __call__(
        self,
        x: Float[Array, "max_seq_len embedding_size"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "max_seq_len embedding_size"]:
        """**Arguments:**

        - `x`: A JAX array of shape `(max_seq_len, embedding_size)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of shape `(max_seq_len, embedding_size)`, with the sinusoidal
        positional encoding applied to the input.
        """
        seq_len, embedding_size = x.shape

        if embedding_size != self.embedding_size:
            raise ValueError(
                f"x.shape[-1] must match self.embedding_size, "
                f"but {x.shape[-1]} != {self.embedding_size}"
            )

        freqs_cis = None
        if (
            embedding_size,
            seq_len,
            self.theta,
        ) in internal_sinusoidal_positional_encoding_cache:
            freqs_cis = internal_sinusoidal_positional_encoding_cache[
                (embedding_size, seq_len, self.theta)
            ]
        else:
            freqs_cis = self.get_positional_encoding(
                embedding_size, seq_len, self.theta
            )
            internal_rope_embedding_cache[
                (embedding_size, seq_len, self.theta)
            ] = freqs_cis

        assert freqs_cis is not None, "freqs_cis must not be None."

        if x.shape != freqs_cis.shape:
            raise ValueError(
                f"x.shape must be freq_cis.shape, but {x.shape} != {freqs_cis.shape}"
            )

        return x + freqs_cis
