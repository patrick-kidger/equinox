from typing import Any, Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax._src.dtypes import TypePromotionError
from jaxtyping import Array, ArrayLike, Float, Int, PRNGKeyArray

from .._caches import cache_clears
from .._filters import is_array_like
from .._misc import default_floating_dtype
from .._module import field, Module
from ._misc import named_scope


internal_rope_embedding_cache: dict[tuple[int, Any], tuple[Array, Array]] = {}
cache_clears.append(internal_rope_embedding_cache.clear)


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
        dtype=None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        """**Arguments:**

        `Embedding` should be initialised with either:

        - `num_embeddings`: Size of embedding dictionary. Must be non-negative.
        - `embedding_size`: Size of each embedding vector. Must be non-negative.
        - `dtype`: The dtype to use for the embedding weights. Defaults to either
            `jax.numpy.float32` or `jax.numpy.float64` depending on whether JAX is in
            64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for initialisation
            of the embedding lookup table. (Keyword only argument.)

        Or:

        - `weight`: The embedding lookup table, of shape
            `(num_embeddings, embedding_size)`.
        """
        dtype = default_floating_dtype() if dtype is None else dtype
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
            self.weight = jrandom.normal(
                key, (num_embeddings, embedding_size), dtype=dtype
            )
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

    @named_scope("eqx.nn.Embedding")
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
    """A rotary positional encoding module, as described in the paper
    "RoFormer: Enhanced Transformer with Rotary Position Embedding". While this module
    can be used in any context, it is particularly useful for providing positional
    information to transformer models.

    !!! Example

        The following example demonstrates how to use `RotaryPositionalEmbedding` in
        a simple transformer model.

        ```python
        class TransformerBlock(eqx.Module):
            rope_embeddings: RotaryPositionalEmbedding

            def __init__(...):
                self.rope_embeddings = RotaryPositionalEmbedding(...)

            def __call__(...):
                def process_heads(
                    query_heads: Float[Array, "seq_length num_heads qk_size"],
                    key_heads: Float[Array, "seq_length num_heads qk_size"],
                    value_heads: Float[Array, "seq_length num_heads vo_size"]
                ) -> tuple[
                    Float[Array, "seq_length num_heads qk_size"],
                    Float[Array, "seq_length num_heads qk_size"],
                    Float[Array, "seq_length num_heads vo_size"]
                ]:
                    query_heads = jax.vmap(self.rope_embeddings,
                                           in_axes=1,
                                           out_axes=1)(query_heads)
                    key_heads = jax.vmap(self.rope_embeddings,
                                         in_axes=1,
                                         out_axes=1)(key_heads)

                    return query_heads, key_heads, value_heads

                x = self.mha_attention(... process_heads=process_heads)
                ...
        ```

    ??? cite

        [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864)

        ```bibtex
        @misc{su2023roformer,
          title={RoFormer: Enhanced Transformer with Rotary Position Embedding},
          author={Jianlin Su and Yu Lu and Shengfeng Pan and Ahmed Murtadha and
              Bo Wen and Yunfeng Liu},
          year={2023},
          eprint={arXiv:2104.09864},
        }
        ```
    """

    embedding_size: int = field(static=True)
    theta: float = field(static=True, default=10_000.0)
    dtype: Any = field(static=True, default_factory=default_floating_dtype)

    def __check_init__(self):
        if self.embedding_size < 0:
            raise ValueError("`embedding_size` must not be negative.")
        if (self.embedding_size % 2) != 0:
            raise ValueError("`embedding_size` must be even.")

    @staticmethod
    def rotate_half(x: Float[Array, "seq_length embedding_size"]):
        d_2 = x.shape[-1] // 2
        return jnp.concatenate([-x[..., d_2:], x[..., :d_2]], axis=-1)

    @staticmethod
    def precompute_freqs_cis(
        embedding_size: int, end: int, theta: float, dtype: Any
    ) -> tuple[Float[Array, "end half_emb_size"], Float[Array, "end half_emb_size"]]:
        freqs = 1.0 / (
            theta
            ** (jnp.arange(0.0, embedding_size, 2)[jnp.newaxis, :] / embedding_size)
        )

        t = jnp.arange(float(end))
        freqs_outer = jnp.outer(t, freqs)

        # we assign the type at the very end to minimize the loss of precision
        return jnp.cos(freqs_outer).astype(dtype), jnp.sin(freqs_outer).astype(dtype)

    @named_scope("eqx.nn.RotaryPositionalEmbedding")
    def __call__(
        self,
        x: Float[Array, "seq_length embedding_size"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "seq_length embedding_size"]:
        """**Arguments:**

        - `x`: A JAX array of shape `(seq_length, embedding_size)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of shape `(seq_length, embedding_size)`, with the rotary positional
        encoding applied to the input.
        """

        seq_len, embedding_size = x.shape
        if embedding_size != self.embedding_size:
            raise ValueError(
                f"x.shape[-1] must match self.embedding_size, "
                f"but {x.shape[-1]} != {self.embedding_size}"
            )

        with jax.ensure_compile_time_eval():
            cache_key = (embedding_size, self.dtype)
            if cache_key not in internal_rope_embedding_cache:
                internal_rope_embedding_cache[cache_key] = self.precompute_freqs_cis(
                    embedding_size, seq_len, self.theta, self.dtype
                )

            freqs_cos, freqs_sin = internal_rope_embedding_cache[cache_key]
            freqs_seq_len, _ = freqs_cos.shape
            if seq_len > freqs_seq_len:
                internal_rope_embedding_cache[cache_key] = self.precompute_freqs_cis(
                    embedding_size, seq_len, self.theta, self.dtype
                )
                freqs_cos, freqs_sin = internal_rope_embedding_cache[cache_key]

            freqs_cos = freqs_cos[:seq_len]
            freqs_sin = freqs_sin[:seq_len]

        freqs_cos = jnp.tile(freqs_cos, (1, 2))
        freqs_sin = jnp.tile(freqs_sin, (1, 2))

        rotate_x = self.rotate_half(x)
        try:
            x_rope = (x * freqs_cos) + (rotate_x * freqs_sin)
        except TypePromotionError as e:
            inp_dtype = jnp.dtype(x.dtype)
            rope_dtype = jnp.dtype(self.dtype)
            raise TypePromotionError(
                f"The type of the passed value differs from the type "
                f"of the rotary embeddings ({inp_dtype} != {rope_dtype}), thus leading "
                "to a conflict when numpy_dtype_promotion is set to strict. To avoid "
                f"this error, either initialiaze RoPE module with {inp_dtype} "
                f"dtype, or explicitly cast the input argument to {rope_dtype}."
            ) from e
        return x_rope.astype(x.dtype)


RotaryPositionalEmbedding.__init__.__doc__ = """**Arguments:**

- `embedding_size`: Size of each embedding vector. Must be non-negative and even.
- `theta`: The base frequency for the sinusoidal functions used in positional encoding.
    Specifies how quickly the inner-product will decay with relative distance between
    tokens. Larger values of theta will result in slower oscillations. Default is
    10_000, as per the original paper.
- `dtype`: The dtype to use for the precomputed frequencies. Defaults to either
    `jax.numpy.float32` or `jax.numpy.float64` depending on whether JAX is in
    64-bit mode.
"""
