from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, ArrayLike, Complex, Float, Int, PRNGKeyArray

from .._filters import is_array_like
from .._module import field, Module


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


class RotaryPositionalEmbedding(Module):
    embedding_size: int = field(static=True)
    max_seq_len: int = field(static=True)
    freqs_cis: Complex[Array, "embedding_size/2"] = field(static=True)

    def __init__(
        self,
        embedding_size: int,
        max_seq_len: int,
        *,
        key: Optional[PRNGKeyArray] = None,
        **kwargs,
    ):
        """**Arguments:**

        `RotaryPositionalEmbedding` requires:

        - `embedding_size`: Size of the token embeddings. Must be non-negative.
        - `max_seq_len`: The maximum sequence length. Must be non-negative.
        - `key`: Not used; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        """
        super().__init__(**kwargs)
        if embedding_size < 0:
            raise ValueError("embedding_size must not be negative.")
        if max_seq_len < 0:
            raise ValueError("max_seq_len must not be negative.")
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len
        self.freqs_cis = self.precompute_freqs_cis(embedding_size, max_seq_len)

    @staticmethod
    def negate_half(x: Float[Array, "max_seq_len embedding_size"]):
        d_2 = x.shape[-1] // 2
        return jnp.concatenate([x[..., :d_2], -x[..., d_2:]], axis=-1)

    @staticmethod
    def precompute_freqs_cis(
        embedding_size: int, end: int, theta: float = 10000.0
    ) -> Complex[Array, "end embedding_size/2"]:
        def polar(abs, angle):
            return jnp.array(
                abs * jnp.cos(angle) + abs * jnp.sin(angle) * 1j, dtype=jnp.complex64
            )

        freqs = 1.0 / (
            theta ** (jnp.arange(0, embedding_size, 2)[jnp.newaxis, :] / embedding_size)
        )
        t = jnp.arange(end)
        freqs_outer = jnp.outer(t, freqs)
        freqs_cis = polar(jnp.ones_like(freqs_outer), freqs_outer)
        return freqs_cis

    @jax.named_scope("eqx.nn.RotaryPositionalEmbedding")
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

        A JAX array of shape `(max_seq_len, embedding_size)`, with the rotary positional
        encoding applied to the input.
        """

        max_seq_len, embedding_size = x.shape
        assert embedding_size == self.embedding_size, (
            f"x.shape[-1] must match self.embedding_size, "
            f"but {x.shape[-1]} != {self.embedding_size}"
        )
        assert (
            embedding_size % 2 == 0
        ), f"x.shape[-1] must be even, but {x.shape[-1]} is not even."
        assert max_seq_len == self.max_seq_len, (
            f"x.shape[0] must be == self.max_seq_len, "
            f"but {x.shape[0]} != {self.max_seq_len}"
        )
        neg_half_x = self.negate_half(x)
        freqs_real = jnp.tile(self.freqs_cis.real, (1, 2))
        freqs_imag = jnp.tile(self.freqs_cis.imag, (1, 2))

        x_rope = (x * freqs_real) + (neg_half_x * freqs_imag)
        return x_rope


class SinusoidalPositionalEmbedding(Module):
    embedding_size: int = field(static=True)
    max_seq_len: int = field(static=True)

    freq_cis: Float[Array, "max_seq_len embedding_size"] = field(static=True)

    def __init__(
        self,
        embedding_size: int,
        max_seq_len: int,
        *,
        key: Optional[PRNGKeyArray] = None,
        **kwargs,
    ):
        """**Arguments:**

        `SinusoidalPositionalEmbedding` requires:

        - `embedding_size`: Size of the token embeddings. Must be non-negative.
        - `max_seq_len`: The maximum sequence length. Must be non-negative.
        - `key`: Not used; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        """
        super().__init__(**kwargs)
        assert (
            embedding_size % 2 == 0
        ), f"embedding_size must be even, but {embedding_size} is not even."
        assert max_seq_len > 0, f"max_seq_len must be positive, but {max_seq_len} <= 0."
        assert (
            embedding_size > 0
        ), f"embedding_size must be positive, but {embedding_size} <= 0."
        self.freq_cis = self.get_positional_encoding(max_seq_len, embedding_size)
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len

    @staticmethod
    def get_positional_encoding(
        max_seq_len: int, embedding_size: int
    ) -> Float[Array, "max_seq_len embedding_size"]:
        pos = jnp.arange(max_seq_len)[:, jnp.newaxis]
        div_term = jnp.exp(
            jnp.arange(0, embedding_size, 2) * -(jnp.log(10000.0) / embedding_size)
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
        max_seq_len, embedding_size = x.shape
        assert embedding_size == self.embedding_size, (
            f"x.shape[-1] must match self.embedding_size, "
            f"but {x.shape[-1]} != {self.embedding_size}"
        )
        assert max_seq_len == self.max_seq_len, (
            f"x.shape[0] must be == self.max_seq_len, "
            f"but {x.shape[0]} != {self.max_seq_len}"
        )
        assert (
            x.shape == self.freq_cis.shape
        ), f"x.shape must be freq_cis.shape, but {x.shape} != {self.freq_cis.shape}"
        return x + self.freq_cis
