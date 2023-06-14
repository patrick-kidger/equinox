from typing import Optional

import jax.random as jrandom
from jaxtyping import Array, Float, PRNGKeyArray

from .._module import field, Module


class Embedding(Module):
    """A simple lookup table that stores embeddings of a fixed size."""

    num_embeddings: int = field(static=True)
    embedding_size: int = field(static=True)
    weight: Array

    def __init__(
        self,
        num_embeddings: int,
        embedding_size: int,
        weight: Optional[Float[Array, "num_embeddings embedding_size"]] = None,
        *,
        key: PRNGKeyArray,
        **kwargs,
    ):
        """**Arguments:**

        - `num_embeddings`: Size of embedding dictionary.
        - `embedding_size`: Size of each embedding vector.
        - `weight`: If given, the embedding lookup table. Will be generated randomly
            if not provided.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__(**kwargs)
        if weight is None:
            self.weight = jrandom.normal(key, (num_embeddings, embedding_size))
        else:
            if weight.shape != (num_embeddings, embedding_size):
                raise ValueError(
                    "weight must have shape (num_embeddings, embedding_size)."
                )
            self.weight = weight
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """**Arguments:**

        - `x`: The table index.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of shape `(embedding_size,)`, from the x-th index of the embedding
        table.
        """
        return self.weight[x]
