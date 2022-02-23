from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom

from ..custom_types import Array
from ..module import Module, static_field


class Embedding(Module):
    """Simple lookup table style embedding"""

    num_embeddings: int = static_field()
    embedding_dim: int = static_field()
    weight: Array

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        weight: Optional[Array] = None,
        *,
        key: "jax.random.PRNGKey",
    ):
        """**Arguments:**

        - `num_embeddings`: Size of embedding dictionary.
        - `embedding_dim`: Size of each embedding vector.
        - `weight`: If given, the embedding lookup table.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if weight is None:
            self.weight = jrandom.normal(key, (num_embeddings, embedding_dim))
        else:
            if list(weight.shape) != [num_embeddings, embedding_dim]:
                raise ValueError(
                    f"Shape of weight ({weight.shape}) does not match num_embeddings"
                    f" ({num_embeddings}) and embedding_dim ({embedding_dim})"
                )
            self.weight = weight

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**

        - `x`: The table index.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of shape `embedding_dim` that gives the xth index of the embedding table.
        """
        if not jnp.issubdtype(x, jnp.integer):
            raise ValueError(
                f"Input must be an array of integer dtype but input was {x.dtype}"
            )
        return self.weight[(x,)]
