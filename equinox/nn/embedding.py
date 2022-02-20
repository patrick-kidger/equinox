from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom

from ..custom_types import Array
from ..module import Module, static_field


class Embedding(Module):

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
        key1, _ = jrandom.split(key, 2)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if weight is None:
            self.weight = jrandom.normal(key1, (num_embeddings, embedding_dim))
        else:
            if list(weight.shape) != [num_embeddings, embedding_dim]:
                raise ValueError(
                    f"Shape of weight ({weight.shape}) does not match num_embeddings ({num_embeddings})"
                    f" and embedding_dim ({embedding_dim})"
                )
            self.weight = weight

    def __call__(self, x: Array) -> Array:
        if not jnp.issubdtype(x, jnp.integer):
            raise ValueError(
                f"Input must be an array of integer dtype but input was {x.dtype}"
            )
        return self.weight[(x,)]
