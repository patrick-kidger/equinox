import jax.numpy as jnp
import jax.random as jrandom

from ..module import Module


class Dropout(Module):
    p: float = 0.5
    deterministic: bool = False

    def __call__(self, x, *, key=None, deterministic=None):
        if deterministic is None:
            deterministic = self.deterministic
        if deterministic:
            return x
        elif key is None:
            raise RuntimeError(
                "Dropout requires a key when running in non-deterministic mode."
            )
        else:
            q = 1 - self.p
            mask = jrandom.bernoulli(key, q, x.shape)
            return jnp.where(mask, x / q, 0)
