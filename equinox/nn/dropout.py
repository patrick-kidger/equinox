import warnings
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom

from ..custom_types import Array
from ..module import Module


class Dropout(Module):
    """Applies dropout."""

    # Not static_fields as it makes sense to want to modify them via equinox.tree_at.
    p: float
    inference: bool

    def __init__(
        self,
        p: float = 0.5,
        inference: bool = False,
        *,
        deterministic: Optional[bool] = None
    ):
        """**Arguments:**

        - `p`: The fraction of entries to set to zero. (On average.)
        - `inference`: Whether to actually apply dropout at all. If `True` then dropout
            is *not* applied. If `False` then dropout is applied. This may be toggled
            with [`equinox.tree_inference`][] or overridden during
            [`equinox.nn.Dropout.__call__`][].
        - `deterministic`: Deprecated alternative to `inference`.
        """

        if deterministic is not None:
            inference = deterministic
            warnings.warn(
                "Dropout(deterministic=...) is deprecated "
                "in favour of Dropout(inference=...)"
            )
        self.p = p
        self.inference = inference

    # Backward compatibility
    @property
    def deterministic(self):
        return self.inference

    def __call__(
        self,
        x: Array,
        *,
        key: Optional["jax.random.PRNGKey"] = None,
        inference: Optional[bool] = None,
        deterministic: Optional[bool] = None
    ) -> Array:
        """**Arguments:**

        - `x`: An any-dimensional JAX array to dropout.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for calculating
            which elements to dropout. (Keyword only argument.)
        - `inference`: As per [`equinox.nn.Dropout.__init__`][]. If `True` or
            `False` then it will take priority over `self.inference`. If `None`
            then the value from `self.inference` will be used.
        - `deterministic`: Deprecated alternative to `inference`.
        """

        if deterministic is not None:
            inference = deterministic
            warnings.warn(
                "Dropout()(deterministic=...) is deprecated "
                "in favour of Dropout()(inference=...)"
            )

        if inference is None:
            inference = self.inference
        if isinstance(self.p, (int, float)) and self.p == 0:
            inference = True
        if inference:
            return x
        elif key is None:
            raise RuntimeError(
                "Dropout requires a key when running in non-deterministic mode."
            )
        else:
            q = 1 - self.p
            mask = jrandom.bernoulli(key, q, x.shape)
            return jnp.where(mask, x / q, 0)
