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
            is *not* applied. If `False` then dropout is applied.
        - `deterministic`: Deprecated alternative to `inference`.

        !!! info

            The `inference` flag is provided as it is common to only apply dropout
            during training, but not to apply it during inference. If you want to change
            this flag between training and inference, then you can either:

            - Override it with the `__call__`-time `inference` flag, see below.
            - Modify the `inference` flag directly -- possible to do because `Dropout`
                is just a PyTree. For example this sets all `inference` flags to
                `True`:
                ```python
                model = ...  # some model featuring Dropout/BatchNorm/etc. layers.

                def find_inference(m):
                    has_inference = lambda x: hasattr(x, "inference")
                    leaves = jax.tree_leaves(m, is_leaf=has_inference)
                    return tuple(k.inference for k in leaves if has_inference(k))

                model = eqx.tree_at(find_inference, model, replace_fn=lambda _: True)
                ```
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
