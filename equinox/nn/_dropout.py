import warnings
from typing import Optional

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray

from .._errors import error_if
from .._filters import is_array
from .._module import field, Module


class Dropout(Module, strict=True):
    """Applies dropout.

    Note that this layer behaves differently during training and inference. During
    training then dropout is randomly applied; during inference this layer does nothing.
    Whether the model is in training or inference mode should be toggled using
    [`equinox.nn.inference_mode`][].
    """

    # Not static fields as it makes sense to modify them via equinox.tree_at
    p: Float[ArrayLike, ""] = field(
        converter=lambda x: x if is_array(x) else np.array(x, dtype=np.float32)
    )
    inference: bool

    def __init__(
        self,
        p: Float[ArrayLike, ""] = 0.5,
        inference: bool = False,
        *,
        deterministic: Optional[bool] = None,
    ):
        """**Arguments:**

        - `p`: The fraction of entries to set to zero. (On average.)
        - `inference`: Whether to actually apply dropout at all. If `True` then dropout
            is *not* applied. If `False` then dropout is applied. This may be toggled
            with [`equinox.nn.inference_mode`][] or overridden during
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

    @jax.named_scope("eqx.nn.Dropout")
    def __call__(
        self,
        x: Array,
        *,
        key: Optional[PRNGKeyArray] = None,
        inference: Optional[bool] = None,
        deterministic: Optional[bool] = None,
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

        if inference:
            return x
        else:
            p = self.p
            if key is None:
                p = error_if(
                    p,
                    p != 0,
                    "Dropout requires a key when running in non-"
                    "deterministic mode with non-zero probability.",
                )

                # placeholder value for the key;
                # this statement is only reachable (during execution)
                # if the probability is zero, so the key does not matter.
                key = jrandom.PRNGKey(0)

            q = 1 - lax.stop_gradient(p)
            mask = jrandom.bernoulli(key, q, x.shape)
            return jnp.where(mask, x / q, 0)
