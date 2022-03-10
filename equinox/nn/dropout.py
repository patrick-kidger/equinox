from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom

from ..custom_types import Array
from ..module import Module


class Dropout(Module):
    """Applies dropout."""

    # Not static_fields as it makes sense to want to modify them via equinox.tree_at.
    p: float = 0.5
    deterministic: bool = False

    def __call__(
        self,
        x: Array,
        *,
        key: Optional["jax.random.PRNGKey"] = None,
        deterministic: Optional[bool] = None
    ) -> Array:
        """**Arguments:**

        - `x`: An any-dimensional JAX array to dropout.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for calculating
            which elements to dropout. (Keyword only argument.)
        - `deterministic`: As per [`equinox.nn.Dropout.__init__`][]. If `True` or
            `False` then it will take priority over `self.deterministic`. If `None`
            then the value from `self.deterministic` will be used.
        """

        if deterministic is None:
            deterministic = self.deterministic
        if isinstance(self.p, (int, float)) and self.p == 0:
            deterministic = True
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


Dropout.__init__.__doc__ = """**Arguments:**

- `p`: The fraction of entries to set to zero. (On average.)
- `deterministic`: Whether to actually apply dropout at all. If `True` then dropout
    is *not* applied. If `False` then dropout is applied.

!!! info

    The `deterministic` flag is provided as it is common to only apply dropout
    during training, but not to apply it during inference. If you want to change this
    flag between training and inference, then you can either:

    - Override it with the `__call__`-time `deterministic` flag, see below.
    - Modify the `deterministic` flag directly -- possible to do because `Dropout` is
        just a PyTree. For example this sets all `deterministic` flags to `True`:
        ```python
        model = ...  # some model featuring a Dropout layer somewhere
        is_dropout = lambda x: isinstance(x, Dropout)
        def find_deterministic(m):
            return tuple(d.deterministic
                         for d in jax.tree_flatten(m, is_leaf=is_dropout)[0]
                         if is_dropout(d))
        model = eqx.tree_at(find_deterministic, model, replace_fn=lambda _: True)
        ```
"""
