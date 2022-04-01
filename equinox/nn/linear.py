import math
from typing import Optional, TypeVar

import jax
import jax.random as jrandom

from ..custom_types import Array
from ..module import Module, static_field


class Linear(Module):
    """Performs a linear transformation."""

    weight: Array
    bias: Optional[Array]
    in_features: int = static_field()
    out_features: int = static_field()
    use_bias: bool = static_field()

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        *,
        key: "jax.random.PRNGKey"
    ):
        """**Arguments:**

        - `in_features`: The input size.
        - `out_features`: The output size.
        - `use_bias`: Whether to add on a bias as well.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__()
        wkey, bkey = jrandom.split(key, 2)
        lim = 1 / math.sqrt(in_features)
        self.weight = jrandom.uniform(
            wkey, (out_features, in_features), minval=-lim, maxval=lim
        )
        if use_bias:
            self.bias = jrandom.uniform(bkey, (out_features,), minval=-lim, maxval=lim)
        else:
            self.bias = None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape `(in_features,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        !!! info

            Note that in contrast to [PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) or other
            deep learning frameworks where the input `x` is allowed to be a higher order tensor of shape
            `(..., in_features)`, here we require `x` to be a vector. If you want to use higher order tensors as
            inputs, use `jax.vmap`.
            For example, for an input `x` of shape `(d, in_features)`, using
            `jax.vmap(linear, in_axes=0, out_axes=0)(x)` produces the appropriate output of shape `(d, out_features)`.

        **Returns:**

        A JAX array of shape `(out_features,)`
        """

        x = self.weight @ x
        if self.bias is not None:
            x = x + self.bias
        return x


_T = TypeVar("T")


class Identity(Module):
    """Identity operation that does nothing. Sometimes useful as a placeholder for
    another Module.
    """

    def __init__(self, *args, **kwargs):
        """Consumes arbitrary `*args` and `**kwargs` but ignores them."""
        # Ignores args and kwargs
        super().__init__()

    def __call__(self, x: _T, *, key: Optional["jax.random.PRNGKey"] = None) -> _T:
        """**Arguments:**

        - `x`: The input, of any type.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        The input, unchanged.
        """
        return x
