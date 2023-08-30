import math
from typing import Any, Literal, Optional, TypeVar, Union

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from .._module import field, Module


class Linear(Module):
    """Performs a linear transformation."""

    weight: Array
    bias: Optional[Array]
    in_features: Union[int, Literal["scalar"]] = field(static=True)
    out_features: Union[int, Literal["scalar"]] = field(static=True)
    use_bias: bool = field(static=True)

    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        use_bias: bool = True,
        *,
        key: PRNGKeyArray
    ):
        """**Arguments:**

        - `in_features`: The input size. The input to the layer should be a vector of
            shape `(in_features,)`
        - `out_features`: The output size. The output from the layer will be a vector
            of shape `(out_features,)`.
        - `use_bias`: Whether to add on a bias as well.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        Note that `in_features` also supports the string `"scalar"` as a special value.
        In this case the input to the layer should be of shape `()`.

        Likewise `out_features` can also be a string `"scalar"`, in which case the
        output from the layer will have shape `()`.
        """
        super().__init__()
        wkey, bkey = jrandom.split(key, 2)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features
        lim = 1 / math.sqrt(in_features_)
        self.weight = jrandom.uniform(
            wkey, (out_features_, in_features_), minval=-lim, maxval=lim
        )
        if use_bias:
            self.bias = jrandom.uniform(bkey, (out_features_,), minval=-lim, maxval=lim)
        else:
            self.bias = None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    @jax.named_scope("eqx.nn.Linear")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape `(in_features,)`. (Or shape
            `()` if `in_features="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        !!! info

            If you want to use higher order tensors as inputs (for example featuring "
            "batch dimensions) then use `jax.vmap`. For example, for an input `x` of "
            "shape `(batch, in_features)`, using
            ```python
            linear = equinox.nn.Linear(...)
            jax.vmap(linear)(x)
            ```
            will produce the appropriate output of shape `(batch, out_features)`.

        **Returns:**

        A JAX array of shape `(out_features,)`. (Or shape `()` if
        `out_features="scalar"`.)
        """

        if self.in_features == "scalar":
            if jnp.shape(x) != ():
                raise ValueError("x must have scalar shape")
            x = jnp.broadcast_to(x, (1,))
        x = self.weight @ x
        if self.bias is not None:
            x = x + self.bias
        if self.out_features == "scalar":
            assert jnp.shape(x) == (1,)
            x = jnp.squeeze(x)
        return x


_T = TypeVar("_T")


class Identity(Module):
    """Identity operation that does nothing. Sometimes useful as a placeholder for
    another Module.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Consumes arbitrary `*args` and `**kwargs` but ignores them."""
        # Ignores args and kwargs
        super().__init__()

    @jax.named_scope("eqx.nn.Identity")
    def __call__(self, x: _T, *, key: Optional[PRNGKeyArray] = None) -> _T:
        """**Arguments:**

        - `x`: The input, of any type.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        The input, unchanged.
        """
        return x
