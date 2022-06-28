from typing import Hashable, Optional, Sequence, Union

import jax
import jax.lax as lax
import jax.numpy as jnp

from ..custom_types import Array
from ..module import Module, static_field
from .stateful import get_state, set_state, StateIndex


# This is marked experimental because it uses the experimental stateful functionality.
class BatchNorm(Module):
    r"""Computes a mean and standard deviation over the batch and spatial
    dimensions of an array, and uses these to normalise the whole array. Optionally
    applies a channelwise affine transformation afterwards.

    Given an input array $x = [x_1, ... x_C]$ with $C$ channels, this layer computes

    $$\frac{x_i - \mathbb{E}[x_i]}{\sqrt{\text{Var}[x_i] + \varepsilon}} * \gamma_i + \beta_i$$

    for all $i$. Here $*$ denotes elementwise multiplication and $\gamma$, $\beta$ have
    shape $(C,)$ if `channelwise_affine=True` and $\gamma = 1$, $\beta = 0$ if
    `channelwise_affine=False`. Expectations are computed over all spatial dimensions
    *and* over the batch dimension, and updated batch-by-batch according to `momentum`.

    !!! example

        ```python
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import jax.random as jr

        key = jr.PRNGKey(0)
        mkey, dkey = jr.split(key)
        model = eqx.nn.Sequential([
            eqx.nn.Linear(in_features=3, out_features=4, key=mkey),
            eqx.experimental.BatchNorm(input_size=4, axis_name="batch"),
        ])

        x = jr.normal(dkey, (10, 3))
        jax.vmap(model, axis_name="batch")(x)
        # BatchNorm will automatically update its running statistics internally.
        ```

    !!! warning

        This layer must be used inside of a `vmap` or `pmap` with a matching
        `axis_name`. (Not doing so will raise a `NameError`.)

    !!! warning

        [`equinox.experimental.BatchNorm`][] updates its running statistics as a side
        effect of its forward pass. Side effects are quite unusual in JAX; as such
        `BatchNorm` is considered experimental. Let us know how you find it!
    """  # noqa: E501

    weight: Optional[Array["input_size"]]
    bias: Optional[Array["input_size"]]
    first_time_index: StateIndex
    state_index: StateIndex
    axis_name: Union[Hashable, Sequence[Hashable]]
    inference: bool
    input_size: int = static_field()
    eps: float = static_field()
    channelwise_affine: bool = static_field()
    momentum: float = static_field()

    def __init__(
        self,
        input_size: int,
        axis_name: str,
        eps: float = 1e-5,
        channelwise_affine: bool = True,
        momentum: float = 0.99,
        inference: bool = False,
        **kwargs,
    ):
        """**Arguments:**

        - `input_size`: The number of channels in the input array.
        - `axis_name`: The name of the batch axis to compute statistics over, as passed
            to `axis_name` in `jax.vmap` or `jax.pmap`. Can also be a sequence (e.g. a
            tuple or a list) of names, to compute statistics over multiple named axes.
        - `eps`: Value added to the denominator for numerical stability.
        - `channelwise_affine`: Whether the module has learnable channel-wise affine
            parameters.
        - `momentum`: The rate at which to update the running statistics. Should be a
            value between 0 and 1 exclusive.
        - `inference`: If `False` then the batch means and variances will be calculated
            and used to update the running statistics. If `True` then the running
            statistics are directly used for normalisation. This may be toggled with
            [`equinox.tree_inference`][] or overridden during
            [`equinox.experimental.BatchNorm.__call__`][].
        """

        super().__init__(**kwargs)

        if channelwise_affine:
            self.weight = jnp.ones((input_size,))
            self.bias = jnp.zeros((input_size,))
        else:
            self.weight = None
            self.bias = None
        self.first_time_index = StateIndex(inference=inference)
        self.state_index = StateIndex(inference=inference)
        self.inference = inference
        self.axis_name = axis_name
        self.input_size = input_size
        self.eps = eps
        self.channelwise_affine = channelwise_affine
        self.momentum = momentum

        set_state(self.first_time_index, jnp.array(True))

    def __call__(
        self,
        x: Array,
        *,
        key: Optional["jax.random.PRNGKey"] = None,
        inference: Optional[bool] = None,
    ) -> Array:
        """**Arguments:**

        - `x`: A JAX array of shape `(input_size, dim_1, ..., dim_N)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        - `inference`: As per [`equinox.experimental.BatchNorm.__init__`][]. If
            `True` or `False` then it will take priority over `self.update_stats`. If
            `None` then the value from `self.update_stats` will be used.

        **Returns:**

        A JAX array of shape `(input_size, dim_1, ..., dim_N)`.

        **Raises:**

        A `NameError` if no `vmap`s are placed around this operation, or if this vmap
        does not have a matching `axis_name`.
        """

        def _stats(y):
            mean = jnp.mean(y)
            mean = lax.pmean(mean, self.axis_name)
            var = jnp.mean((y - mean) ** 2)
            var = lax.pmean(var, self.axis_name)
            return mean, var

        batch_state = jax.vmap(_stats)(x)

        if inference is None:
            inference = self.inference
        if inference:
            running_mean, running_var = get_state(self.state_index, like=batch_state)
        else:
            first_time = get_state(self.first_time_index, like=jnp.array(False))
            running_state = lax.cond(
                first_time,
                lambda: batch_state,
                lambda: get_state(self.state_index, like=batch_state),
            )
            set_state(self.first_time_index, jnp.array(False))
            running_mean, running_var = running_state

            batch_mean, batch_var = batch_state
            running_mean = (
                1 - self.momentum
            ) * batch_mean + self.momentum * running_mean
            running_var = (1 - self.momentum) * batch_var + self.momentum * running_var
            set_state(self.state_index, (running_mean, running_var))

        def _norm(y, m, v, w, b):
            out = (y - m) / jnp.sqrt(v + self.eps)
            if self.channelwise_affine:
                out = out * w + b
            return out

        return jax.vmap(_norm)(x, running_mean, running_var, self.weight, self.bias)
