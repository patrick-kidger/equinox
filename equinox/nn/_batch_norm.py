import warnings
from collections.abc import Hashable, Sequence
from typing import Literal, Optional, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .._misc import default_floating_dtype
from .._module import field
from ._sequential import StatefulLayer
from ._stateful import State, StateIndex


class BatchNorm(StatefulLayer, strict=True):
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

        See [this example](../../examples/stateful.ipynb) for example usage.

    !!! warning

        This layer must be used inside of a `vmap` or `pmap` with a matching
        `axis_name`. (Not doing so will raise a `NameError`.)

    Note that this layer behaves differently during training and inference. During
    training then statistics are computed using the input data, and the running
    statistics updated. During inference then just the running statistics are used.
    Whether the model is in training or inference mode should be toggled using
    [`equinox.nn.inference_mode`][].

    With `approach = "batch"` during training the batch mean and variance are used
    for normalization.  For inference the exponential running mean and ubiased
    variance are used for normalization in accordance with the cited paper below:

    $\text{TrainStats}_t = \text{BatchStats}_t$

    $\text{InferenceStats}_t = \frac{\left(1.0 - m\right)\sum_{i=0}^{t}m^{t-i}
    \text{BatchStats}_i}{\text{max} \left(1.0 - m^{t+1}, \varepsilon \right)}$

    With `approach = "ema"` exponential running means and variances are kept.  During
    training the batch statistics are used to fill in the running statistics until
    they are populated.  In addition a linear iterpolation is used between the batch
    and running statistics over the `warmup_period`.  During inference the running
    statistics are used for normalization:



    $\text{WarmupFrac}_t = \text{min} \left(1.0, \frac{t}{\text{WarmupPeriod}} \right)$

    $\text{TrainStats}_t = (1.0 - \text{WarmupFrac}_t) * BatchStats_t +
    \text{WarmupFrac}_t * \left(1.0 - m\right)\sum_{i=0}^{t}m^{t-i}\text{BatchStats}_i$

    $\text{InferenceStats}_t = \frac{\left(1.0 - m\right)\sum_{i=0}^{t}m^{t-i}
    \text{BatchStats}_i}{\text{Max} \left(1.0 - m^{t+1}, \varepsilon \right)}$


    $\text{Note: } \frac{(1.0 - m)\sum_{i=0}^{t}m^{t-i}}{1.0 - m^{t+1}} =
    \frac{(1.0 - m)\sum_{i=0}^{t}m^{i}}{1.0 - m^{t+1}}$
    $= \frac{(1.0 - m)\frac{1.0 - m^{t+1}}{1.0 - m}}{1.0 - m^{t+1}} = 1$


    ??? cite

        [Batch Normalization: Accelerating Deep Network Training by Reducing
         Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

        ```bibtex
        @article{DBLP:journals/corr/IoffeS15,
        author       = {Sergey Ioffe and
                        Christian Szegedy},
        title        = {Batch Normalization: Accelerating Deep Network Training
                        by Reducing Internal Covariate Shift},
        journal      = {CoRR},
        volume       = {abs/1502.03167},
        year         = {2015},
        url          = {http://arxiv.org/abs/1502.03167},
        eprinttype    = {arXiv},
        eprint       = {1502.03167},
        timestamp    = {Mon, 13 Aug 2018 16:47:06 +0200},
        biburl       = {https://dblp.org/rec/journals/corr/IoffeS15.bib},
        bibsource    = {dblp computer science bibliography, https://dblp.org}
        }
        ```

    """  # noqa: E501

    weight: Optional[Float[Array, "input_size"]]
    bias: Optional[Float[Array, "input_size"]]
    count_index: StateIndex[Int[Array, ""]]
    state_index: StateIndex[
        tuple[Float[Array, "input_size"], Float[Array, "input_size"]]
    ]
    zero_frac_index: StateIndex[Float[Array, ""]]
    axis_name: Union[Hashable, Sequence[Hashable]]
    inference: bool
    input_size: int = field(static=True)
    approach: Literal["batch", "ema"] = field(static=True)
    eps: float = field(static=True)
    channelwise_affine: bool = field(static=True)
    momentum: float = field(static=True)
    warmup_period: int = field(static=True)

    def __init__(
        self,
        input_size: int,
        axis_name: Union[Hashable, Sequence[Hashable]],
        approach: Optional[Literal["batch", "ema"]] = None,
        eps: float = 1e-5,
        channelwise_affine: bool = True,
        momentum: float = 0.99,
        warmup_period: int = 1,
        inference: bool = False,
        dtype=None,
    ):
        """**Arguments:**

        - `input_size`: The number of channels in the input array.
        - `axis_name`: The name of the batch axis to compute statistics over, as passed
            to `axis_name` in `jax.vmap` or `jax.pmap`. Can also be a sequence (e.g. a
            tuple or a list) of names, to compute statistics over multiple named axes.
        - `approach`: The approach to use for the running statistics. If `approach=None`
            a warning will be raised and approach will default to `"batch"`. During
            training `"batch"` only uses batch statisics while`"ema"` uses the running
            statistics.
        - `eps`: Value added to the denominator for numerical stability.
        - `channelwise_affine`: Whether the module has learnable channel-wise affine
            parameters.
        - `momentum`: The rate at which to update the running statistics. Should be a
            value between 0 and 1 exclusive.
        - `warmup_period`: The interpolation period between batch and running
            statistics. Only used when `approach=\"ema\"`.
        - `inference`: If `False` then the batch means and variances will be calculated
            and used to update the running statistics. If `True` then the running
            statistics are directly used for normalisation. This may be toggled with
            [`equinox.nn.inference_mode`][] or overridden during
            [`equinox.nn.BatchNorm.__call__`][].
        - `dtype`: The dtype to use for the running statistics. Defaults to either
            `jax.numpy.float32` or `jax.numpy.float64` depending on whether JAX is in
            64-bit mode.
        """

        if approach is None:
            warnings.warn('BatchNorm approach is None, defaults to approach="batch"')
            approach = "batch"

        valid_approaches = {"batch", "ema"}
        if approach not in valid_approaches:
            raise ValueError(f"approach must be one of {valid_approaches}")
        self.approach = approach

        if warmup_period < 1:
            raise ValueError("warmup_period must be >= 1")

        if channelwise_affine:
            self.weight = jnp.ones((input_size,))
            self.bias = jnp.zeros((input_size,))
        else:
            self.weight = None
            self.bias = None
        self.count_index = StateIndex(jnp.array(0, dtype=jnp.int32))
        if dtype is None:
            dtype = default_floating_dtype()
        init_buffers = (
            jnp.zeros((input_size,), dtype=dtype),
            jnp.zeros((input_size,), dtype=dtype),
        )
        self.state_index = StateIndex(init_buffers)
        self.zero_frac_index = StateIndex(jnp.array(1.0, dtype=dtype))
        self.inference = inference
        self.axis_name = axis_name
        self.input_size = input_size
        self.eps = eps
        self.channelwise_affine = channelwise_affine
        self.momentum = momentum
        self.warmup_period = warmup_period

    @jax.named_scope("eqx.nn.BatchNorm")
    def __call__(
        self,
        x: Array,
        state: State,
        *,
        key: Optional[PRNGKeyArray] = None,
        inference: Optional[bool] = None,
    ) -> tuple[Array, State]:
        """**Arguments:**

        - `x`: A JAX array of shape `(input_size, dim_1, ..., dim_N)`.
        - `state`: An [`equinox.nn.State`][] object (which is used to store the
            running statistics).
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        - `inference`: As per [`equinox.nn.BatchNorm.__init__`][]. If
            `True` or `False` then it will take priority over `self.inference`. If
            `None` then the value from `self.inference` will be used.

        **Returns:**

        A 2-tuple of:

        - A JAX array of shape `(input_size, dim_1, ..., dim_N)`.
        - An updated state object (storing the updated running statistics).

        **Raises:**

        A `NameError` if no `vmap`s are placed around this operation, or if this vmap
        does not have a matching `axis_name`.
        """

        if inference is None:
            inference = self.inference
        if inference:
            # renormalize running stats to account for the zeroed part
            zero_frac = state.get(self.zero_frac_index)
            running_mean, running_var = state.get(self.state_index)
            norm_mean = running_mean / jnp.maximum(1.0 - zero_frac, self.eps)
            norm_var = running_var / jnp.maximum(1.0 - zero_frac, self.eps)
        else:

            def _stats(y):
                mean = jnp.mean(y)
                mean = lax.pmean(mean, self.axis_name)
                var = jnp.mean((y - mean) * jnp.conj(y - mean))
                var = lax.pmean(var, self.axis_name)
                var = jnp.maximum(0.0, var)
                return mean, var

            momentum = self.momentum
            zero_frac = state.get(self.zero_frac_index)
            zero_frac = zero_frac * momentum
            state = state.set(self.zero_frac_index, zero_frac)

            batch_mean, batch_var = jax.vmap(_stats)(x)
            running_mean, running_var = state.get(self.state_index)
            running_mean = (1 - momentum) * batch_mean + momentum * running_mean

            if self.approach == "ema":
                running_var = (1 - momentum) * batch_var + momentum * running_var
                warmup_count = state.get(self.count_index)
                warmup_count = jnp.minimum(warmup_count + 1, self.warmup_period)
                state = state.set(self.count_index, warmup_count)

                # fill in unpopulated part of running stats with batch stats
                warmup_frac = warmup_count / self.warmup_period
                norm_mean = zero_frac * batch_mean + running_mean
                norm_var = zero_frac * batch_var + running_var

                # apply warmup interpolation between batch and running statistics
                norm_mean = (1.0 - warmup_frac) * batch_mean + warmup_frac * norm_mean
                norm_var = (1.0 - warmup_frac) * batch_var + warmup_frac * norm_var
            else:
                # calculate unbiased variance for saving
                axis_size = jax.lax.psum(jnp.array(1.0), self.axis_name)
                debias_coef = (axis_size) / jnp.maximum(axis_size - 1, self.eps)
                running_var = (
                    1 - momentum
                ) * debias_coef * batch_var + momentum * running_var

                # just use batch statistics when not in inference mode
                norm_mean, norm_var = batch_mean, batch_var

            state = state.set(self.state_index, (running_mean, running_var))

        def _norm(y, m, v, w, b):
            out = (y - m) / jnp.sqrt(v + self.eps)
            if self.channelwise_affine:
                out = out * w + b
            return out

        out = jax.vmap(_norm)(x, norm_mean, norm_var, self.weight, self.bias)
        return out, state
