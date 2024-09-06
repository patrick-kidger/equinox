from typing import Generic, Optional, TypeVar

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from .._eval_shape import filter_eval_shape
from .._module import field
from .._tree import tree_at
from ._sequential import StatefulLayer
from ._stateful import State, StateIndex


def _power_iteration(forward, transpose, v_prev, eps):
    _, tangents_out = jax.jvp(forward, (v_prev,), (v_prev,))
    u_norm = jnp.sqrt(jnp.sum(tangents_out**2))
    u = tangents_out / jnp.maximum(eps, u_norm)
    _, v = jax.jvp(lambda x: transpose(x)[0], (u,), (u,))
    v_norm = jnp.sqrt(jnp.sum(v**2))
    v = v / jnp.maximum(eps, v_norm)

    return u, v


_Layer = TypeVar("_Layer")


class SpectralNorm(StatefulLayer, Generic[_Layer], strict=True):
    """Applies spectral normalisation to a given parameter.

    Given a weight matrix $W$, and letting $σ(W)$ denote (an approximation to) its
    largest singular value, then this computes $W/σ(W)$.

    The approximation $σ(W)$ is computed using
    [power iterations](https://en.wikipedia.org/wiki/Power_iteration)
    which are updated (as a side-effect) every time $W/σ(W)$ is computed.

    Spectral normalisation is particularly commonly used when training generative
    adversarial networks; see
    [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957)
    for more details and motivation.

    Default approaches to spectral normalization rely on inaccurate approximations to the
    spectral norm, although it often perform better; see
    [Why Spectral Normalization Stabilizes GANs: Analysis and Improvements](https://arxiv.org/abs/2009.02773),
    and [Generalizable Adversarial Training via Spectral Normalization](https://arxiv.org/abs/1811.07457).
    Equinox offers functionality for both exact and approximate spectral norms.

    !!! example

        See [this example](../../examples/stateful.ipynb) for example usage.

    Note that this layer behaves differently during training and inference. During
    training then power iterations are updated; during inference they are fixed.
    Whether the model is in training or inference mode should be toggled using
    [`equinox.nn.inference_mode`][].
    """  # noqa: E501

    layer: _Layer
    exact: bool
    weight_name: str = field(static=True)
    uv_index: StateIndex[tuple[Float[Array, " u_size"], Float[Array, " v_size"]]]
    num_power_iterations: int = field(static=True)
    eps: float = field(static=True)
    inference: bool

    def __init__(
        self,
        layer: _Layer,
        weight_name: str,
        num_power_iterations: int = 1,
        eps: float = 1e-12,
        inference: bool = False,
        exact: bool = False,
        input_shape: Optional[jax.ShapeDtypeStruct] = None,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `layer`: The layer to wrap. Usually a [`equinox.nn.Linear`][] or
            a convolutional layer (e.g. [`equinox.nn.Conv2d`][]).
        - `weight_name`: The name of the layer's parameter (a JAX array) to apply
            spectral normalisation to.
        - `num_power_iterations`: The number of power iterations to apply every time
            the array is accessed.
        - `eps`: Epsilon for numerical stability when calculating norms.
        - `inference`: Whether this is in inference mode, at which time no power
            iterations are performed.  This may be toggled with
            [`equinox.nn.inference_mode`][].
        - `exact`: Whether or not to compute the exact linear transpose for power series
            iteration. Traditional approaches rely on reshaping >2D linear operators,
            rather than doing the linear transpose in >2D.
        - `input_shape`: If `exact` is true, the input structure to the layer must be
            specified
        - `key`: A `jax.random.PRNGKey` used to provide randomness for initialisation.
            (Keyword only argument.)


        !!! info

            The `dtype` of the weight array of the `layer` input is applied to all
            parameters in this layer.


        !!! Caution

            If `exact` is true, it computes the transpose via `jax.linear_transpose` of
            the layer. This includes all operations of the layer call, which means for
            layers with a bias, this can result in the incorrect spectral value.

        """
        self.layer = layer
        self.weight_name = weight_name
        self.num_power_iterations = num_power_iterations
        self.eps = eps
        self.inference = inference

        weight = getattr(layer, weight_name)
        ukey, vkey = jr.split(key)

        if not callable(self.layer):
            raise ValueError("`layer` must be callable.")

        if exact:
            if input_shape is None:
                raise ValueError(
                    "Must specify `input_shape` to use exact spectral norm!"
                )
            u_shape = filter_eval_shape(self.layer, input_shape)
            u0 = jr.normal(ukey, u_shape.shape, dtype=u_shape.dtype)
            v0 = jr.normal(vkey, input_shape.shape, dtype=input_shape.dtype)
            reverse = jax.linear_transpose(self.layer, input_shape)
            for _ in range(15):
                u0, v0 = _power_iteration(self.layer, reverse, v0, self.eps)
        else:
            if weight.ndim < 2:
                raise ValueError("`weight` must be at least two-dimensional")
            weight = jnp.reshape(weight, (weight.shape[0], -1))
            dtype = weight.dtype
            u_len, v_len = weight.shape
            u0 = jr.normal(ukey, (u_len,), dtype=dtype)
            v0 = jr.normal(vkey, (v_len,), dtype=dtype)
            for _ in range(15):
                u0, v0 = _power_iteration(
                    lambda y: weight @ y, lambda z: (weight.T @ z,), v0, self.eps
                )
        self.uv_index = StateIndex((u0, v0))
        self.exact = exact

    @jax.named_scope("eqx.nn.SpectralNorm")
    def __call__(
        self,
        x: Array,
        state: State,
        *,
        key: Optional[PRNGKeyArray] = None,
        inference: Optional[bool] = None,
    ) -> tuple[Array, State]:
        """**Arguments:**

        - `x`: A JAX array.
        - `state`: An [`equinox.nn.State`][] object (which is used to store the
            power iterations).
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        - `inference`: As per [`equinox.nn.SpectralNorm.__init__`][]. If
            `True` or `False` then it will take priority over `self.inference`. If
            `None` then the value from `self.inference` will be used.

        **Returns:**

        A 2-tuple of:

        - The JAX array from calling `self.layer(x)` (with spectral normalisation
            applied).
        - An updated context object (storing the updated power iterations).
        """

        u, v = state.get(self.uv_index)
        weight = getattr(self.layer, self.weight_name)
        if self.exact:
            if inference is None:
                inference = self.inference
            if not inference:
                stop_weight = lax.stop_gradient(weight)
                layer = tree_at(
                    lambda l: getattr(l, self.weight_name), self.layer, stop_weight
                )
                reverse = jax.linear_transpose(layer, x)
                for _ in range(self.num_power_iterations):
                    u, v = _power_iteration(layer, reverse, v, self.eps)
                state = state.set(self.uv_index, (u, v))
            else:
                layer = self.layer
            assert callable(layer)  # checked in __init__ but pyright wants it here too
            _, tangents_out = jax.jvp(layer, (v,), (v,))
            σ = jnp.sum(u * tangents_out)
            σ_weight = weight / σ
        else:
            weight_shape = weight.shape
            weight = jnp.reshape(weight, (weight.shape[0], -1))
            if inference is None:
                inference = self.inference
            if not inference:
                stop_weight = lax.stop_gradient(weight)
                for _ in range(self.num_power_iterations):
                    u, v = _power_iteration(
                        lambda y: stop_weight @ y,
                        lambda z: (stop_weight.T @ z,),
                        v,
                        self.eps,
                    )
                state = state.set(self.uv_index, (u, v))
            σ = jnp.einsum("i,ij,j->", u, weight, v)
            σ_weight = jnp.reshape(weight / σ, weight_shape)
        layer = tree_at(lambda l: getattr(l, self.weight_name), self.layer, σ_weight)
        out = layer(x)
        return out, state
