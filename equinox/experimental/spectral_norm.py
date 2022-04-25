from typing import Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr

from ..custom_types import Array
from ..module import Module, static_field
from .stateful import get_state, set_state, StateIndex


def _power_iteration(weight, u, v, eps):
    u = weight @ v
    u_norm = jnp.sqrt(jnp.sum(u**2))
    u = u / jnp.maximum(eps, u_norm)

    v = weight.T @ u
    v_norm = jnp.sqrt(jnp.sum(v**2))
    v = v / jnp.maximum(eps, v_norm)

    return u, v


class SpectralNorm(Module):
    """Applies spectral normalisation to a given parameter.

    Given a weight matrix $W$, and letting $σ(W)$ denote (an approximation to) its
    largest singular value, then this computes $W/σ(W)$.

    The approximation $σ(W)$ is computed using [power iterations](https://en.wikipedia.org/wiki/Power_iteration)
    which are updated (as a side-effect) every time $W/σ(W)$ is computed.

    Spectral normalisation is particularly commonly used when training generative
    adversarial networks; see [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957)
    for more details and motivation.

    [`equinox.experimental.SpectralNorm`][] should be used to replace an individual
    parameter. (Unlike some libraries, not the layer containing that parameter.)

    !!! example

        To add spectral normalisation during model creation:
        ```python
        import equinox as eqx
        import equinox.experimental as eqxe
        import jax.random as jr

        key = jr.PRNGKey(0)
        linear = eqx.nn.Linear(2, 2, key=key)
        sn_weight = eqxe.SpectralNorm(linear.weight, key=key)
        linear = eqx.tree_at(lambda l: l.weight, linear, sn_weight)
        ```

    !!! example

        Alternatively, iterate over the model to add spectral normalisation after model
        creation:
        ```python
        import equinox as eqx
        import equinox.experimental as eqxe
        import jax
        import jax.random as jr
        import functools as ft

        key = jr.PRNGKey(0)
        model_key, spectral_key = jr.split(key)
        SN = ft.partial(eqxe.SpectralNorm, key=spectral_key)

        def _is_linear(leaf):
            return isinstance(leaf, eqx.nn.Linear)

        def _apply_sn_to_linear(module):
            if _is_linear(module):
                module = eqx.tree_at(lambda m: m.weight, module, replace_fn=SN)
            return module

        def apply_sn(model):
            return jax.tree_map(_apply_sn_to_linear, model, is_leaf=_is_linear)

        model = eqx.nn.MLP(2, 2, 2, 2, key=model_key)
        model_with_sn = apply_sn(model)
        ```

    !!! example

        Switching the model to inference mode after training:
        ```python
        import equinox as eqx
        import equinox.experimental as eqxe
        import jax

        def _is_sn(leaf):
            return isinstance(leaf, eqxe.SpectralNorm)

        def _set_inference_on_sn(module):
            if _is_sn(module):
                module = eqx.tree_at(lambda m: m.inference, module, True)
            return module

        def set_inference(model):
            return jax.tree_map(_set_inference_on_sn, model, is_leaf=_is_sn)

        model = ...  # set up model, train it, etc.
        model = set_inference(model)
        ```

    !!! warning

        [`equinox.experimental.SpectralNorm`][] updates its running statistics as a side
        effect of its forward pass. Side effects are quite unusual in JAX; as such
        `SpectralNorm` is considered experimental. Let us know how you find it!
    """  # noqa: E501

    weight_shape: Tuple[int, ...] = static_field()
    weight: Array
    uv_index: StateIndex
    num_power_iterations: int = static_field()
    eps: float = static_field()
    inference: bool

    def __init__(
        self,
        weight: Array,
        num_power_iterations: int = 1,
        eps: float = 1e-12,
        inference: bool = False,
        *,
        key: "jax.random.PRNGKey",
        **kwargs
    ):
        """**Arguments:**

        - `weight`: The parameter (a JAX array) to apply spectral normalisation to.
        - `num_power_iterations`: The number of power iterations to apply every time the array is accessed.
        - `eps`: Epsilon for numerical stability when calculating norms.
        - `inference`: Whether this is in inference mode, at which time no power iterations are performed.
            This may be toggled with [`equinox.tree_inference`][].
        - `key`: A `jax.random.PRNGKey` used to provide randomness for initialisation. (Keyword only argument.)
        """
        super().__init__(**kwargs)
        if weight.ndim < 2:
            raise ValueError("`weight` must be at least two-dimensional")

        self.weight_shape = weight.shape
        weight = jnp.reshape(weight, (weight.shape[0], -1))
        self.weight = weight
        self.uv_index = StateIndex(inference=inference)
        self.num_power_iterations = num_power_iterations
        self.eps = eps
        self.inference = inference

        u_len, v_len = weight.shape
        ukey, vkey = jr.split(key)
        u0 = jr.normal(ukey, (u_len,))
        v0 = jr.normal(vkey, (v_len,))
        for _ in range(15):
            u0, v0 = _power_iteration(weight, u0, v0, eps)
        set_state(self.uv_index, (u0, v0))

    def __jax_array__(self):
        u_like = self.weight[:, -1]
        v_like = self.weight[-1]
        u, v = get_state(self.uv_index, (u_like, v_like))
        if not self.inference:
            for _ in range(self.num_power_iterations):
                u, v = _power_iteration(self.weight, u, v, self.eps)
            set_state(self.uv_index, (u, v))
        u = lax.stop_gradient(u)
        v = lax.stop_gradient(v)
        σ = jnp.einsum("i,ij,j->", u, self.weight, v)
        return jnp.reshape(self.weight / σ, self.weight_shape)

    @classmethod
    def withkey(cls, key: "jax.random.PRNGKey"):
        return lambda *args, **kwargs: cls(*args, key=key, **kwargs)
