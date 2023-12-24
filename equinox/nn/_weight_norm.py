from typing import Generic, Optional, TypeVar, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from .._module import field, Module
from .._tree import tree_at


_Layer = TypeVar("_Layer")


def _norm_except_axis(
    v: Array, pow: Optional[Union[int, str]] = None, axis: Optional[int] = 0
) -> Array:
    norm_func = lambda x: jnp.linalg.norm(x, ord=pow, keepdims=True)
    vmapped_norm_func = lambda axis: jax.vmap(
        norm_func,
        in_axes=axis,
        out_axes=axis,
    )(v)
    if axis is not None:
        return vmapped_norm_func(axis)
    else:
        return norm_func(vmapped_norm_func(0)).reshape([])


class WeightNorm(Module, Generic[_Layer]):
    r"""
    Applies weight normalisation to a given parameter.

    Given a weight matrix $\mathbf{W}$, computes the follow reparametrization:

    $\mathbf{W} = g \frac{\mathbf{v}}{\lVert \mathbf{v} \rVert}$

    where $g$ is initially chosen to equal $\lVert \mathbf{v} \rVert$
    , and $\mathbf{v}$ is initially chosen as $\mathbf{W}$ .

    ??? cite
        [Weight Normalisation](https://arxiv.org/abs/1602.07868)

        ```bibtex
        @article{DBLP:journals/corr/SalimansK16,
        author       = {Tim Salimans and
                        Diederik P. Kingma},
        title        = {Weight Normalisation: {A} Simple
                        Reparameterization to Accelerate
                        Training of Deep Neural Networks},
        journal      = {CoRR},
        volume       = {abs/1602.07868},
        year         = {2016},
        url          = {http://arxiv.org/abs/1602.07868},
        eprinttype   = {arXiv},
        eprint       = {1602.07868},
        timestamp    = {Mon, 13 Aug 2018 16:47:07 +0200},
        biburl       = {https://dblp.org/rec/journals/corr/SalimansK16.bib},
        bibsource    = {dblp computer science bibliography, https://dblp.org}
        }
        ```


    """

    layer: _Layer
    v: Array
    g: Array
    weight_name: str = field(static=True)
    axis: Optional[int] = field(static=True)

    def __init__(
        self,
        layer: _Layer,
        weight_name: str = "weight",
        axis: Optional[int] = 0,
    ):
        """**Arguments:**

        - `layer`: The layer to wrap. Usually a [`equinox.nn.Linear`][] or
        a convolutional layer (e.g. [`equinox.nn.Conv2d`][]).
        - `weight_name`: The name of the layer's parameter (a JAX array) to apply
        weight normalisation to.
        - `axis`: The norm is computed across every axis except this one.
        If `None`, compute across every axis.
        """
        self.layer = layer
        self.weight_name = weight_name
        self.axis = axis

        self.v = getattr(layer, weight_name)
        self.g = _norm_except_axis(self.v, axis=axis)

    @jax.named_scope("eqx.nn.WeightNorm")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """**Arguments:**

        - `x`: A JAX Array.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.

        **Returns:**

        - The JAX array from calling `self.layer(x)` (with weight normalisation
        applied).
        """
        weight = self.v * self.g / _norm_except_axis(self.v, axis=self.axis)
        layer = tree_at(lambda l: getattr(l, self.weight_name), self.layer, weight)
        return layer(x)
