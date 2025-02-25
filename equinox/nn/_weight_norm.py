import functools as ft
from collections.abc import Callable
from typing import Generic, Optional, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, Scalar

from .._module import field, Module
from .._tree import tree_at
from ._misc import named_scope


_Layer = TypeVar("_Layer")


def _norm_except_axis(v: Array, norm: Callable[[Array], Scalar], axis: Optional[int]):
    if axis is None:
        return norm(v)
    else:
        return jax.vmap(norm, in_axes=axis, out_axes=axis)(v)


class WeightNorm(Module, Generic[_Layer], strict=True):
    r"""Applies weight normalisation to a given parameter.

    Given the 2D weight matrix
    $W = (W_{ij})_{ij} \in \mathbb{R}^{\text{out}} \times \mathbb{R}^{\text{in}}$ of a
    linear layer, then it replaces it with the following reparameterisation:

    $g \frac{v_{ij}}{\lVert v_{i\, \cdot} \rVert} \in \mathbb{R}^{\text{out}} \times \mathbb{R}^{\text{in}}$

    where $v_{ij}$ is initialised as $W_{ij}$, and $g$ is initialised as
    $\lVert v_{i\, \cdot} \rVert = \sum_j {v_{ij}}^2$.

    Overall, the direction ($v$) and the magnitude ($g$) of the output of each neuron
    are treated separately.

    Given n-dimensional weight matrices $W$ (in convolutional layers), then the
    normalisation is analogusly instead computed over every axis except the first.

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
    """  # noqa: E501

    layer: _Layer
    g: Array
    weight_name: str = field(static=True)
    axis: Optional[int] = field(static=True)
    _norm: Callable[[Array], Scalar]

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

        self._norm = ft.partial(
            _norm_except_axis,
            norm=ft.partial(jnp.linalg.norm, keepdims=True),
            axis=axis,
        )
        self.g = self._norm(getattr(layer, weight_name))

    @named_scope("eqx.nn.WeightNorm")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """**Arguments:**

        - `x`: A JAX Array.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.

        **Returns:**

        - The JAX array from calling `self.layer(x)` (with weight normalisation
        applied).
        """

        v = getattr(self.layer, self.weight_name)
        weight = v * self.g / self._norm(v)
        layer = tree_at(lambda l: getattr(l, self.weight_name), self.layer, weight)
        return layer(x)
