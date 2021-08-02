from typing import List

import jax.nn as jnn
import jax.random as jrandom

from ..module import Module
from .linear import Linear


class MLP(Module):
    layers: List[Linear]
    activation: callable
    final_activation: callable

    def __init__(
        self,
        in_size,
        out_size,
        width_size,
        depth,
        activation=jnn.relu,
        final_activation=lambda x: x,
        *,
        key,
        **kwargs
    ):
        super().__init__(**kwargs)
        keys = jrandom.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(Linear(in_size, out_size, key=keys[0]))
        else:
            layers.append(Linear(in_size, width_size, key=keys[0]))
            for i in range(depth - 1):
                layers.append(Linear(width_size, width_size, key=keys[i + 1]))
            layers.append(Linear(width_size, out_size, key=keys[-1]))
        self.layers = layers
        self.activation = activation
        self.final_activation = final_activation

    def __call__(self, x, *, key=None):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        x = self.final_activation(x)
        return x


class Sequential(Module):
    layers: List[Module]

    def __call__(self, x, *, key=None):
        if key is None:
            keys = [None] * len(self.layers)
        else:
            keys = jrandom.split(key, len(self.layers))
        for layer, key in zip(self.layers, keys):
            x = layer(x, key=key)
        return x
