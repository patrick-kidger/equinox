import jax.random as jrandom
from typing import List

from .activations import ReLU
from .linear import Identity, Linear
from .module import Module


class MLP(Module):
    layers: List[Linear]
    activation: Module
    final_activation: Module

    def __init__(
        self, in_size, out_size, width_size, depth, activation=ReLU(), final_activation=Identity(), *, key, **kwargs
    ):
        super().__init__(**kwargs)
        keys = jrandom.split(key, depth + 1)
        layers = []
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
    layers: List

    def __call__(self, x, *, key=None):
        keys = jrandom.split(key, len(self.layers))
        for layer, key in zip(self.layers, keys):
            x = layer(x, key=key)
        return x
