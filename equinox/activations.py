import jax.nn as jnn

from .module import Module


def wrap_activation(fn):
    class WrappedFn(Module):
        def forward(self, x):
            return x

    return WrappedFn


ReLU = wrap_activation(jnn.relu)
