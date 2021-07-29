import jax.random as jrandom
import math
from typing import Optional

from ..custom_types import Array
from ..module import Module


class Linear(Module):
    weight: Array["out_features", "in_features"]  # noqa: F821
    bias: Optional[Array["out_features"]]  # noqa: F821

    def __init__(self, in_features, out_features, use_bias=True, *, key):
        super().__init__()
        wkey, bkey = jrandom.split(key, 2)
        lim = 1 / math.sqrt(in_features)
        self.weight = jrandom.uniform(wkey, (out_features, in_features), minval=-lim, maxval=lim)
        if use_bias:
            self.bias = jrandom.uniform(bkey, (out_features,), minval=-lim, maxval=lim)
        else:
            self.bias = None

    def __call__(self, x, *, key=None):
        x = self.weight @ x
        if self.bias is not None:
            x = x + self.bias
        return x


class Identity(Module):
    def __init__(self, *args, **kwargs):
        # Ignores args and kwargs
        super().__init__()

    def __call__(self, x, *, key=None):
        return x
