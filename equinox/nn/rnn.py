import math
from typing import Optional

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from ..custom_types import Array
from ..module import Module


class GRUCell(Module):
    weight_ih: Array
    weight_hh: Array
    bias: Optional[Array]
    bias_n: Optional[Array]

    def __init__(self, input_size, hidden_size, bias=True, *, key, **kwargs):
        super().__init__(**kwargs)

        ihkey, hhkey, bkey, bkey2 = jrandom.split(key, 4)
        lim = math.sqrt(1 / hidden_size)

        self.weight_ih = jrandom.uniform(
            ihkey, (3 * hidden_size, input_size), minval=-lim, maxval=lim
        )
        self.weight_hh = jrandom.uniform(
            hhkey, (3 * hidden_size, hidden_size), minval=-lim, maxval=lim
        )
        if bias:
            self.bias = jrandom.uniform(
                bkey, (3 * hidden_size,), minval=-lim, maxval=lim
            )
            self.bias_n = jrandom.uniform(
                bkey2, (hidden_size,), minval=-lim, maxval=lim
            )
        else:
            self.bias = None
            self.bias_n = None

    def __call__(self, input, hidden, *, key=None):
        if self.bias is None:
            bias = 0
            bias_n = 0
        else:
            bias = self.bias
            bias_n = self.bias_n
        igates = jnp.split(self.weight_ih @ input + bias, 3)
        hgates = jnp.split(self.weight_hh @ hidden, 3)
        reset = jnn.sigmoid(igates[0] + hgates[0])
        inp = jnn.sigmoid(igates[1] + hgates[1])
        new = jnn.tanh(igates[2] + reset * (hgates[2] + bias_n))
        return new + inp * (hidden - new)


class LSTMCell(Module):
    weight_ih: Array
    weight_hh: Array
    bias: Optional[Array]

    def __init__(self, input_size, hidden_size, bias=True, *, key, **kwargs):
        super().__init__(**kwargs)

        ihkey, hhkey, bkey = jrandom.split(key, 3)
        lim = math.sqrt(1 / hidden_size)

        self.weight_ih = jrandom.uniform(
            ihkey, (4 * hidden_size, input_size), minval=-lim, maxval=lim
        )
        self.weight_hh = jrandom.uniform(
            hhkey, (4 * hidden_size, hidden_size), minval=-lim, maxval=lim
        )
        if bias is None:
            self.bias = jrandom.uniform(
                bkey, (4 * hidden_size,), minval=-lim, maxval=lim
            )
        else:
            self.bias = None

    def __call__(self, input, hidden, *, key=None):
        h, c = hidden
        lin = self.weight_ih @ input + self.weight_hh @ h
        if self.bias is not None:
            lin = lin + self.bias
        i, f, g, o = jnp.split(lin, 4)
        i = jnn.sigmoid(i)
        f = jnn.sigmoid(f)
        g = jnn.tanh(g)
        o = jnn.sigmoid(o)
        c = f * c + i * g
        h = o * jnn.tanh(c)
        return (h, c)
