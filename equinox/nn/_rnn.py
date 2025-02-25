import math
from typing import Optional

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from .._misc import default_floating_dtype
from .._module import field, Module
from ._misc import default_init, named_scope


class GRUCell(Module, strict=True):
    """A single step of a Gated Recurrent Unit (GRU).

    !!! example

        This is often used by wrapping it into a `jax.lax.scan`. For example:

        ```python
        class Model(Module):
            cell: GRUCell

            def __init__(self, **kwargs):
                self.cell = GRUCell(**kwargs)

            def __call__(self, xs):
                scan_fn = lambda state, input: (self.cell(input, state), None)
                init_state = jnp.zeros(self.cell.hidden_size)
                final_state, _ = jax.lax.scan(scan_fn, init_state, xs)
                return final_state
        ```
    """

    weight_ih: Array
    weight_hh: Array
    bias: Optional[Array]
    bias_n: Optional[Array]
    input_size: int = field(static=True)
    hidden_size: int = field(static=True)
    use_bias: bool = field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `input_size`: The dimensionality of the input vector at each time step.
        - `hidden_size`: The dimensionality of the hidden state passed along between
            time steps.
        - `use_bias`: Whether to add on a bias after each update.
        - `dtype`: The dtype to use for all weights and biases in this GRU cell.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending on
            whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        ihkey, hhkey, bkey, bkey2 = jrandom.split(key, 4)
        lim = math.sqrt(1 / hidden_size)

        ihshape = (3 * hidden_size, input_size)
        self.weight_ih = default_init(ihkey, ihshape, dtype, lim)
        hhshape = (3 * hidden_size, hidden_size)
        self.weight_hh = default_init(hhkey, hhshape, dtype, lim)
        if use_bias:
            self.bias = default_init(bkey, (3 * hidden_size,), dtype, lim)
            self.bias_n = default_init(bkey2, (hidden_size,), dtype, lim)
        else:
            self.bias = None
            self.bias_n = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias

    @named_scope("eqx.nn.GRUCell")
    def __call__(
        self, input: Array, hidden: Array, *, key: Optional[PRNGKeyArray] = None
    ):
        """**Arguments:**

        - `input`: The input, which should be a JAX array of shape `(input_size,)`.
        - `hidden`: The hidden state, which should be a JAX array of shape
            `(hidden_size,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        The updated hidden state, which is a JAX array of shape `(hidden_size,)`.
        """
        if self.use_bias:
            bias = self.bias
            bias_n = self.bias_n
        else:
            bias = 0
            bias_n = 0
        igates = jnp.split(self.weight_ih @ input + bias, 3)
        hgates = jnp.split(self.weight_hh @ hidden, 3)
        reset = jnn.sigmoid(igates[0] + hgates[0])
        inp = jnn.sigmoid(igates[1] + hgates[1])
        new = jnn.tanh(igates[2] + reset * (hgates[2] + bias_n))
        return new + inp * (hidden - new)


class LSTMCell(Module, strict=True):
    """A single step of a Long-Short Term Memory unit (LSTM).

    !!! example

        This is often used by wrapping it into a `jax.lax.scan`. For example:

        ```python
        class Model(Module):
            cell: LSTMCell

            def __init__(self, ...):
                self.cell = LSTMCell(...)

            def __call__(self, xs):
                scan_fn = lambda state, input: (self.cell(input, state), None)
                init_state = (jnp.zeros(self.cell.hidden_size),
                              jnp.zeros(self.cell.hidden_size))
                (h, c), _ = jax.lax.scan(scan_fn, init_state, xs)
                return h, c
        ```
    """

    weight_ih: Array
    weight_hh: Array
    bias: Optional[Array]
    input_size: int = field(static=True)
    hidden_size: int = field(static=True)
    use_bias: bool = field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `input_size`: The dimensionality of the input vector at each time step.
        - `hidden_size`: The dimensionality of the hidden state passed along between
            time steps.
        - `use_bias`: Whether to add on a bias after each update.
        - `dtype`: The dtype to use for all weights and biases in this LSTM cell.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending on
            whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        ihkey, hhkey, bkey = jrandom.split(key, 3)
        lim = math.sqrt(1 / hidden_size)

        ihshape = (4 * hidden_size, input_size)
        self.weight_ih = default_init(ihkey, ihshape, dtype, lim)
        hhshape = (4 * hidden_size, hidden_size)
        self.weight_hh = default_init(hhkey, hhshape, dtype, lim)
        bshape = (4 * hidden_size,)
        self.bias = default_init(bkey, bshape, dtype, lim) if use_bias else None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias

    @named_scope("eqx.nn.LSTMCell")
    def __call__(self, input, hidden, *, key=None):
        """**Arguments:**

        - `input`: The input, which should be a JAX array of shape `(input_size,)`.
        - `hidden`: The hidden state, which should be a 2-tuple of JAX arrays, each of
            shape `(hidden_size,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        The updated hidden state, which is a 2-tuple of JAX arrays, each of shape
        `(hidden_size,)`.
        """
        h, c = hidden
        lin = self.weight_ih @ input + self.weight_hh @ h
        if self.use_bias:
            lin = lin + self.bias
        i, f, g, o = jnp.split(lin, 4)
        i = jnn.sigmoid(i)
        f = jnn.sigmoid(f)
        g = jnn.tanh(g)
        o = jnn.sigmoid(o)
        c = f * c + i * g
        h = o * jnn.tanh(c)
        return (h, c)
