from collections.abc import Callable
from typing import (
    Literal,
)

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jaxtyping import Array, Int, PRNGKeyArray

from .._doc_utils import doc_repr
from .._filters import is_array
from .._misc import default_floating_dtype
from .._module import field, Module
from .._vmap_pmap import filter_vmap
from ._linear import Linear
from ._misc import named_scope


_identity = doc_repr(lambda x: x, "lambda x: x")
_relu = doc_repr(jnn.relu, "<function relu>")

_vmap_a_of_b = filter_vmap(lambda a, b: a(b))


class MLP(Module):
    """Standard Multi-Layer Perceptron; also known as a feed-forward network.

    !!! faq

        If you get a TypeError saying an object is not a valid JAX type, see the
            [FAQ](https://docs.kidger.site/equinox/faq/)."""

    layers: tuple[Linear, ...]
    activation: Callable
    final_activation: Callable
    use_bias: bool = field(static=True)
    use_final_bias: bool = field(static=True)
    in_size: int | Literal["scalar"] = field(static=True)
    out_size: int | Literal["scalar"] = field(static=True)
    width_size: int = field(static=True)
    depth: int = field(static=True)
    scan: bool = field(static=True, default=False)

    def __init__(
        self,
        in_size: int | Literal["scalar"],
        out_size: int | Literal["scalar"],
        width_size: int,
        depth: int,
        activation: Callable = _relu,
        final_activation: Callable = _identity,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype=None,
        *,
        scan: bool = False,
        key: PRNGKeyArray,
    ):
        """**Arguments**:

        - `in_size`: The input size. The input to the module should be a vector of
            shape `(in_features,)`
        - `out_size`: The output size. The output from the module will be a vector
            of shape `(out_features,)`.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers, including the output layer.
            For example:
            - `depth=0` is just a linear layer
                `[Linear(in_size, out_size)]`;
            - `depth=1` results in a network with layers
                `[Linear(in_size, width_size), Linear(width_size, out_size)]`;
            - `depth=2` results in a network with layers
                `[Linear(in_size, width_size), Linear(width_size, width_size),
                Linear(width_size, out_size)]`.
        - `activation`: The activation function after each hidden layer. Defaults to
            ReLU.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `use_bias`: Whether to add on a bias to internal layers. Defaults
            to `True`.
        - `use_final_bias`: Whether to add on a bias to the final layer. Defaults
            to `True`.
        - `dtype`: The dtype to use for all the weights and biases in this MLP.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `scan`: Whether to use a scan-over-layers pattern. Defaults to `False` for
            backward compatibility, but setting to `True` will improve compilation
            speed.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        Note that `in_size` also supports the string `"scalar"` as a special value.
        In this case the input to the module should be of shape `()`.

        Likewise `out_size` can also be a string `"scalar"`, in which case the
        output from the module will have shape `()`.
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        keys = jrandom.split(key, depth + 1)

        layers = []

        if depth == 0:
            layers.append(
                Linear(
                    in_size, out_size, use_bias=use_final_bias, dtype=dtype, key=keys[0]
                )
            )
        else:
            # Input layer
            layers.append(
                Linear(in_size, width_size, use_bias, dtype=dtype, key=keys[0])
            )
            # Hidden layers: create `depth - 1` many identical layers.
            make_hidden_layer = lambda k: Linear(
                width_size, width_size, use_bias=use_bias, dtype=dtype, key=k
            )
            if scan:
                layers.append(jax.vmap(make_hidden_layer)(keys[1:depth]))
            else:
                layers.extend(make_hidden_layer(keys[i]) for i in range(1, depth))
            # Output layer
            layers.append(
                Linear(width_size, out_size, use_final_bias, dtype=dtype, key=keys[-1])
            )

        self.layers = tuple(layers)
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        self.scan = scan
        # In case `activation` or `final_activation` are learnt, then make a separate
        # copy of their weights for every neuron.
        self.activation = filter_vmap(
            filter_vmap(lambda: activation, axis_size=width_size), axis_size=depth
        )()
        if out_size == "scalar":
            self.final_activation = final_activation
        else:
            self.final_activation = filter_vmap(
                lambda: final_activation, axis_size=out_size
            )()
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

    @named_scope("eqx.nn.MLP")
    def __call__(self, x: Array, *, key: PRNGKeyArray | None = None) -> Array:
        """**Arguments:**

        - `x`: A JAX array with shape `(in_size,)`. (Or shape `()` if
            `in_size="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(out_size,)`. (Or shape `()` if `out_size="scalar"`.)
        """
        del key

        if self.depth == 0:
            [layer] = self.layers
            return layer(x)

        def get_layer_activation(i: int | Array) -> Callable:
            return jtu.tree_map(lambda x: x[i] if is_array(x) else x, self.activation)

        if self.scan:
            # Input layer + activation
            input_layer, hidden_layers, output_layer = self.layers
            x = input_layer(x)
            x = _vmap_a_of_b(get_layer_activation(0), x)

            # Hidden layers with scan
            def scan_fn(
                x: Array, layer_params: tuple[Int[Array, ""], Linear]
            ) -> tuple[Array, None]:
                i, layer = layer_params
                x = layer(x)  # Apply the layer
                x = _vmap_a_of_b(get_layer_activation(i), x)  # Apply activation
                return x, None

            x, _ = jax.lax.scan(scan_fn, x, (jnp.arange(1, self.depth), hidden_layers))
            x = output_layer(x)
        else:
            for i, layer in enumerate(self.layers[:-1]):
                x = layer(x)
                x = _vmap_a_of_b(get_layer_activation(i), x)
            x = self.layers[-1](x)

        if self.out_size == "scalar":
            x = self.final_activation(x)
        else:
            x = _vmap_a_of_b(self.final_activation, x)
        return x
