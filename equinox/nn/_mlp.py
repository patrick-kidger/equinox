from collections.abc import Callable
from typing import (
    Literal,
)

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray

from .._doc_utils import doc_repr
from .._filters import combine, is_array, partition
from .._misc import default_floating_dtype
from .._module import field, Module
from .._vmap_pmap import filter_vmap
from ._linear import Linear
from ._misc import named_scope


_identity = doc_repr(lambda x: x, "lambda x: x")
_relu = doc_repr(jnn.relu, "<function relu>")


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
        key: PRNGKeyArray,
    ):
        """**Arguments**:

        - `in_size`: The input size. The input to the module should be a vector of
            shape `(in_features,)`
        - `out_size`: The output size. The output from the module will be a vector
            of shape `(out_features,)`.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers, including the output layer.
            For example, `depth=2` results in an network with layers:
            [`Linear(in_size, width_size)`, `Linear(width_size, width_size)`,
            `Linear(width_size, out_size)`].
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
                Linear(in_size, out_size, use_final_bias, dtype=dtype, key=keys[0])
            )
        else:
            layers.append(
                Linear(in_size, width_size, use_bias, dtype=dtype, key=keys[0])
            )
            for i in range(depth - 1):
                layers.append(
                    Linear(
                        width_size, width_size, use_bias, dtype=dtype, key=keys[i + 1]
                    )
                )
            layers.append(
                Linear(width_size, out_size, use_final_bias, dtype=dtype, key=keys[-1])
            )
        self.layers = tuple(layers)
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
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
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            layer_activation = jtu.tree_map(
                lambda x: x[i] if is_array(x) else x, self.activation
            )
            x = filter_vmap(lambda a, b: a(b))(layer_activation, x)
        x = self.layers[-1](x)
        if self.out_size == "scalar":
            x = self.final_activation(x)
        else:
            x = filter_vmap(lambda a, b: a(b))(self.final_activation, x)
        return x


class ScanOverMLP(MLP):
    r"""Multi-layer perceptron with scan-over-layers pattern for fast compilation.

    Similar to ``equinox.nn.MLP``, but uses ``jax.lax.scan`` to iterate over
    identical hidden layers for improved compilation speed.

    The network consists of three components:
    - Input layer: maps in_size -> width_size
    - Hidden layers: scan-over-layers pattern, width_size -> width_size
    (repeated depth - 1 times)
    - Output layer: maps width_size -> out_size

    """

    layers: tuple[Linear, object, Linear]
    activation: Callable
    final_activation: Callable

    use_bias: bool = field(static=True)
    use_final_bias: bool = field(static=True)
    in_size: int | Literal["scalar"] = field(static=True)
    out_size: int | Literal["scalar"] = field(static=True)

    width_size: int = field(static=True)
    depth: int = field(static=True)

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
        key: PRNGKeyArray,
    ) -> None:
        """**Arguments**:

        - `in_size`: The input size. The input to the module should be a vector of
            shape `(in_features,)`
        - `out_size`: The output size. The output from the module will be a vector
            of shape `(out_features,)`.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers, including the output layer.
            For example, `depth=2` results in an network with layers:
            [`Linear(in_size, width_size)`, `Linear(width_size, width_size)`,
            `Linear(width_size, out_size)`].
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
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        Note that `in_size` also supports the string `"scalar"` as a special value.
        In this case the input to the module should be of shape `()`.

        Likewise `out_size` can also be a string `"scalar"`, in which case the
        output from the module will have shape `()`.
        """
        if depth < 1:
            msg = "depth must be at least 1"
            raise ValueError(msg)

        dtype = default_floating_dtype() if dtype is None else dtype
        keys = jrandom.split(key, depth + 1)

        # Input layer
        input_layer = Linear(
            in_size, width_size, use_bias=use_bias, dtype=dtype, key=keys[0]
        )

        # Hidden layers: create depth-1 identical layers using filter_vmap
        def make_hidden_layer(k: PRNGKeyArray) -> Linear:
            return Linear(width_size, width_size, use_bias=use_bias, dtype=dtype, key=k)

        if depth > 1:
            hidden_keys = keys[1:depth]
            hidden_layers = filter_vmap(make_hidden_layer)(hidden_keys)
        else:
            # For depth == 1, construct an empty collection of hidden layers
            # with the correct tree structure and array dtypes/shapes.
            single_hidden = filter_vmap(make_hidden_layer)(keys[1:2])
            hidden_layers = jtu.tree_map(lambda x: x[:0], single_hidden)

        # Output layer
        output_layer = Linear(
            width_size, out_size, use_bias=use_final_bias, dtype=dtype, key=keys[-1]
        )

        # Store as tuple following Equinox MLP pattern
        self.layers = (input_layer, hidden_layers, output_layer)

        self.in_size = in_size
        self.width_size = width_size
        self.out_size = out_size
        self.depth = depth
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

        # In case `activation` or `final_activation` are learnt, then make a
        # separate copy of their weights for every neuron.
        self.activation = filter_vmap(
            filter_vmap(lambda: activation, axis_size=width_size), axis_size=depth
        )()
        if out_size == "scalar":
            self.final_activation = final_activation
        else:
            self.final_activation = filter_vmap(
                lambda: final_activation, axis_size=out_size
            )()

    @named_scope("eqx.nn.ScanOverMLP")
    def __call__(self, x: Array, *, key: PRNGKeyArray | None = None) -> Array:
        """**Arguments:**

        - `x`: A JAX array with shape `(in_size,)`. (Or shape `()` if
            `in_size="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(out_size,)`. (Or shape `()` if `out_size="scalar"`.)
        """
        input_layer, hidden_layers, output_layer = self.layers

        # Input layer + activation
        x = input_layer(x)
        # Extract the first activation (index 0) from the vmapped tree
        layer_activation = jtu.tree_map(
            lambda act: act[0] if is_array(act) else act, self.activation
        )
        x = filter_vmap(lambda a, b: a(b))(layer_activation, x)

        # Scan over hidden layers
        dynamic, static = partition(hidden_layers, is_array)

        def scan_fn(
            carry: tuple[Array, Array], layer_params: Linear
        ) -> tuple[tuple[Array, Array], None]:
            x, layer_idx = carry  # Unpack carry
            # Evaluate the layer
            layer = combine(layer_params, static)
            x = layer(x)
            # Extract and apply layer_idx-th activation from the vmapped tree
            layer_activation = jtu.tree_map(
                lambda act: act[layer_idx + 1] if is_array(act) else act,
                self.activation,
            )
            x = filter_vmap(lambda a, b: a(b))(layer_activation, x)
            # Return the output and incremented index
            return (x, layer_idx + 1), None

        (x, _), _ = jax.lax.scan(scan_fn, (x, jnp.array(0)), dynamic)

        # Output layer + final activation
        x = output_layer(x)
        if self.out_size == "scalar":
            x = self.final_activation(x)
        else:
            x = filter_vmap(lambda a, b: a(b))(self.final_activation, x)

        return x
