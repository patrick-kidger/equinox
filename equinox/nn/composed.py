import typing
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
    Sequence,
    Tuple,
    Union,
)

import jax.nn as jnn
import jax.random as jrandom
import jax.numpy as jnp
from jaxtyping import Array
from jax import lax
from jax.tree_util import tree_flatten

from ..custom_types import PRNGKey
from ..module import Module, static_field
from .linear import Linear


def _identity(x):
    return x


if getattr(typing, "GENERATING_DOCUMENTATION", False):

    def relu(_):
        pass

    jnn.relu = relu
    _identity.__qualname__ = "identity"  # Renamed for nicer documentation.


class MLP(Module):
    """Standard Multi-Layer Perceptron; also known as a feed-forward network.

    !!! faq

        If you get a TypeError saying an object is not a valid JAX type, see the
            [FAQ](https://docs.kidger.site/equinox/faq/)."""

    layers: Tuple[Linear, ...]
    activation: Callable
    final_activation: Callable
    in_size: Union[int, Literal["scalar"]] = static_field()
    out_size: Union[int, Literal["scalar"]] = static_field()
    width_size: int = static_field()
    depth: int = static_field()

    def __init__(
        self,
        in_size: Union[int, Literal["scalar"]],
        out_size: Union[int, Literal["scalar"]],
        width_size: int,
        depth: int,
        activation: Callable = jnn.relu,
        final_activation: Callable = _identity,
        *,
        key: PRNGKey,
        **kwargs,
    ):
        """**Arguments**:

        - `in_size`: The input size. The input to the module should be a vector of
            shape `(in_features,)`
        - `out_size`: The output size. The output from the module will be a vector
            of shape `(out_features,)`.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers.
        - `activation`: The activation function after each hidden layer. Defaults to
            ReLU.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        Note that `in_size` also supports the string `"scalar"` as a special value.
        In this case the input to the module should be of shape `()`.

        Likewise `out_size` can also be a string `"scalar"`, in which case the
        output from the module will have shape `()`.
        """

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
        self.layers = tuple(layers)
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        self.activation = activation
        self.final_activation = final_activation

    def _scan_hidden_layers(self, x):
        def step(inp, layer_weights):
            weight = layer_weights[:, :-1]
            bias = layer_weights[:, -1].T
            layer = self.layers[1]._tree_unflatten(
                [
                    ["weight", "bias"],
                    ["in_features", "out_features", "use_bias"],
                    [
                        self.layers[1].in_features,
                        self.layers[1].out_features,
                        self.layers[1].use_bias,
                    ],
                ],
                [weight, bias],
            )
            inp = self.activation(layer(inp))
            return inp, None

        flattened_layers, _ = tree_flatten(self.layers[1:-1])
        concatenated_weight_bias = [
            jnp.concatenate([weight, bias.reshape(-1, 1)], axis=1)
            for weight, bias in zip(flattened_layers[::2], flattened_layers[1::2])
        ]
        stacked_weights = jnp.stack(concatenated_weight_bias)

        x, _ = lax.scan(step, x, stacked_weights)
        return x

    def __call__(self, x: Array, *, key: Optional[PRNGKey] = None) -> Array:
        """**Arguments:**

        - `x`: A JAX array with shape `(in_size,)`. (Or shape `()` if
            `in_size="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(out_size,)`. (Or shape `()` if `out_size="scalar"`.)
        """

        if len(self.layers) > 1:
            x = self.layers[0](x)
            x = self.activation(x)

        if len(self.layers) > 2:
            x = self._scan_hidden_layers(x)

        x = self.layers[-1](x)
        x = self.final_activation(x)
        return x


@runtime_checkable
class _SequentialLayer(Protocol):
    def __call__(self, x: Array, *, key: Optional[PRNGKey] = None) -> Array:
        ...


class Sequential(Module):
    """A sequence of [`equinox.Module`][]s applied in order.

    !!! note

        Activation functions can be added by wrapping them in [`equinox.nn.Lambda`][].
    """

    layers: Tuple[_SequentialLayer, ...]

    def __init__(self, layers: Sequence[_SequentialLayer]):
        self.layers = tuple(layers)

    def __call__(self, x: Array, *, key: Optional[PRNGKey] = None) -> Array:
        """**Arguments:**

        - `x`: Argument passed to the first member of the sequence.
        - `key`: A `jax.random.PRNGKey`, which will be split and passed to every layer
            to provide any desired randomness. (Optional. Keyword only argument.)

        **Returns:**

        The output of the last member of the sequence.
        """

        if key is None:
            keys = [None] * len(self.layers)
        else:
            keys = jrandom.split(key, len(self.layers))
        for layer, key in zip(self.layers, keys):
            x = layer(x, key=key)
        return x

    def __getitem__(self, i: Union[int, slice]) -> _SequentialLayer:
        if isinstance(i, int):
            return self.layers[i]
        elif isinstance(i, slice):
            return Sequential(self.layers[i])
        else:
            raise TypeError(f"Indexing with type {type(i)} is not supported")

    def __iter__(self):
        yield from self.layers

    def __len__(self):
        return len(self.layers)


Sequential.__init__.__doc__ = """**Arguments:**

- `layers`: A sequence of [`equinox.Module`][]s.
"""


class Lambda(Module):
    """Wraps a callable (e.g. an activation function) for use with
    [`equinox.nn.Sequential`][].

    Precisely, this just adds an extra `key` argument (that is ignored). Given some
    function `fn`, then `Lambda` is essentially a convenience for `lambda x, key: f(x)`.

    !!! faq

        If you get a TypeError saying the function is not a valid JAX type, see the
            [FAQ](https://docs.kidger.site/equinox/faq/).

    !!! Example

        ```python
           model = eqx.nn.Sequential(
               [
                   eqx.nn.Linear(...),
                   eqx.nn.Lambda(jax.nn.relu),
                   ...
               ]
           )
        ```
    """

    fn: Callable[[Any], Any]

    def __call__(self, x: Array, *, key: Optional[PRNGKey] = None) -> Array:
        """**Arguments:**

        - `x`: The input JAX array.
        - `key`: Ignored.

        **Returns:**

        The output of the `fn(x)` operation.
        """
        return self.fn(x)


Lambda.__init__.__doc__ = """**Arguments:**

- `fn`: A callable to be wrapped in [`equinox.Module`][].
"""
