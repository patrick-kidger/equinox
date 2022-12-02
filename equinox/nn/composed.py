import typing
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import jax
import jax.nn as jnn
import jax.random as jrandom
from jaxtyping import Array

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

    layers: List[Linear]
    activation: Callable
    final_activation: Callable
    in_size: int = static_field()
    out_size: int = static_field()
    width_size: int = static_field()
    depth: int = static_field()

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        activation: Callable = jnn.relu,
        final_activation: Callable = _identity,
        *,
        key: "jax.random.PRNGKey",
        **kwargs,
    ):
        """**Arguments**:

        - `in_size`: The size of the input layer.
        - `out_size`: The size of the output layer.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers.
        - `activation`: The activation function after each hidden layer. Defaults to
            ReLU.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
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
        self.layers = layers
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        self.activation = activation
        self.final_activation = final_activation

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**

        - `x`: A JAX array with shape `(in_size,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(out_size,)`.
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        x = self.final_activation(x)
        return x


class Sequential(Module):
    """A sequence of [`equinox.Module`][]s applied in order.

    !!! note

        Activation functions can be added by wrapping them in [`equinox.nn.Lambda`][].
    """

    layers: Tuple[Module, ...]

    def __init__(self, layers: Sequence[Module]):
        self.layers = tuple(layers)

    def __call__(self, x: Any, *, key: Optional["jax.random.PRNGKey"] = None) -> Any:
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

    def __getitem__(self, i: Union[int, slice]) -> Module:
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

    fn: Callable

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
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
