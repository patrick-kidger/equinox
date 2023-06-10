from collections.abc import Callable, Sequence
from typing import (
    Any,
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
    Union,
)

import jax.nn as jnn
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from .._doc_utils import doc_repr
from .._module import field, Module
from ._linear import Linear


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
    in_size: Union[int, Literal["scalar"]] = field(static=True)
    out_size: Union[int, Literal["scalar"]] = field(static=True)
    width_size: int = field(static=True)
    depth: int = field(static=True)

    def __init__(
        self,
        in_size: Union[int, Literal["scalar"]],
        out_size: Union[int, Literal["scalar"]],
        width_size: int,
        depth: int,
        activation: Callable = _relu,
        final_activation: Callable = _identity,
        use_bias: bool = True,
        use_final_bias: bool = True,
        *,
        key: PRNGKeyArray,
        **kwargs,
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
            layers.append(Linear(in_size, out_size, use_final_bias, key=keys[0]))
        else:
            layers.append(Linear(in_size, width_size, use_bias, key=keys[0]))
            for i in range(depth - 1):
                layers.append(Linear(width_size, width_size, use_bias, key=keys[i + 1]))
            layers.append(Linear(width_size, out_size, use_final_bias, key=keys[-1]))
        self.layers = tuple(layers)
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        self.activation = activation
        self.final_activation = final_activation
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """**Arguments:**

        - `x`: A JAX array with shape `(in_size,)`. (Or shape `()` if
            `in_size="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(out_size,)`. (Or shape `()` if `out_size="scalar"`.)
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        x = self.final_activation(x)
        return x


@runtime_checkable
class _SequentialLayer(Protocol):
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        ...


class Sequential(Module):
    """A sequence of [`equinox.Module`][]s applied in order.

    !!! note

        Activation functions can be added by wrapping them in [`equinox.nn.Lambda`][].
    """

    layers: tuple[_SequentialLayer, ...]

    def __init__(self, layers: Sequence[_SequentialLayer]):
        self.layers = tuple(layers)

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
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

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
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
