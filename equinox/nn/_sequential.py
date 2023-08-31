import abc
from collections.abc import Callable, Sequence
from typing import Any, Optional, overload, Union

import jax
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from .._custom_types import sentinel
from .._module import Module
from ._stateful import State


class Sequential(Module):
    """A sequence of [`equinox.Module`][]s applied in order.

    !!! note

        Activation functions can be added by wrapping them in [`equinox.nn.Lambda`][].
    """

    layers: tuple

    def __init__(self, layers: Sequence[Callable]):
        self.layers = tuple(layers)

    @overload
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        ...

    @overload
    def __call__(
        self, x: Array, state: State, *, key: Optional[PRNGKeyArray] = None
    ) -> tuple[Array, State]:
        ...

    @jax.named_scope("eqx.nn.Sequential")
    def __call__(
        self,
        x: Array,
        state: State = sentinel,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Union[Array, tuple[Array, State]]:
        """**Arguments:**

        - `x`: passed to the first member of the sequence.
        - `state`: If provided, then it is passed to, and updated from, any layer
            which subclasses [`equinox.nn.StatefulLayer`][].
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**
        The output of the last member of the sequence.

        If `state` is passed, then a 2-tuple of `(output, state)` is returned.
        If `state` is not passed, then just the output is returned.
        """

        if key is None:
            keys = [None] * len(self.layers)
        else:
            keys = jr.split(key, len(self.layers))
        for layer, key in zip(self.layers, keys):
            if isinstance(layer, StatefulLayer):
                x, state = layer(x, state=state, key=key)
            else:
                x = layer(x, key=key)
        if state is sentinel:
            return x
        else:
            return x, state

    def __getitem__(self, i: Union[int, slice]) -> Callable:
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


class StatefulLayer(Module):
    """An abstract base class, used to mark a stateful layer for the sake of
    [`equinox.nn.Sequential`][]. If `Sequential` sees that a layer inherits
    from `StatefulLayer`, then it will know to pass in `state` as well as the
    piped data `x`.

    Subclasses must implement the `__call__` method that takes input data and the
    current state as arguments and returns the output data and updated state.
    """

    @abc.abstractmethod
    def __call__(
        self,
        x: Array,
        state: State,
        *,
        key: Optional[PRNGKeyArray],
    ) -> tuple[Array, State]:
        """The function signature that stateful layers should conform to, to be
        compatible with [`equinox.nn.Sequential`][].
        """
        raise NotImplementedError("Subclasses must implement the __call__ method.")


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
