from collections.abc import Callable, Sequence
from typing import Any, Optional, overload, Union

import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from .._better_abstract import AbstractClassVar
from .._custom_types import sentinel
from .._module import Module, StrictConfig
from ._misc import named_scope
from ._stateful import State


class StatefulLayer(Module, strict=StrictConfig(allow_abstract_name=True)):
    """An abstract base class, used by [`equinox.nn.Sequential`][], to mark that a
    layer might be stateful. If `Sequential` sees that a layer inherits from
    `StatefulLayer`, then it will call `layer.is_stateful()` to check whether to
    call the layer as `new_x = layer(x)` or `(new_x, new_state) = layer(x, state)`.
    """

    def is_stateful(self) -> bool:
        """Indicates whether this layer should be considered stateful.

        The default implementation just returns True, but subclasses may override
        this to provide custom logic if the layer is only "maybe stateful". (E.g. if
        they optionally use stateful sublayers themselves.)

        **Arguments:**

        None

        **Returns:**

        A boolean. `True` indicates that the layer should be called as
        `(new_x, new_state) = layer(x, state)`. `False` indicates that the layer
        should be called as `new_x = layer(x)`.
        """
        return True

    __call__: AbstractClassVar[Callable]


class Sequential(StatefulLayer, strict=StrictConfig(allow_method_override=True)):
    """A sequence of [`equinox.Module`][]s applied in order.

    !!! note

        Activation functions can be added by wrapping them in [`equinox.nn.Lambda`][].
    """

    layers: tuple

    def __init__(self, layers: Sequence[Callable]):
        """**Arguments:**

        - `layers`: A sequence of [`equinox.Module`][]s.
        """

        self.layers = tuple(layers)

    def is_stateful(self) -> bool:
        return any(
            isinstance(x, StatefulLayer) and x.is_stateful() for x in self.layers
        )

    @overload
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array: ...

    @overload
    def __call__(
        self, x: Array, state: State, *, key: Optional[PRNGKeyArray] = None
    ) -> tuple[Array, State]: ...

    @named_scope("eqx.nn.Sequential")
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
            if isinstance(layer, StatefulLayer) and layer.is_stateful():
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


class Lambda(Module, strict=True):
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
