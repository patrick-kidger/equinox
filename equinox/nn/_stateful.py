import types
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree

from .._module import Module
from .._pretty_print import bracketed, named_objs, text, tree_pformat


_Value = TypeVar("_Value")


class StateIndex(Module, Generic[_Value]):
    """This is an advanced feature, used when creating custom stateful layers.

    This wraps a dictionary key used for looking up the stateful value.

    See the source code of [`equinox.nn.BatchNorm`][] for an example.
    """

    marker: object
    init: types.FunctionType

    def __init__(self, init: Callable[..., _Value]):
        """**Arguments:**

        - `init`: A function that is used to initialise the state of the layer. Should
            be a function that returns a PyTree of JAX arrays. It will be called with
            all keyword arguments passed to [`equinox.nn.State.__init__`].
        """
        if not isinstance(init, types.FunctionType):
            raise TypeError("`StateIndex(init=...)` must be a function")
        self.marker = object()
        self.init = init


def _is_index(x: Any) -> bool:
    return isinstance(x, StateIndex)


_sentinel = object()


_state_error = """
Attempted to use old state. Probably you have done something like:

x, state2 = layer1(x, state1)
x, state3 = layer1(x, state1)  # bug! Re-used state1 instead of using state2.
""".strip()


# Basically just a dictionary which (a) works only with Markers, and which (b) works
# around a JAX bug that prevents flattening dicts with `object()` keys, and which (c)
# does error-checking that you're using the most up-to-date version of it.
@jtu.register_pytree_node_class
class State:
    """Stores the state of a model. For example, the running statistics of all
    [`equinox.nn.BatchNorm`][] layers in the model.

    Most models won't need this. (As most models don't have any stateful layers.)
    If used, the state will be passed to each layer at call time; see the
    [stateful example](../../examples/stateful.ipynb).
    """

    def __init__(self, model: PyTree, **kwargs):
        """**Arguments:**

        - `model`: any PyTree. All stateful layers (e.g. [`equinox.nn.BatchNorm`][])
            will have their state initialised and stored inside the `State` object.
        - `**kwargs`: all keyword arguments are forwarded to the `init` function of
            `equinox.nn.StateIndex(init=...)`  (used inside each stateful layer).
        """
        # Note that de/serialisation depends on the ordered-ness of this dictionary,
        # between serialisation and deserialisation.
        state = {}
        leaves = jtu.tree_leaves(model, is_leaf=_is_index)
        for leaf in leaves:
            if _is_index(leaf):
                value = leaf.init(**kwargs)
                value = jtu.tree_map(jnp.asarray, value)
                state[leaf.marker] = value
        self._state = state

    def get(self, item: StateIndex[_Value]) -> _Value:
        if self._state is _sentinel:
            raise ValueError(_state_error)
        if type(item) is not StateIndex:
            raise ValueError("Can only use `eqx.nn.Marker`s as state keys.")
        return self._state[item.marker]  # pyright: ignore

    def set(self, item: StateIndex[_Value], value: _Value) -> "State":
        if self._state is _sentinel:
            raise ValueError(_state_error)
        if type(item) is not StateIndex:
            raise ValueError("Can only use `eqx.nn.Marker`s as state keys.")
        old_value = self._state[item.marker]  # pyright: ignore
        value = jtu.tree_map(jnp.asarray, value)
        if jax.eval_shape(lambda: old_value) != jax.eval_shape(lambda: value):
            raise ValueError("Old and new values have different structures.")
        self._state[item.marker] = value  # pyright: ignore
        new_self = object.__new__(State)
        new_self._state = self._state
        self._state = _sentinel
        return new_self

    def __repr__(self):
        return tree_pformat(self)

    def __tree_pp__(self, **kwargs):
        if self._state is _sentinel:
            return text("State(~old~)")
        else:
            objs = named_objs(
                [
                    (hex(id(key)), value)
                    for key, value in self._state.items()  # pyright: ignore
                ],
                **kwargs,
            )
            return bracketed(
                name=text("State"),
                indent=kwargs["indent"],
                objs=objs,
                lbracket="(",
                rbracket=")",
            )

    def tree_flatten(self):
        if self._state is _sentinel:
            raise ValueError(_state_error)
        keys = tuple(self._state.keys())  # pyright: ignore
        values = tuple(self._state[k] for k in keys)  # pyright: ignore
        return values, keys

    @classmethod
    def tree_unflatten(cls, keys, values):
        self = object.__new__(cls)
        state = {}
        assert len(keys) == len(values)
        for key, value in zip(keys, values):
            state[key] = value
        self._state = state
        return self
