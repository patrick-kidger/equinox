import types
from collections.abc import Callable
from typing import Any, Generic, TYPE_CHECKING, TypeVar, Union
from typing_extensions import ParamSpec

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import wadler_lindig as wl
from jaxtyping import PyTree

from .._module import field, Module
from .._pretty_print import tree_pformat
from .._tree import tree_at, tree_equal


_Value = TypeVar("_Value")
_P = ParamSpec("_P")
_T = TypeVar("_T")


class _Sentinel(Module):
    """A module for sentinels that can be passed dynamically."""

    pass


# Used as a sentinel in two ways: keeping track of updated `State`s, and keeping track
# of deleted initial states.
_sentinel = _Sentinel()


class StateIndex(Module, Generic[_Value], strict=True):
    """This wraps together (a) a unique dictionary key used for looking up a stateful
    value, and (b) how that stateful value should be initialised.

    !!! Example

        ```python
        class MyStatefulLayer(eqx.Module):
            index: eqx.nn.StateIndex

            def __init__(self):
                init_state = jnp.array(0)
                self.index = eqx.nn.StateIndex(init_state)

            def __call__(self, x: Array, state: eqx.nn.State) -> tuple[Array, eqx.nn.State]:
                current_state = state.get(self.index)
                new_x = x + current_state
                new_state = state.set(self.index, current_state + 1)
                return new_x, new_state
        ```

    See also e.g. the source code of built-in stateful layers like
    [`equinox.nn.BatchNorm`][] for further reference.
    """  # noqa: E501

    # Starts off as None when initialised; later replaced with an `int` inside
    # `make_with_state`.
    marker: Union[object, int] = field(static=True)
    init: Union[_Value, _Sentinel]

    def __init__(self, init: _Value):
        """**Arguments:**

        - `init`: The initial value for the state.
        """
        if isinstance(init, types.FunctionType):
            # Technically a function is valid here, since we could allow any pytree.
            # In practice that's weird / kind of useless, so better to explicitly raise
            # the deprecation error.
            raise ValueError(
                "As of Equinox v0.11.0, `eqx.nn.StateIndex` now accepts the value "
                "of the initial state directly. (Not a function that creates the "
                "initial state.)"
            )
        self.marker = object()
        self.init = init


def _is_index(x: Any) -> bool:
    return isinstance(x, StateIndex)


_state_error = """
Attempted to use old state. Probably you have done something like:
```
x, state2 = layer1(x, state1)
x, state3 = layer1(x, state1)  # bug! Re-used state1 instead of using state2.
```

If you have done this intentionally, because you want to use an old state, then you can
avoid this error by making a clone of the state:
```
leaves, treedef = jax.tree_util.tree_flatten(state)
state_clone = jax.tree_util.tree_unflatten(treedef, leaves)
```
""".strip()


# Basically just a dictionary which (a) works only with StateIndex-s, and which (b)
# works around a JAX bug that prevents flattening dicts with `object()` keys, and which
# (c) does error-checking that you're using the most up-to-date version of it.
@jtu.register_pytree_node_class
class State:
    """Stores the state of a model. For example, the running statistics of all
    [`equinox.nn.BatchNorm`][] layers in the model.

    This is essentially a dictionary mapping from [`equinox.nn.StateIndex`][]s to
    PyTrees of arrays.

    This class should be initialised via [`equinox.nn.make_with_state`][].
    """

    def __init__(self, model: PyTree):
        """**Arguments:**

        - `model`: any PyTree. All stateful layers (e.g. [`equinox.nn.BatchNorm`][])
            will have their initial state stored inside the `State` object.
        """
        # Note that de/serialisation depends on the ordered-ness of this dictionary,
        # between serialisation and deserialisation.
        state = {}
        leaves = jtu.tree_leaves(model, is_leaf=_is_index)
        for leaf in leaves:
            if _is_index(leaf):
                if isinstance(leaf.init, _Sentinel):
                    raise ValueError(
                        "Do not call `eqx.nn.State(model)` directly. You should call "
                        "`eqx.nn.make_with_state(ModelClass)(...args...)` instead."
                    )
                state[leaf.marker] = jtu.tree_map(jnp.asarray, leaf.init)
        self._state: Union[_Sentinel, dict[object | int, Any]] = state

    def get(self, item: StateIndex[_Value]) -> _Value:
        """Given an [`equinox.nn.StateIndex`][], returns the value of its state.

        **Arguments:**

        - `item`: an [`equinox.nn.StateIndex`][].

        **Returns:**

        The current state associated with that index.
        """
        if isinstance(self._state, _Sentinel):
            raise ValueError(_state_error)
        if type(item) is not StateIndex:
            raise ValueError("Can only use `eqx.nn.StateIndex`s as state keys.")
        return self._state[item.marker]

    def set(self, item: StateIndex[_Value], value: _Value) -> "State":
        """Sets a new value for an [`equinox.nn.StateIndex`][], **and returns the
        updated state**.

        **Arguments:**

        - `item`: an [`equinox.nn.StateIndex`][].
        - `value`: the new value associated with that index.

        **Returns:**

        A new [`equinox.nn.State`][] object, with the update.

        As a safety guard against accidentally writing `state.set(item, value)` without
        assigning it to a new value, then the old object (`self`) will become invalid.
        """
        if isinstance(self._state, _Sentinel):
            raise ValueError(_state_error)
        if type(item) is not StateIndex:
            raise ValueError("Can only use `eqx.nn.StateIndex`s as state keys.")
        old_value = self._state[item.marker]
        value = jtu.tree_map(jnp.asarray, value)
        old_struct = jax.eval_shape(lambda: old_value)
        new_struct = jax.eval_shape(lambda: value)
        if tree_equal(old_struct, new_struct) is not True:
            old_repr = tree_pformat(old_struct, struct_as_array=True)
            new_repr = tree_pformat(new_struct, struct_as_array=True)
            raise ValueError(
                "Old and new values have different structures/shapes/dtypes. The old "
                f"value is {old_repr} and the new value is {new_repr}."
            )
        state = self._state.copy()  # pyright: ignore
        state[item.marker] = value
        new_self = object.__new__(State)
        new_self._state = state
        self._state = _sentinel
        return new_self

    def substate(self, pytree: PyTree) -> "State":
        """Creates a smaller `State` object, that tracks only the states of some smaller
        part of the overall model.

        **Arguments:**

        - `pytree`: any PyTree. It will be iterated over to check for
            [`equinox.nn.StateIndex`]s.

        **Returns:**

        A new [`equinox.nn.State`][] object, which tracks only some of the overall
        states.
        """
        if isinstance(self._state, _Sentinel):
            raise ValueError(_state_error)
        leaves = jtu.tree_leaves(pytree, is_leaf=_is_index)
        markers = [x.marker for x in leaves if _is_index(x)]
        substate = {k: self._state[k] for k in markers}  # pyright: ignore
        subself = object.__new__(State)
        subself._state = substate
        return subself

    def update(self, substate: "State") -> "State":
        """Takes a smaller `State` object (typically produces via `.substate`), and
        updates states by using all of its values.

        **Arguments:**

        - `substate`: a `State` object whose keys are a subset of the keys of `self`.

        **Returns:**

        A new [`equinox.nn.State`][] object, containing all of the updated values.

        As a safety guard against accidentally writing `state.set(item, value)` without
        assigning it to a new value, then the old object (`self`) will become invalid.
        """
        if isinstance(self._state, _Sentinel):
            raise ValueError(_state_error)
        if type(substate) is not State:
            raise ValueError("Can only use `eqx.nn.State`s in `update`.")
        state = self._state.copy()  # pyright: ignore
        for key, value in substate._state.items():  # pyright: ignore
            if key not in state:
                raise ValueError(
                    "Cannot call `state1.update(state2)` unless `state2` is a substate "
                    "of `state1`."
                )
            state[key] = value
        new_self = object.__new__(State)
        new_self._state = state
        self._state = _sentinel
        return new_self

    def __repr__(self):
        return tree_pformat(self)

    def __pdoc__(self, **kwargs):
        if isinstance(self._state, _Sentinel):
            return wl.TextDoc("State(~old~)")
        else:
            docs = wl.named_objs(
                [
                    (hex(id(key)), value)
                    for key, value in self._state.items()  # pyright: ignore
                ],
                **kwargs,
            )
            return wl.bracketed(
                begin=wl.TextDoc("State("),
                docs=docs,
                sep=wl.comma,
                end=wl.TextDoc(")"),
                indent=kwargs["indent"],
            )

    def tree_flatten(self):
        if isinstance(self._state, _Sentinel):
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


def _delete_init_state(x):
    if _is_index(x):
        return tree_at(lambda y: y.init, x, _sentinel)
    else:
        return x


def delete_init_state(model: PyTree) -> PyTree:
    """For memory efficiency, this deletes the initial state stored within a model.

    Every stateful layer keeps a copy of the initial value of its state. This is then
    collected by [`equinox.nn.State`][], when it is called on the model. However, this
    means that the model must keep a copy of the initial state around, in case
    `eqx.nn.State` is called on it again. This extra copy consumes extra memory.

    But in practice, it is quite common to only need to initialise the state once. In
    this case, we can use this function to delete this extra copy, and in doing so save
    some memory.

    !!! Example

        Here is the typical pattern in which this is used:
        ```python
        model_and_state = eqx.nn.BatchNorm(...)
        state = eqx.nn.State(model_and_state)
        model = eqx.nn.delete_init_state(model)
        del model_and_state  # ensure this goes out of scope and is garbage collected
        ```
        Indeed the above is precisely what [`equinox.nn.make_with_state`][] does.

    **Arguments:**

    - `model`: any PyTree.

    **Returns:**

    A copy of `model`, with all the initial states stripped out. (As in the examples
    above, you should then dispose of the original `model` object.)
    """
    return jtu.tree_map(_delete_init_state, model, is_leaf=_is_index)


def make_with_state(make_model: Callable[_P, _T]) -> Callable[_P, tuple[_T, State]]:
    """This function is the most common API for working with stateful models. This
    initialises both the parameters and the state of a stateful model.

    `eqx.nn.make_with_state(Model)(*args, **kwargs)` simply calls
    `model_with_state = Model(*args, **kwargs)`, and then partitions the resulting
    PyTree into two pieces: the parameters, and the state.

    **Arguments:**

    - `make_model`: some callable returning a PyTree.

    **Returns:**

    A callable, which when evaluated returns a 2-tuple of `(model, state)`, where
    `model` is the result of `make_model(*args, **kwargs)` but with all of the initial
    states stripped out, and `state` is an [`equinox.nn.State`][] object encapsulating
    the initial states.

    !!! Example

        See [the stateful example](../../examples/stateful.ipynb) for a runnable
        example.

        ```python
        class Model(eqx.Module):
            def __init__(self, foo, bar):
                ...

            ...

        model, state = eqx.nn.make_with_state(Model)(foo=3, bar=4)
        ```
    """

    # _P.{args, kwargs} not supported by beartype
    if TYPE_CHECKING:

        def make_with_state_impl(
            *args: _P.args, **kwargs: _P.kwargs
        ) -> tuple[_T, State]: ...

    else:

        def make_with_state_impl(*args, **kwargs) -> tuple[_T, State]:
            model = make_model(*args, **kwargs)

            # Replace all markers with `int`s. This is needed to ensure that two calls
            # to `make_with_state` produce compatible models and states.
            leaves, treedef = jtu.tree_flatten(model, is_leaf=_is_index)
            counter = 0
            new_leaves = []
            for leaf in leaves:
                if _is_index(leaf):
                    leaf = StateIndex(leaf.init)
                    object.__setattr__(leaf, "marker", counter)
                    counter += 1
                new_leaves.append(leaf)
            model = jtu.tree_unflatten(treedef, new_leaves)

            state = State(model)
            model = delete_init_state(model)
            return model, state

    return make_with_state_impl
