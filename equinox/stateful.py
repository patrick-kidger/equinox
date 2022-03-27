import weakref
from typing import Tuple

import jax
import jax.experimental.host_callback as hcb
import jax.lax as lax

from .custom_types import PyTree
from .filters import combine, is_array, partition
from .module import Module, static_field


array_cache = weakref.WeakKeyDictionary()
nonarray_cache = weakref.WeakKeyDictionary()
array_shape_cache = weakref.WeakKeyDictionary()


class _IndexObj:
    __slots__ = ("__weakref__",)


class StateIndex(Module):
    """An index for setting or getting a piece of state with [`equinox.get_state`][] or
    [`equinox.set_state`][].
    """

    obj: _IndexObj = static_field()

    def __init__(self):
        self.obj = _IndexObj()


def _get_state_hcb(index: StateIndex) -> PyTree:
    return array_cache[index.obj]


def get_state(index: StateIndex) -> PyTree:
    """Get some previously saved state.

    **Arguments:**

    - `index`: The index of the state to look up. Should be an instance of
        [`equinox.StateIndex`][].

    **Returns:**

    Whatever the previously saved state is.

    **Raises:**

    A `RuntimeError` if no state has previously been saved with this `index`.

    !!! warning

        This means that your operation will no longer be a pure function.
    """
    try:
        array_shapes = array_shape_cache[index.obj]
    except KeyError as e:
        raise RuntimeError("Trying to access state that has not yet been set.") from e
    arrays = hcb.call(_get_state_hcb, index, result_shape=array_shapes)
    nonarrays = nonarray_cache[index.obj]
    return combine(arrays, nonarrays)


def _set_state(index__arrays: Tuple[StateIndex, PyTree]) -> None:
    index, arrays = index__arrays
    array_cache[index.obj] = arrays


def set_state(index: StateIndex, value: PyTree) -> None:
    """Save an arbitrary PyTree as a side-effect.

    **Arguments:**

    - `index`: A key under which to save the state. Should be an instance of
        [`equinox.StateIndex`][].
    - `value`: An arbitrary PyTree to save.

    **Returns:**

    `None`.

    **Raises:**

    A `ValueError` if this `index` has previously been used to save a `value` with a
    different PyTree structure.

    !!! info

        The same `index` can be used multiple times, to overwrite a previously saved
        value. The new and old `value` must both have the same PyTree structure, however.
    """

    arrays, nonarrays = partition(value, is_array)
    arrays = jax.tree_map(lax.stop_gradient, arrays)
    try:
        array_shapes = array_shape_cache[index.obj]
    except KeyError:
        array_shapes = jax.eval_shape(lambda: arrays)
        array_shape_cache[index.obj] = array_shapes
    else:
        # It isn't technically necessary that we preserve shape and dtype, but it seems
        # like a worthwhile sanity check in 99% of use cases.
        if jax.eval_shape(lambda: arrays) != array_shapes:
            raise ValueError(
                "New state has different shape or dtype to previous state."
            )
    hcb.call(_set_state, (index, arrays))
    nonarray_cache[index.obj] = nonarrays
