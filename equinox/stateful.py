import gc
import weakref
from typing import Tuple

import jax
import jax.experimental.host_callback as hcb
import jax.interpreters.batching as batching
import jax.interpreters.xla as xla
import jax.lax as lax

from .custom_types import Array, PyTree
from .filters import is_array
from .module import Module, static_field


# Note that `_unvmap` is not a safe-to-use primitive in general.
# For example, jnp.min(_unvmap(x)) will not vmap correctly.
# As such it is private and only used here.
# See also the unvmap_min etc. of Diffrax for safe-to-use unvmap-reduction primitives.


def _unvmap(x):
    batch_axes_smuggle = []
    out = _unvmap_p.bind(x, batch_axes_smuggle=batch_axes_smuggle)
    return out, batch_axes_smuggle


def _unvmap_batch_rule(inputs, batch_axes, *, batch_axes_smuggle):
    (x,) = inputs
    (batch_axis,) = batch_axes
    batch_axes_smuggle.append(batch_axis)
    return _unvmap_p.bind(x, batch_axes_smuggle=batch_axes_smuggle), batching.not_mapped


_unvmap_p = jax.core.Primitive("unvmap")
_unvmap_p.def_impl(lambda x, *, batch_axes_smuggle: x)
_unvmap_p.def_abstract_eval(
    lambda x, *, batch_axes_smuggle: jax.ShapedArray(x.shape, x.dtype)
)
batching.primitive_batchers[_unvmap_p] = _unvmap_batch_rule
xla.register_translation(
    _unvmap_p,
    xla.lower_fun(
        lambda x, *, batch_axes_smuggle: x, multiple_results=False, new_style=True
    ),
)


def _use_batching(x, y):
    return _use_batching_p.bind(x, y)


def _use_batching_batch_rule(inputs, batch_axes):
    x, y = inputs
    b, c = batch_axes
    assert b is batching.not_mapped
    assert c is not batching.not_mapped
    return _use_batching(x, y), c


_use_batching_p = jax.core.Primitive("use_batching")
_use_batching_p.def_impl(lambda x, y: x)
_use_batching_p.def_abstract_eval(lambda x, y: jax.ShapedArray(y.shape, y.dtype))
batching.primitive_batchers[_use_batching_p] = _use_batching_batch_rule
xla.register_translation(
    _use_batching_p,
    xla.lower_fun(lambda x, y: x, multiple_results=False, new_style=True),
)


# So the use of weak dictionaries is a bit of wishful thinking here, really.
# In practice JAX will cache the _IndexObj when it is passed across the hcb.call
# boundary.
# Which at least in part is what we want! We want the cached state to persist for
# as long as the XLA graph it's part of.
# The annoying bit is that even once that XLA graph vanishes, JAX still seems to keep
# things cached somewhere.
_state_cache = weakref.WeakKeyDictionary()
_state_shape_cache = weakref.WeakKeyDictionary()
_batch_axes_cache = weakref.WeakKeyDictionary()


def size_state_cache():
    gc.collect()  # Not 100% certain this is needed but worth doing just in case?
    assert len(_state_cache) == len(_state_shape_cache)
    assert len(_state_cache) == len(_batch_axes_cache)
    return len(_state_cache)


class _IndexObj:
    __slots__ = ("__weakref__",)


class StateIndex(Module):
    """An index for setting or getting a piece of state with [`equinox.get_state`][] or
    [`equinox.set_state`][].
    """

    obj: _IndexObj = static_field()

    def __init__(self):
        self.obj = _IndexObj()


class _NotAPyTree:
    def __init__(self, value):
        self.value = value


def _unvmap_wrap(x):
    x, batch_axes = _unvmap(x)
    return x, _NotAPyTree(batch_axes)


def _unvmap_pytree(x: PyTree):
    x, batch_axes = jax.tree_transpose(
        jax.tree_structure(x),
        jax.tree_structure((0, 0)),
        jax.tree_map(_unvmap_wrap, x),
    )
    batch_axes = jax.tree_map(lambda x: x.value, batch_axes)
    return x, batch_axes


def _get_state_hcb(index__batch_axes: StateIndex) -> PyTree:
    index, batch_axes = index__batch_axes
    # will raise a KeyError if get_state is called before set_state.
    _batch_axes = _batch_axes_cache[index.obj]
    if batch_axes != _batch_axes:
        raise ValueError("`like` must have the same batch axes as the saved state.")
    return _state_cache[index.obj]


def get_state(index: StateIndex, like: PyTree) -> PyTree:
    """Get some previously saved state.

    **Arguments:**

    - `index`: The index of the state to look up. Should be an instance of
        [`equinox.StateIndex`][].
    - `like`: A JAX array of the same shape, dtype, and batching as the state being
        looked up. In particular this is used to give the retrieved state the correct
        vmap behaviour.

    **Returns:**

    Whatever the previously saved state is.

    **Raises:**

    A `KeyError` if no state has previously been saved with this `index`.

    !!! warning

        This means that your operation will no longer be a pure function.
    """
    like = jax.tree_map(lax.stop_gradient, like)
    unvmap_like, batch_axes = _unvmap_pytree(like)
    unvmap_like_shape = jax.eval_shape(lambda: unvmap_like)
    state = hcb.call(
        _get_state_hcb, (index, batch_axes), result_shape=unvmap_like_shape
    )
    state = jax.tree_map(_use_batching, state, like)
    return state


def _set_state(index__state: Tuple[StateIndex, PyTree[Array]]) -> None:
    index, state = index__state
    _state_cache[index.obj] = state


def set_state(index: StateIndex, state: PyTree[Array]) -> None:
    """Save a PyTree of JAX arrays as a side-effect.

    **Arguments:**

    - `index`: A key under which to save the state. Should be an instance of
        [`equinox.StateIndex`][].
    - `state`: An PyTree of JAX arrays to save.

    **Returns:**

    `None`.

    **Raises:**

    A `ValueError` if this `index` has previously been used to save a `state` with a
    different PyTree structure.

    !!! info

        The same `index` can be used multiple times, to overwrite a previously saved
        value. The new and old `state` must both have the same PyTree structure, however.

    !!! warning

        Note that gradients will not preserved across [`equinox.get_state`][] and
        [`equinox.set_state`][].
    """

    if any(not is_array(x) for x in jax.tree_leaves(state)):
        raise ValueError("`state` must be a PyTree only containing JAX arrays.")
    state = jax.tree_map(lax.stop_gradient, state)
    state, batch_axes = _unvmap_pytree(state)
    try:
        _state_shape = _state_shape_cache[index.obj]
        _batch_axes = _batch_axes_cache[index.obj]
    except KeyError:
        _state_shape_cache[index.obj] = jax.eval_shape(lambda: state)
        _batch_axes_cache[index.obj] = batch_axes
    else:
        # It isn't technically necessary that we preserve shape and dtype, but it seems
        # like a worthwhile sanity check in 99% of use cases.
        # Not forgetting that our tracing could occur out-of-order wrt runtime -- so if
        # don't preserve these things then we'd have to make sure that order-of-tracing
        # matches order-of-evaluation and that'd be very finickity.
        if jax.eval_shape(lambda: state) != _state_shape:
            raise ValueError(
                "New state has different shape or dtype to previous state."
            )
        if batch_axes != _batch_axes:
            raise ValueError("New state has different batch axes to previous state.")
    hcb.call(_set_state, (index, state))
