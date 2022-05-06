import weakref
from dataclasses import field
from typing import Tuple

import jax
import jax.experimental.host_callback as hcb
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.interpreters.xla as xla
import jax.lax as lax
import jax.numpy as jnp

from ..custom_types import Array, PyTree
from ..filters import is_array
from ..module import Module, static_field
from ..tree import tree_at


# So the use of a weak dictionary is a bit of wishful thinking here, really.
# In practice JAX will cache the _IndexObj when it is passed across the hcb.call
# boundary.
# Which at least in part is what we want! We want the cached state to persist for
# as long as the XLA graph it's part of.
# The annoying bit is that even once that XLA graph vanishes, JAX still seems to keep
# things cached somewhere.
_state_cache = weakref.WeakKeyDictionary()


class _IndexObj:
    __slots__ = ("__weakref__",)


class _FixedInt:
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return type(self) == type(other)


class StateIndex(Module):
    """An index for setting or getting a piece of state with
    [`equinox.experimental.get_state`][] or [`equinox.experimental.set_state`][].

    You should typically treat this like a model parameter.

    !!! example

        ```python
        import equinox as eqx
        import equinox.experimental as eqxe
        import jax.numpy as jnp

        class CacheInput(eqx.Module):
            index: eqxe.StateIndex

            def __init__(self, input_shape):
                self.index = eqxe.StateIndex()
                eqxe.set_state(self.index, jnp.zeros(input_shape))

            def __call__(self, x):
                last_x = eqxe.get_state(self.index, x)
                eqxe.set_state(self.index, x)
                print(f"last_x={last_x}, x={x}")

        x = jnp.array([1., 2.])
        y = jnp.array([3., 4.])
        shape = x.shape
        ci = CacheInput(shape)
        ci(x)
        ci(y)
        ```
    """

    _obj: _IndexObj = static_field(repr=False)
    _version: int = static_field(repr=False)
    _state: PyTree[Array] = field(repr=False)
    inference: bool

    def __init__(self, inference: bool = False):
        """**Arguments:**

        - `inference`: If `True`, then the state can only be get, but not set. All
            stored states will looked up when crossing the JIT boundary -- rather than
            dynamically at runtime -- and treated as inputs to the XLA computation
            graph. This improves speed at runtime. This may be toggled with
            [`equinox.tree_inference`][].

        !!! warning

            You should not modify the `inference` flag whilst inside a JIT region. For
            example, the following will produced undefined behaviour:

            ```python
            @jax.jit
            def f(...):
                ...
                index = eqx.tree_at(lambda i: i.inference, index, True)
                ...
            ```
        """
        self._obj = _IndexObj()
        self._version = _FixedInt(-1)
        self._state = None
        self.inference = inference

    def unsafe_get(self):
        return _state_cache[self._obj]

    #
    # Wall of text, round one.
    # See get_state for the matching round two.
    #
    # So there's four scenarios we need to consider.
    # (inference=True vs inference=False) x (JIT vs non-JIT)
    #
    # First, `inference=True` and JIT.
    #
    # We update our local copy of the state each time we flatten.
    # In particular, this happens when crossing the JIT API boundary, which means that
    # in the (inference=True, JIT) case, the JIT'd region will see the latest state as
    # input.
    # (The reconstructed unflattened StateIndex within the JIT region will carry this
    # updated version through, regardless of what version our outside-JIT-region copy
    # carries.)
    #
    # We have a `new_version != self._version.value` check in case we un/flatten
    # whilst *within* a JIT region. In this case we want to continue using the same
    # state, which will be tracers wrt the input. (Not doing so would bake in the state
    # used, and moreover later updates to the state wouldn't be noticed as they
    # wouldn't trigger a re-JIT.)
    #
    # Note that `self._version` is a static field with a fixed hash.
    # It has to be a static field because anything that's a leaf is prone to being
    # overwritten with arbitrary data, (e.g. tree_at replaces all leaves with
    # integers). That is, it's a requirement of flattening and unflattening
    # functions that they not be dependent on the value/type of the leaves.
    # It can have a fixed hash when inference=True, because we look up the new state
    # in `tree_flatten` and as such don't need to re-JIT.
    #
    # Second: all the other cases.
    #
    # In these cases, updating the state is completely superfluous; we only have to
    # have an updated state in the (inference=True, JIT) case; this is discussed in
    # get_state below (follow up on the discussion there).
    #
    # However, we can't disable the superfluous (ever-so-mildly time-wasting) update,
    # as per the above requirement on flattening functions: adding an
    # `if self.inference` check would violate the conditions of a flattening function.
    # (Meanwhile there's no obvious way to check if we're flattening because we're
    # about to enter a JIT region. Most notably we are not yet in a JIT region, so
    # `isinstance(jnp.array(1) + 1, jax.core.Tracer)` won't work.)
    #
    # Regarding `self._version` have a fixed hash.
    #
    # Second, `inference=False` and JIT.
    # It must have a fixed hash when inference=False, because then we look up state
    # dynamically via `host_callback.call`, and mustn't trigger re-JITs then.
    #
    # Other miscellaneous things.
    # - Note the conversion of `new_state` from NumPy array to JAX array.
    # - Note that this array is non-sticky wrt device, so not specifying a device here
    #   is fine.
    # - Note that this happens during flatten, not unflatten, so that it happens
    #   outside a JIT region.
    #
    def tree_flatten(self):
        try:
            new_state, _, new_version = self.unsafe_get()
        except KeyError:
            new_state = None
            new_version = -1
        if new_version != self._version.value:
            # Make a copy of self so we can make our modifications.
            leaves, aux = super().tree_flatten()
            self = super().tree_unflatten(aux, leaves)
            # Not using `tree_at` because that goes via flattening and we'd get an
            # infinite loop.
            new_state = jax.tree_map(jnp.asarray, new_state)
            object.__setattr__(self, "_state", new_state)
            object.__setattr__(self, "_version", _FixedInt(new_version))

        # explicit self as we may have a different self
        return super(StateIndex, self).tree_flatten()

    #
    # What happens when someone passes in a StateIndex into a JIT region via a hashable
    # wrapper that ignores its contents to create the hash?
    # Well, in that case the user clearly wants to bake in the contents of that wrapper
    # and that's handled as per the discussion in `get_state`, below.
    #
    # But the more interesting scenario is what happens when someone passes in a
    # StateIndex via a hashable wrapper that examines its contents to handle hashing
    # and equality.
    # (For example as is sometimes done with hashable array wrappers.)
    # It's not obvious that the desired behaviour here is to bake things in, but that's
    # the only option available to us. So just to be sure we set `__hash__ = None`
    # to avoid potential bugs.
    #

    __hash__ = None


class _Leaf:  # Not a PyTree
    def __init__(self, value):
        self.value = value


# Monkey-patch the batching rule for host_callback.call to work with get_state and set_state.
_have_monkey_patched = False


def _monkey_patch():
    global _have_monkey_patched
    if not _have_monkey_patched:
        _have_monkey_patched = True

        _old_outside_call_impl = hcb.outside_call_p.impl
        _old_outside_call_translation_rule = xla._translations[hcb.outside_call_p]
        _old_outside_call_batching_rule = batching.primitive_batchers[
            hcb.outside_call_p
        ]

        #
        # Overwrite impl and abstract_eval:
        # Make `get_state` not actually pass `index._state` or `like` into the
        # callback. This means we don't need to wait for `like` to be computed at
        # runtime.
        #

        def _outside_call_impl(*arg_flat, arg_treedef, **params):
            leaves = [None] * arg_treedef.num_leaves
            call_type = type(jax.tree_unflatten(arg_treedef, leaves))
            # Not using isinstance for speed. (Questionable choice?)
            if call_type is _GetStateArg:
                arg = jax.tree_unflatten(arg_treedef, arg_flat)
                token_index = jax.tree_map(lambda _: jax.core.token, arg.index)
                token_like = jax.tree_map(lambda _: jax.core.token, arg.like)
                arg = tree_at(
                    lambda a: jax.tree_leaves((a.index, a.like)),
                    arg,
                    jax.tree_leaves((token_index, token_like)),
                )
                arg_flat = jax.tree_leaves(arg)
            return _old_outside_call_impl(*arg_flat, arg_treedef=arg_treedef, **params)

        def _outside_call_translation_rule(ctx, avals_in, *args, arg_treedef, **kwargs):
            leaves = [None] * arg_treedef.num_leaves
            call_type = type(jax.tree_unflatten(arg_treedef, leaves))
            if call_type is _GetStateArg:
                arg_flat = avals_in[:-2]
                extra_tokens = avals_in[-2:]
                arg = jax.tree_unflatten(arg_treedef, arg_flat)
                token_index = jax.tree_map(lambda _: jax.core.abstract_token, arg.index)
                token_like = jax.tree_map(lambda _: jax.core.abstract_token, arg.like)
                arg = tree_at(
                    lambda a: jax.tree_leaves((a.index, a.like)),
                    arg,
                    jax.tree_leaves((token_index, token_like)),
                )
                arg_flat = jax.tree_leaves(arg)
                avals_in = arg_flat + extra_tokens
            return _old_outside_call_translation_rule(
                ctx, avals_in, *args, arg_treedef=arg_treedef, **kwargs
            )

        #
        # Overwrite batching:
        # Allows us to use get_state and set_state inside vmap.
        # (Not implemented for general `host_callback.call`s.)
        #

        def _target_batch_axes(batch_axes_flat, arg_treedef, target):
            batch_axes_leaves_flat = [_Leaf(b) for b in batch_axes_flat]
            batch_axes_leaves_tree = jax.tree_unflatten(
                arg_treedef, batch_axes_leaves_flat
            )
            batch_axes_target_leaves_tree = getattr(batch_axes_leaves_tree, target)
            batch_axes_target_leaves_flat = jax.tree_leaves(
                batch_axes_target_leaves_tree
            )
            batch_axes_target_flat = [b.value for b in batch_axes_target_leaves_flat]
            return batch_axes_target_flat

        def _outside_call_batching_rule(
            arg_flat, batch_axes_flat, *, arg_treedef, result_treedef, **params
        ):
            arg = jax.tree_unflatten(arg_treedef, arg_flat)
            if type(arg) is _GetStateArg:
                batch_axes_like_flat = _target_batch_axes(
                    batch_axes_flat, arg_treedef, "like"
                )
                state = _get_state(
                    arg.index, arg.like, arg.batch_axes + batch_axes_like_flat
                )
                state_leaves, state_treedef = jax.tree_flatten(state)
                assert state_treedef == result_treedef
                assert len(state_leaves) == len(batch_axes_like_flat)
                return state_leaves, batch_axes_like_flat
            elif type(arg) is _SetStateArg:
                batch_axes_state_flat = _target_batch_axes(
                    batch_axes_flat, arg_treedef, "state"
                )
                _set_state(arg.index, arg.state, arg.batch_axes + batch_axes_state_flat)
                return (), ()
            else:
                return _old_outside_call_batching_rule(
                    arg_flat,
                    batch_axes_flat,
                    arg_treedef=arg_treedef,
                    result_treedef=result_treedef,
                    **params
                )

        hcb.outside_call_p.def_impl(_outside_call_impl)
        batching.primitive_batchers[hcb.outside_call_p] = _outside_call_batching_rule
        xla.register_translation(hcb.outside_call_p, _outside_call_translation_rule)


def _batchify_impl(*flat, treedef, like_batch_axes, current_batch_axes):
    if current_batch_axes != like_batch_axes:
        raise RuntimeError("`like` and the saved state have different batch axes")
    state, like = jax.tree_unflatten(treedef, flat)
    return jax.tree_leaves(state)


def _batchify_abstract_eval(*flat, treedef, like_batch_axes, current_batch_axes):
    state, like = jax.tree_unflatten(treedef, flat)
    like_avals = jax.tree_map(lambda l: jax.ShapedArray(l.shape, l.dtype), like)
    return jax.tree_leaves(like_avals)


def _batchify_batching_rule(
    flat, batch_axes_flat, *, treedef, like_batch_axes, current_batch_axes
):
    state, like = jax.tree_unflatten(treedef, flat)
    batch_axes_flat = [_Leaf(b) for b in batch_axes_flat]
    state_batch_axis, like_batch_axis = jax.tree_unflatten(treedef, batch_axes_flat)
    for b in jax.tree_leaves(state_batch_axis):
        assert b.value is batching.not_mapped
    like_batch_axis = jax.tree_leaves(like_batch_axis)
    like_batch_axis = [b.value for b in like_batch_axis]
    return (
        _batchify_p.bind(
            *flat,
            treedef=treedef,
            like_batch_axes=like_batch_axes + like_batch_axis,
            current_batch_axes=current_batch_axes
        ),
        like_batch_axis,
    )


_batchify_p = jax.core.Primitive("batchify")
_batchify_p.multiple_results = True
_batchify_p.def_impl(_batchify_impl)
_batchify_p.def_abstract_eval(_batchify_abstract_eval)
batching.primitive_batchers[_batchify_p] = _batchify_batching_rule
# `xla.lower_fun` is getting removed in later JAX versions.
# See https://github.com/patrick-kidger/diffrax/pull/91
if hasattr(xla, "lower_fun"):
    xla.register_translation(
        _batchify_p,
        xla.lower_fun(_batchify_impl, multiple_results=True, new_style=True),
    )
mlir.register_lowering(
    _batchify_p, mlir.lower_fun(_batchify_impl, multiple_results=True)
)


class _GetStateArg(Module):
    index: StateIndex
    like: PyTree[Array]
    batch_axes: Tuple[int] = static_field()


def _get_state_hcb(arg: _GetStateArg) -> PyTree:
    index = arg.index
    batch_axes = arg.batch_axes
    try:
        current_state, current_batch_axes, _ = _state_cache[index._obj]
    except KeyError as e:
        raise RuntimeError("Cannot get state before it has been set") from e
    if current_batch_axes != batch_axes:
        raise RuntimeError("`like` and the saved state have different batch axes")
    return current_state


def _get_state(
    index: StateIndex, like: PyTree[Array], batch_axes: Tuple[int]
) -> PyTree:
    arg = _GetStateArg(index, like, batch_axes)
    like_shape = jax.eval_shape(lambda: like)
    # Will raise an error if `like_shape` does not match the result.
    return hcb.call(_get_state_hcb, arg, result_shape=like_shape)


def get_state(index: StateIndex, like: PyTree[Array]) -> PyTree[Array]:
    """Get some previously saved state.

    **Arguments:**

    - `index`: The index of the state to look up. Should be an instance of
        [`equinox.experimental.StateIndex`][].
    - `like`: A PyTree of JAX arrays of the same shape, dtype, PyTree structure, and
        batch axes as the state being looked up.

    **Returns:**

    Whatever the previously saved state is.

    **Raises:**

    A `TypeError` at trace time if `like` is not a PyTree of JAX arrays.

    A `RuntimeError` at run time if `like` is not of the same shape, dtype, PyTree
    structure, and batch axes as the retrieved value.

    A `RuntimeError` at run time if no state has previously been saved with this
    `index`.

    !!! warning

        This means that your operation will no longer be a pure function.
    """
    if any(not is_array(x) for x in jax.tree_leaves(like)):
        raise TypeError("`like` must be a PyTree containing only JAX arrays")

    #
    # Wall of text, round two.
    # See StateIndex.tree_flatten for round one.
    #
    # Once again we have to consider all four possibilities
    # (inference=True vs inference=False) x (JIT vs non-JIT)
    #
    # First, the inference=False x (JIT vs non-JIT) cases.
    #
    # In this case, we just go and look up the latest state dynamically using a
    # `host_callback.call`. Slow, but correct, and exactly what we want, because we
    # might update the state through another `host_callback.call`.
    #
    # Next, the inference=True x JIT case. This has two sub-cases.
    #
    # First, `index` was passed in to the JIT region via closure. (Or, equivalently,
    # snuck in by wrapping `index` in something non-pytreeable, that misses the
    # `tree_flatten`.) This is pretty uncommon; JIT'd functions are usually assumed to
    # be pure and will always capture anything closed-over as a static variable. In
    # this case `index` may not have hit `tree_flatten` above and may be out of date.
    # In any case, the desired behaviour is to bake in whatever the current state is.
    # This is the reason for the `version == index._version.value` check.
    # If `index._version` is out-of-date then we'll go and get the version directly
    # from the cache. (If it's up to date then `index._state` is just the same state.)
    # In either case, it's just like baking in a closed-over (or snuck-in) JAX array.
    # Why not just always get the up-to-date version from the cache; why have the
    # `version == index._version.value` check at all? Read on for the next sub-case.
    #
    # Second, `index` was passed it to the JIT region directly as an argument. (Note
    # that difference pieces of it may be passed as static/dynamic, but this doesn't
    # matter.) In this case `tree_flatten` was hit above, and our `._state` will be
    # up-to-date. (And it cannot become out-of-date because we disable `set_state` when
    # `inference=True`.) So we hit the `version == index._version.value` branch; this
    # is important because we want to use the traced version of JAX arrays we obtained
    # on entering the JIT region; using the version retrived from the cache would
    # produce ConcreteArrays instead, that get baked in.
    #
    # Finally, the inference=True x non-JIT case. In this case we may or may not hit
    # the `version == index._version.value` check, but either way we obtain the latest
    # state from the cache. There's no discussion on baking in etc., because we're
    # not in a JIT region.
    #
    if index.inference:
        try:
            current_state, current_batch_axes, current_version = _state_cache[
                index._obj
            ]
        except KeyError as e:
            raise RuntimeError("Cannot get state before it has been set") from e
        if current_version == index._version.value:
            state = index._state
        else:
            state = jax.tree_map(jnp.asarray, current_state)
        _treedef = jax.tree_structure(state)
        if _treedef != jax.tree_structure(state):
            raise RuntimeError(
                "`like` has different PyTree structure to the stored state"
            )
        flat, treedef = jax.tree_flatten((state, like))
        out = _batchify_p.bind(
            *flat,
            treedef=treedef,
            like_batch_axes=[],
            current_batch_axes=current_batch_axes
        )
        return jax.tree_unflatten(_treedef, out)
    else:
        _monkey_patch()
        return _get_state(index, like, [])


class _SetStateArg(Module):
    index: StateIndex
    state: PyTree[Array]
    batch_axes: Tuple[int] = static_field()


def _set_state_hcb(arg: _SetStateArg) -> None:
    # Note that these checks cannot happen inside `set_state` as we have to consider
    # the possibility in which `set_state` is traced into a jaxpr and then transformed
    # (e.g. vmap'd.)
    # In principle it should be possible to perform these checks at compile time but it
    # would likely require us to create our own primitive? Which in turn wouldn't play
    # well with all the custom primitive handling that experimental.host_callback does?
    index = arg.index
    state = arg.state
    batch_axes = arg.batch_axes
    try:
        current_state, current_batch_axes, current_version = _state_cache[index._obj]
    except KeyError:
        current_version = -1
    else:
        current_state_shape = jax.eval_shape(lambda: current_state)
        state_shape = jax.eval_shape(lambda: state)
        if current_state_shape != state_shape:
            raise RuntimeError(
                "New state and old state have different shape, dtype, or PyTree structure"
            )
        if current_batch_axes != batch_axes:
            raise RuntimeError("New state and old state have different batch axes")
    _state_cache[index._obj] = (state, batch_axes, current_version + 1)


def set_state(index: StateIndex, state: PyTree[Array]) -> None:
    """Save a PyTree of JAX arrays as a side-effect.

    **Arguments:**

    - `index`: A key under which to save the state. Should be an instance of
        [`equinox.experimental.StateIndex`][].
    - `state`: An PyTree of JAX arrays to save.

    **Returns:**

    `None`.

    **Raises:**

    A `TypeError` at trace time if `state` is not a PyTree of JAX arrays.

    A `RuntimeError` at run time if this `index` has previously been used to save a
    `state` with a different shape, dtype, PyTree structure, or batch axes.

    A `RuntimeError` at trace time if `index.inference` is truthy.

    !!! info

        The same `index` can be used multiple times, to overwrite a previously saved
        value. The new and old `state` must both have the same PyTree structure, however.

    !!! warning

        Note that gradient information in `state` will not preserved.

    !!! warning

        This means that your operation will no longer be a pure function. Moreover note
        that the saving-as-a-side-effect may occur even when `set_state` is wrapped in
        `lax.cond` etc. (As e.g. under `vmap` then `lax.cond` is transformed into
        `lax.select`.)
    """
    if index.inference:
        # Important to make sure that arrays passed in the (inference=True, JIT) case
        # don't become invalidated.
        # Moreover the only way to set state is to use a `host_callback.call`, and
        # avoiding that is the purpose of `inference=True`.
        # We could technically allow this in the (inference=False, non-JIT) case, but
        # it's better to be consistent between JIT and non-JIT.
        raise RuntimeError("Cannot use `set_state` during inference.")
    if any(not is_array(x) for x in jax.tree_leaves(state)):
        raise TypeError("`state` must be a PyTree containing only JAX arrays")
    _monkey_patch()
    state = jax.tree_map(lax.stop_gradient, state)
    _set_state(index, state, [])


def _set_state(index: StateIndex, state: PyTree[Array], batch_axes: Tuple[int]) -> None:
    arg = _SetStateArg(index, state, batch_axes)
    hcb.call(_set_state_hcb, arg)
