from typing import Any, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Bool, Shaped

from ...filters import is_array
from ...module import Module, static_field
from ...tree import tree_at, tree_equal
from ..unvmap import unvmap_any


class _Buffer(Module):
    _array: Union[Shaped[Array, "..."], "_Buffer"]
    _pred: Bool[Array, ""]
    _tag: object = static_field()
    _readable: bool = static_field()

    def __getitem__(self, item):
        if self._readable:
            return self._array[item]
        else:
            raise ValueError("Cannot read from write-only buffer inside loop.")

    def _set(self, pred, item, x):
        pred = pred & self._pred
        if isinstance(self._array, _Buffer):
            array = self._array._set(pred, item, x)
        else:
            old_x = self._array[item]
            x = jnp.where(pred, x, old_x)
            array = self._array.at[item].set(x)
        return _Buffer(array, self._pred, self._tag, self._readable)

    @property
    def at(self):
        return _BufferAt(self)

    @property
    def shape(self):
        return self._array.shape

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def size(self):
        return self._array.size


class _BufferAt(Module):
    _buffer: _Buffer

    def __getitem__(self, item):
        return _BufferItem(self._buffer, item)


class _BufferItem(Module):
    _buffer: _Buffer
    _item: Any

    def set(self, x, *, pred=True):
        return self._buffer._set(pred, self._item, x)


def _is_buffer(x):
    return isinstance(x, _Buffer)


def _unwrap_buffers(x):
    while _is_buffer(x):
        x = x._array
    return x


def common_rewrite(cond_fun, body_fun, init_val, max_steps, buffers, readable):
    """Handles:

    - Efficient in-place updates;
    - Efficient/correct vmap;
    - max_steps.

    The efficient in-place updates are done using buffers. These mean that we get to
    interchange `select` (from vmap) and `scatter` (from the in-place update) so that
    they happen in the efficient order. Specifically for `checkpointed_while_loop`, it
    also means we don't need to save our buffers in the list of checkpoints.

    The efficient vmap is done by always having an unvmap'd return value from
    `cond_fun`: essentially we reduce from the vmap'd case to the no-vmap case here.
    This ensures that (a) `bounded_while_loop` exhibits early-exit, as our `lax.cond`s
    don't get turned into `lax.select`s, and (b) `checkpointed_while_loop` has the
    correct logic on the backward pass. (As all checkpoints need to be in lockstep
    across the batch.)
    """

    if buffers is None:
        new_buffers = lambda _: lambda _: ()
    else:

        def new_buffers(is_leaf):
            def new_buffers2(val):
                _, _, val = val  # ignore step and pred
                bufs = buffers(val)
                # Ignore the ._pred attribute of nested buffers.
                # This is kind of a hack: we're special-casing support for nested
                # buffers so that end users don't have to do the ._array lookup
                # themselves.
                tree = jtu.tree_map(_unwrap_buffers, bufs, is_leaf=_is_buffer)
                return jtu.tree_leaves(tree, is_leaf=is_leaf)

            return new_buffers2

    def new_cond_fun(val):
        _, pred, _ = val
        return unvmap_any(pred)

    def new_body_fun(val):
        tag = object()

        def is_our_buffer(node):
            return isinstance(node, _Buffer) and node._tag is tag

        def wrap_buffer(leaf):
            if not is_array(leaf):
                raise ValueError("Only arrays can be treated as buffers.")
            return _Buffer(leaf, pred, tag, readable)

        def unwrap_and_select(leaf, leaf2):
            if is_our_buffer(leaf):
                assert is_our_buffer(leaf2)
                # sanity check that this is the lowest buffer, i.e. when nesting
                # multiple checkpointed_while_loops.
                assert is_array(leaf._array)
                assert is_array(leaf2._array)
                return leaf2._array
            else:
                return lax.select(pred, leaf2, leaf)

        step, pred, val = val
        _, _, buffer_val = tree_at(
            new_buffers(None), (None, None, val), replace_fn=wrap_buffer
        )
        buffer_val2 = body_fun(buffer_val)
        if not tree_equal(
            jax.eval_shape(lambda: buffer_val), jax.eval_shape(lambda: buffer_val2)
        ):
            raise ValueError("`body_fun` must have the same input and output structure")
        val2 = jtu.tree_map(
            unwrap_and_select, buffer_val, buffer_val2, is_leaf=is_our_buffer
        )
        step2 = step + 1
        pred2 = pred & cond_fun(buffer_val2)
        if max_steps is not None:
            if type(max_steps) is not int:
                raise ValueError("`max_steps` must be a Python integer")
            pred2 = pred2 & (step2 < max_steps)
        return step2, pred2, val2

    new_init_val = (jnp.asarray(0), jnp.asarray(cond_fun(init_val)), init_val)

    return new_cond_fun, new_body_fun, new_init_val, new_buffers
