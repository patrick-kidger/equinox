import itertools as it
from typing import Any, TYPE_CHECKING, Union

import jax
import jax.extend.core
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Bool

from ..._filters import combine, is_array, partition
from ..._module import field, Module
from ..._pretty_print import tree_pformat
from ..._tree import tree_at, tree_equal
from ..._unvmap import unvmap_any
from .._nontraceable import nonbatchable
from .._primitive import create_vprim


def _select_if_vmap_impl(pred, x, y):
    # Not including the following, as it destroys performance on the GPU: it seems like
    # a copy of `x` is being made.
    #
    # msg = (
    #     "Internal error in Equinox. Please report a bug at "
    #     "https://github.com/patrick-kidger/equinox."
    # )
    # x = error_if(x, jnp.invert(pred), msg)
    return x


def _select_if_vmap_abstract(pred, x, y):
    return x


def _select_if_vmap_jvp(primals, tangents):
    pred, x, y = primals
    _, tx, ty = tangents
    assert x.shape == tx.aval.shape
    assert x.dtype == tx.aval.dtype
    assert y.shape == ty.aval.shape
    assert y.dtype == ty.aval.dtype
    out = _select_if_vmap(pred, x, y, makes_false_steps=False)
    if type(tx) is ad.Zero and type(ty) is ad.Zero:
        t_out = tx
    else:
        if type(tx) is ad.Zero:
            tx = jnp.zeros(tx.aval.shape, tx.aval.dtype)  # pyright: ignore
        if type(ty) is ad.Zero:
            ty = jnp.zeros(ty.aval.shape, ty.aval.dtype)  # pyright: ignore
        t_out = _select_if_vmap(pred, tx, ty, makes_false_steps=False)
    return out, t_out


def _select_if_vmap_transpose(ct, pred, x, y):
    assert not ad.is_undefined_primal(pred)
    assert ct.shape == x.aval.shape
    assert ct.dtype == x.aval.dtype
    assert ct.shape == y.aval.shape
    assert ct.dtype == y.aval.dtype
    if type(ct) is ad.Zero:
        ct_x = None
        ct_y = None
    else:
        zero = jnp.zeros(ct.shape, ct.dtype)
        if ad.is_undefined_primal(x):
            ct_x = _select_if_vmap(pred, ct, zero, makes_false_steps=False)
        else:
            ct_x = None
        if ad.is_undefined_primal(y):
            ct_y = _select_if_vmap(pred, zero, ct, makes_false_steps=False)
        else:
            ct_y = None
    return [None, ct_x, ct_y]


def _select_if_vmap_batch(axis_size, axis_name, trace, inputs, batch_axes):
    del axis_name, trace
    pred, x, y = inputs
    bp, bx, by = batch_axes
    if bp is batching.not_mapped:
        if bx is batching.not_mapped:
            if by is batching.not_mapped:
                out_axis = None
            else:
                x = jnp.broadcast_to(x, (axis_size,) + x.shape)
                y = jnp.moveaxis(y, by, 0)
                out_axis = 0
        else:
            if by is batching.not_mapped:
                x = jnp.moveaxis(x, bx, 0)
                y = jnp.broadcast_to(y, (axis_size,) + y.shape)
                out_axis = 0
            else:
                x = jnp.moveaxis(x, bx, 0)
                y = jnp.moveaxis(y, by, 0)
                out_axis = 0
        out = _select_if_vmap(pred, x, y, makes_false_steps=False)
    else:
        out = jax.vmap(lax.select, in_axes=(bp, bx, by))(pred, x, y)
        out_axis = 0
    return out, out_axis


select_if_vmap_p = jax.extend.core.Primitive("select_if_vmap")
select_if_vmap_p.def_impl(_select_if_vmap_impl)
select_if_vmap_p.def_abstract_eval(_select_if_vmap_abstract)
ad.primitive_jvps[select_if_vmap_p] = _select_if_vmap_jvp
ad.primitive_transposes[select_if_vmap_p] = _select_if_vmap_transpose
batching.axis_primitive_batchers[select_if_vmap_p] = _select_if_vmap_batch
mlir.register_lowering(
    select_if_vmap_p, mlir.lower_fun(_select_if_vmap_impl, multiple_results=False)
)


# We want to insert `lax.select`s to avoid updating any batch elements in the loop that
# have a False predicate. (But the loop is still going whilst other batch elements have
# a True predicate). However, if we have no vmap at all, then we can be slightly more
# efficient: don't introduce a select at all.
def _select_if_vmap(pred, x, y, makes_false_steps):
    """As `lax.select(pred, x, y)` if `pred` is vmap'd. Not-vmap'd `pred` are assumed to
    be `True`, so that in this case `x` is returned unconditionally.
    """
    if makes_false_steps:
        return lax.select(pred, x, y)
    else:
        pred = fixed_asarray(pred)
        assert pred.shape == ()
        assert pred.dtype == jnp.bool_
        x = fixed_asarray(x)
        y = fixed_asarray(y)
        assert x.shape == y.shape
        assert x.dtype == y.dtype
        return select_if_vmap_p.bind(pred, x, y)


def _maybe_set_impl(
    pred, xs, x, *i_dynamic_leaves, i_static, i_treedef, kwargs, makes_false_steps
):
    i = combine(i_static, jtu.tree_unflatten(i_treedef, i_dynamic_leaves))
    x = _select_if_vmap(pred, x, xs.at[i].get(**kwargs), makes_false_steps)
    return [xs.at[i].set(x, **kwargs)]


def _maybe_set_abstract(
    pred, xs, x, *i_dynamic_leaves, i_static, i_treedef, kwargs, makes_false_steps
):
    return [xs]


def _maybe_set_jvp(
    primals, tangents, *, i_static, i_treedef, kwargs, makes_false_steps
):
    pred, xs, x, *i_dynamic_leaves = primals
    _, t_xs, t_x, *_ = tangents
    i = combine(i_static, jtu.tree_unflatten(i_treedef, i_dynamic_leaves))
    out = _maybe_set(pred, xs, x, i, kwargs=kwargs, makes_false_steps=makes_false_steps)
    if type(t_x) is ad.Zero and type(t_xs) is ad.Zero:
        t_out = t_xs
    else:
        if type(t_x) is ad.Zero:
            t_x = jnp.zeros(t_x.aval.shape, t_x.aval.dtype)  # pyright: ignore
        if type(t_xs) is ad.Zero:
            t_xs = jnp.zeros(t_xs.aval.shape, t_xs.aval.dtype)  # pyright: ignore
        t_out = _maybe_set(
            pred, t_xs, t_x, i, kwargs=kwargs, makes_false_steps=makes_false_steps
        )
    return [out], [t_out]


def _maybe_set_transpose(
    ct_out,
    pred,
    xs,
    x,
    *i_dynamic_leaves,
    i_static,
    i_treedef,
    kwargs,
    makes_false_steps,
):
    assert not ad.is_undefined_primal(pred)
    for z in i_dynamic_leaves:
        assert not ad.is_undefined_primal(z)
    i = combine(i_static, jtu.tree_unflatten(i_treedef, i_dynamic_leaves))
    [ct_out] = ct_out
    if ad.is_undefined_primal(xs):
        # Not updating a zero! `_Buffer`s are write-once to each location, so when
        # transposed they are read-once from each location, meaning if we read from `i`
        # now then we will never do so again, so we can skip placing a zero there.
        #
        # Besides just being a nice efficiency gain: this is actually important for
        # working around b/288798733, in which having extra dynamic_update_slices causes
        # operations to happen out-of-place.
        ct_xs = ct_out
    else:
        ct_xs = None
    if ad.is_undefined_primal(x):
        if type(ct_out) is ad.Zero:
            ct_x = None
        else:
            ct_x = ct_out.at[i].get(**kwargs)
            ct_x = _select_if_vmap(pred, ct_x, jnp.zeros_like(ct_x), makes_false_steps)
    else:
        ct_x = None
    return [None, ct_xs, ct_x] + [None] * len(i_dynamic_leaves)


maybe_set_p = create_vprim(
    "maybe_set",
    _maybe_set_impl,
    _maybe_set_abstract,
    _maybe_set_jvp,
    _maybe_set_transpose,
)


# This is a carefully optimised routine, that relies on the special behaviour exhibited
# by `_Buffer`.
# The main one is fact that `_Buffer`s are write-once to each location, so they
# are read-once to each location when transposed. This means we can skip placing zeros
# in the cotangent of `xs`. This reduces the number of in-place updates on the backward
# pass from 2 to 0. This ends up being really important for efficiency, due to
# b/288798733.
# Second, the fact that unbatched `pred` are necessarily always True (due to being
# used inside a while loop) means that we use `_select_if_vmap` over simply
# `lax.select`.
def _maybe_set(pred, xs, x, i, *, kwargs, makes_false_steps):
    """As `lax.select(pred, xs.at[i].set(x, **kwargs), xs)`, under the assumption that
    every location `i` is written to at most once. (So that we can have a more efficient
    transpose rule. Also assumes that non-vmap'd `pred` is always `True`.)
    """
    if jnp.shape(pred) != () or jnp.result_type(pred) != jnp.bool_:
        raise ValueError("predicate must be a boolean scalar.")
    dtype = jnp.result_type(x, xs)
    if dtype != jnp.result_type(xs):
        raise ValueError(
            "When doing `buffer.at[i].set(value)`, then `value` must have a dtype that "
            "can be promoted to the same dtype as `buffer`."
        )
    x = fixed_asarray(x).astype(dtype)
    x = jnp.broadcast_to(x, jax.eval_shape(lambda: xs[i]).shape)
    i_dynamic, i_static = partition(i, is_array)
    i_dynamic_leaves, i_treedef = jtu.tree_flatten(i_dynamic)
    [out] = maybe_set_p.bind(
        pred,
        xs,
        x,
        *i_dynamic_leaves,
        i_static=i_static,
        i_treedef=i_treedef,
        kwargs=kwargs,
        makes_false_steps=makes_false_steps,
    )
    return out


if TYPE_CHECKING:
    from typing import Annotated, TypeVar
    from typing_extensions import TypeAlias

    _T = TypeVar("_T")
    MaybeBuffer: TypeAlias = Annotated[_T, "MaybeBuffer"]
else:

    class _MetaBufferItem(type):
        def __instancecheck__(cls, instance):
            annotation = cls.annotation
            while type(instance) is _Buffer:
                instance = instance._array
            return isinstance(instance, annotation)

    class MaybeBuffer:
        def __class_getitem__(cls, item):
            class _BufferItem(metaclass=_MetaBufferItem):
                annotation = item

            return _BufferItem


class _Buffer(Module):
    # annotation removed because beartype can't handle the forward reference.
    _array: Any  # Union[Shaped[Array, "..."], _Buffer]
    _pred: Bool[Array, ""]
    _tag: object = field(static=True)
    _makes_false_steps: bool = field(static=True)

    def __getitem__(self, item):
        return self._array[item]

    def _op(self, pred, item, x, op, kwargs, makes_false_steps):
        pred = pred & self._pred
        if isinstance(self._array, _Buffer):
            array = self._array._op(pred, item, x, op, kwargs, makes_false_steps)
        else:
            array = op(
                pred,
                self._array,
                x,
                item,
                kwargs=kwargs,
                makes_false_steps=makes_false_steps,
            )
        return _Buffer(array, self._pred, self._tag, self._makes_false_steps)

    @property
    def at(self):
        return _BufferAt(self, self._makes_false_steps)

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
    _makes_false_steps: bool = field(static=True)

    def __getitem__(self, item):
        return _BufferItem(self._buffer, item, self._makes_false_steps)


class _BufferItem(Module):
    _buffer: _Buffer
    _item: Any
    _makes_false_steps: bool = field(static=True)

    def set(self, x, *, pred=True, **kwargs):
        if pred is True:
            makes_false_steps = self._makes_false_steps
        else:
            makes_false_steps = True
        return self._buffer._op(
            pred, self._item, x, _maybe_set, kwargs, makes_false_steps
        )


def buffer_at_set(buffer: Union[Array, _Buffer], item, x, *, pred=True, **kwargs):
    """As `buffer.at[...].set(...)`, and supports the `pred` argument even if it is an
    array.

    This is primarily useful when calling a buffer-using cond or body function outside
    of a while loop, for any reason.
    """
    if isinstance(buffer, _Buffer):
        return buffer.at[item].set(x, pred=pred, **kwargs)
    else:
        if pred is not True:
            x = jnp.where(pred, x, buffer.at[item].get(**kwargs))
        return buffer.at[item].set(x)


def _is_buffer(x):
    return isinstance(x, _Buffer)


def _unwrap_buffers(x):
    while _is_buffer(x):
        x = x._array
    return x


# Work around JAX issue #15676.
# This issue arises with both JVP tracing and make_jaxpr tracing. The former can be
# handled with a custom_jvp, but the latter cannot. So we need to just call `jnp.array`
# instead.
fixed_asarray = jnp.array


def common_rewrite(cond_fun, body_fun, init_val, max_steps, buffers, makes_false_steps):
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
        new_buffers = lambda is_leaf: lambda val: []
    else:

        def new_buffers(is_leaf):
            def new_buffers2(val):
                _, _, _, val = val  # ignore step and pred
                bufs = buffers(val)
                # Ignore the ._pred attribute of nested buffers.
                # This is kind of a hack: we're special-casing support for nested
                # buffers so that end users don't have to do the ._array lookup
                # themselves.
                tree = jtu.tree_map(_unwrap_buffers, bufs, is_leaf=_is_buffer)
                return jtu.tree_leaves(tree, is_leaf=is_leaf)

            return new_buffers2

    def _wrap_buffers(val, pred, tag):
        def wrap_buffer(leaf):
            if not is_array(leaf):
                raise ValueError("Only arrays can be treated as buffers.")
            return _Buffer(leaf, pred, tag, makes_false_steps)

        _, _, _, buffer_val = tree_at(
            new_buffers(None), (None, None, None, val), replace_fn=wrap_buffer
        )
        return buffer_val

    def new_cond_fun(val):
        step, pred, prev_pred, val = val
        del pred
        # Note that this is actually recomputing `pred`!
        # For some reason this is actually a minor performance optimisation.
        # See https://github.com/patrick-kidger/diffrax/issues/274
        buffer_val = _wrap_buffers(val, prev_pred, None)
        out = unvmap_any(cond_fun(buffer_val))
        if max_steps is not None:
            if type(max_steps) is not int:
                raise ValueError("`max_steps` must be a Python integer")
            out = out & (step < max_steps)
        # Need to allow being constant across the batch. This seems to arise in some
        # edge case when using `bounded_while_loop`, in which:
        # - `lax.scan` saves a copy of its state (in particular `step`) for use in the
        #    backward pass;
        # - for some reason the `jax.checkpoint` causes such state to pick up a spurious
        #   batch tracer.
        # See:
        # https://github.com/patrick-kidger/optimistix/issues/48#issuecomment-2009221739
        return nonbatchable(
            out, name="`equinox.internal.while_loop`", allow_constant_across_batch=True
        )

    def new_body_fun(val):
        tag = object()

        def is_our_buffer(node):
            return isinstance(node, _Buffer) and node._tag is tag

        def unwrap_and_select(leaf, leaf2):
            if is_our_buffer(leaf):
                assert is_our_buffer(leaf2)
                # sanity check that this is the lowest buffer, i.e. when nesting
                # multiple checkpointed_while_loops.
                assert is_array(leaf._array)
                assert is_array(leaf2._array)
                return leaf2._array
            else:
                return _select_if_vmap(pred, leaf2, leaf, makes_false_steps)

        step, pred, _, val = val
        buffer_val = _wrap_buffers(val, pred, tag)
        buffer_val2 = body_fun(buffer_val)
        # Needed to work with `disable_jit`, as then we lose the automatic
        # ArrayLike->Array cast provided by JAX's while loops.
        # The input `val` is already cast to Array below, so this matches that.
        buffer_val2 = jtu.tree_map(fixed_asarray, buffer_val2)
        # Strip `.named_shape`; c.f. Diffrax issue #246
        struct = jax.eval_shape(lambda: buffer_val)
        struct2 = jax.eval_shape(lambda: buffer_val2)
        struct = jtu.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), struct)
        struct2 = jtu.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), struct2
        )
        if not tree_equal(struct, struct2):
            string = tree_pformat(struct, struct_as_array=True)
            string2 = tree_pformat(struct2, struct_as_array=True)
            out = []
            for line, line2 in it.zip_longest(
                string.split("\n"), string2.split("\n"), fillvalue=""
            ):
                if line == line2:
                    out.append("  " + line)
                else:
                    out.append("- " + line)
                    out.append("+ " + line2)
            out = "\n".join(out)
            raise ValueError(
                "`body_fun` must have the same input and output structure. Difference "
                "is:\n" + out
            )
        val2 = jtu.tree_map(
            unwrap_and_select, buffer_val, buffer_val2, is_leaf=is_our_buffer
        )
        step2 = step + 1
        pred2 = pred & cond_fun(buffer_val2)
        return step2, pred2, pred, val2

    init_val = jtu.tree_map(fixed_asarray, init_val)
    init_buffer_val = _wrap_buffers(init_val, jnp.array(True), None)
    init_pred = jnp.array(cond_fun(init_buffer_val))
    new_init_val = (jnp.array(0), init_pred, jnp.array(True), init_val)

    return new_cond_fun, new_body_fun, new_init_val, new_buffers
