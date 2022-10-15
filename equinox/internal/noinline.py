import functools as ft
from typing import Callable

import jax
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.numpy as jnp
import jax.tree_util as jtu

from ..compile_utils import hashable_combine, hashable_partition
from ..eval_shape import filter_eval_shape
from ..filters import combine, partition
from ..module import Module, module_update_wrapper
from .primitive import (
    filter_primitive_batching,
    filter_primitive_bind,
    filter_primitive_def,
    filter_primitive_jvp,
    filter_primitive_transpose,
)


def _is_shapedarray(x):
    return isinstance(x, jax.core.ShapedArray)


def _to_shapedarray(x):
    return jax.core.ShapedArray(x.shape, x.dtype)


def _call(fn, *args, **kwargs):
    return fn(*args, **kwargs)


@ft.lru_cache(maxsize=128)
def _cache_filter_eval_shape(key):
    flat_dynamic, treedef_dynamic, static = key
    flat_dynamic = [jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in flat_dynamic]
    dynamic = jtu.tree_unflatten(treedef_dynamic, flat_dynamic)
    fn, args, kwargs = hashable_combine(dynamic, static)
    return filter_eval_shape(_call, fn, *args, **kwargs)


def _is_zero(x):
    return type(x) is ad.Zero


def _materialise_zero(t):
    if _is_zero(t):
        return jnp.zeros(t.aval.shape, t.aval.dtype)
    else:
        return t


def _is_undefined(x):
    return type(x) is ad.UndefinedPrimal


def _is_none(x):
    return x is None


def _move_to_front(input, batch_axis):
    if batch_axis is batching.not_mapped:
        return input
    else:
        return jnp.swapaxes(input, 0, batch_axis)


@ft.lru_cache(maxsize=None)
def _get_callback(treedef, static_leaves, static_treedef):
    static = jtu.tree_unflatten(static_treedef, static_leaves)

    @jax.jit
    def _callback(*flat):
        fn, args, kwargs = combine(jtu.tree_unflatten(treedef, flat), static)
        out = fn(*args, **kwargs)
        return tuple(jtu.tree_leaves(out))

    return _callback


@filter_primitive_def
def _noinline_impl(fn, args, kwargs):
    return fn(*args, **kwargs)


@filter_primitive_def
def _noinline_abstract(fn, args, kwargs):
    dynamic, static = hashable_partition((fn, args, kwargs), _is_shapedarray)
    flat_dynamic, treedef_dynamic = jtu.tree_flatten(dynamic)
    key = tuple((x.shape, x.dtype) for x in flat_dynamic), treedef_dynamic, static
    out_struct = _cache_filter_eval_shape(key)
    return jtu.tree_map(_to_shapedarray, out_struct)


@filter_primitive_jvp
def _noinline_jvp(primals, tangents):
    # TODO: add custom partial-eval rule to avoid the double-noinline?
    fn, args, kwargs = primals
    tangents = jtu.tree_map(_materialise_zero, tangents, is_leaf=_is_zero)
    primal_outs = noinline(fn)(*args, **kwargs)
    tangent_outs = _tangent(primals, tangents)
    return primal_outs, tangent_outs


@filter_primitive_transpose
def _noinline_transpose(inputs, cts_out):
    # Note that `defined` may also include non-JAX-arrays
    undefined, defined = partition(inputs, _is_undefined, is_leaf=_is_undefined)
    undefined = jtu.tree_map(lambda x: x.aval, undefined, is_leaf=_is_undefined)
    undefined_leaves, undefined_treedef = jtu.tree_flatten(undefined)
    undefined_leaves = tuple(undefined_leaves)
    return _transpose(undefined_leaves, undefined_treedef)(defined, cts_out)


@filter_primitive_batching
def _noinline_batch(inputs, batch_axes):
    inputs = jtu.tree_map(_move_to_front, inputs, batch_axes)
    out = _vmap(inputs)
    return out, jtu.tree_map(lambda _: 0, out)


# No "filter_primitive_mlir", we want to be sure `ctx.avals_in`
# matches up perfectly with `list(flat)`.
def _noinline_mlir(ctx, *flat, treedef, static, smuggle):
    static_leaves, static_treedef = jtu.tree_flatten(static)
    static_leaves = tuple(static_leaves)
    _callback = _get_callback(treedef, static_leaves, static_treedef)
    result, _, keepalive = mlir.emit_python_callback(
        ctx,
        _callback,
        None,
        list(flat),
        ctx.avals_in,
        ctx.avals_out,
        False,
        sharding=None,
    )
    ctx.module_context.add_keepalive(keepalive)
    return result


_noinline_p = jax.core.Primitive("noinline")
_noinline_p.multiple_results = True
_noinline_p.def_impl(_noinline_impl)
_noinline_p.def_abstract_eval(_noinline_abstract)
ad.primitive_jvps[_noinline_p] = _noinline_jvp
ad.primitive_transposes[_noinline_p] = _noinline_transpose
batching.primitive_batchers[_noinline_p] = _noinline_batch
mlir.register_lowering(_noinline_p, _noinline_mlir)


class _NoInlineWrapper(Module):
    fn: Callable

    def __call__(self, *args, **kwargs):
        return filter_primitive_bind(_noinline_p, self.fn, args, kwargs)


def noinline(fn: Callable) -> Callable:
    """Marks a function as not being inlined into a larger computation.

    `fn` should be a function `PyTree[Any] -> PyTree[ArrayLike]`.

    This can help to reduce compile time at the expense of increased runtime.
    """
    return module_update_wrapper(_NoInlineWrapper(fn), fn)


@noinline
def _tangent(primals, tangents):
    no_tangent = jtu.tree_map(_is_none, tangents, is_leaf=_is_none)
    static, dynamic = partition(primals, no_tangent)

    def _tangent2(_dynamic):
        fn, args, kwargs = combine(_dynamic, static)
        return fn(*args, **kwargs)

    _, tangent_outs = jax.jvp(_tangent2, (dynamic,), (tangents,))
    return tangent_outs


@ft.lru_cache(maxsize=None)
def _transpose(undefined_leaves, undefined_treedef):
    undefined = jtu.tree_unflatten(undefined_treedef, undefined_leaves)

    @noinline
    def _transpose2(defined, cts_out):
        def _transpose3(_undefined):
            _fn, _args, _kwargs = combine(defined, _undefined)
            return _fn(*_args, **_kwargs)

        (cts_undefined,) = jax.linear_transpose(_transpose3, undefined)(cts_out)
        return cts_undefined

    return _transpose2


@noinline
def _vmap(inputs):
    fn, args, kwargs = inputs
    return jax.vmap(fn)(*args, **kwargs)
