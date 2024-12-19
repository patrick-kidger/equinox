import functools as ft
from collections.abc import Callable
from typing import Any, Optional, Union

import jax
import jax.core
import jax.extend.core
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, Int, PyTree

from .._ad import filter_jvp
from .._caches import cache_clears
from .._compile_utils import hashable_combine, hashable_partition
from .._eval_shape import filter_eval_shape
from .._filters import combine, is_array, partition
from .._module import field, Module, module_update_wrapper
from .._vmap_pmap import filter_vmap
from . import _primitive
from ._primitive import (
    filter_primitive_batching,
    filter_primitive_bind,
    filter_primitive_def,
    filter_primitive_jvp,
    filter_primitive_transpose,
    materialise_zeros,
)


def _is_shapedarray(x):
    return isinstance(x, jax.core.ShapedArray)


def _to_shapedarray(x):
    if isinstance(x, jax.ShapeDtypeStruct):
        return jax.core.ShapedArray(x.shape, x.dtype)
    else:
        return x


def _only_shapedarrays(rule):
    def _rule_wrapper(*args, **params):
        for arg in args:
            if not _is_shapedarray(arg):
                raise NotImplementedError(
                    "noinline only supports ShapedArrays in abstract evaluation"
                )
        return rule(*args, **params)

    return _rule_wrapper


@ft.lru_cache(maxsize=128)
def _cache_filter_eval_shape(key):
    flat_dynamic, treedef_dynamic, static = key
    flat_dynamic = [jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in flat_dynamic]
    dynamic = jtu.tree_unflatten(treedef_dynamic, flat_dynamic)
    (
        abstract_fn,
        transforms,
        args,
    ) = hashable_combine(dynamic, static)
    assert transforms[0] is _impl_transform
    abstract_fn = _abstract_transform(abstract_fn)
    for transform in transforms[1:]:
        abstract_fn = transform(abstract_fn)
    return filter_eval_shape(abstract_fn, args)


cache_clears.append(_cache_filter_eval_shape.cache_clear)


def _is_undefined(x):
    return type(x) is ad.UndefinedPrimal


def _is_none(x):
    return x is None


def _is_not_mapped(x):
    return x is batching.not_mapped


def _move_to_front(input, batch_axis):
    if batch_axis is batching.not_mapped:
        return input
    else:
        return jnp.swapaxes(input, 0, batch_axis)


def _int_to_zero(batch_axis):
    if batch_axis is batching.not_mapped:
        return batch_axis
    else:
        return 0


@ft.lru_cache(maxsize=None)
def _get_callback(treedef, static, is_float0):
    @ft.partial(jax.jit, static_argnums=0)
    def callback(static_fn, dynamic):
        # pure_callback casts float0 to bool, so need to cast back
        assert len(dynamic) == len(is_float0)
        dynamic = [
            np.broadcast_to(np.zeros((), dtype=jax.dtypes.float0), x.shape) if y else x
            for x, y in zip(dynamic, is_float0)
        ]
        iter_dynamic = iter(dynamic)
        flat = [next(iter_dynamic) if x is None else x for x in static]
        assert next(iter_dynamic, None) is None
        transforms, args = jtu.tree_unflatten(treedef, flat)
        for transform in transforms:
            static_fn = transform(static_fn)
        out = static_fn(args)
        return tuple(jtu.tree_leaves(out))

    def callback_lookup(dynamic_index, *dynamic):
        static_fn = _index_to_fn[dynamic_index.item()]
        return callback(static_fn, dynamic)

    return callback_lookup


cache_clears.append(_get_callback.cache_clear)


def _impl_transform(static_fn):
    def _impl_transform_impl(args):
        dynamic_fn, args, kwargs = args
        fn = hashable_combine(dynamic_fn, static_fn)
        return fn(*args, **kwargs)

    return _impl_transform_impl


def _abstract_transform(abstract_fn):
    def _abstract_transform_impl(args):
        dynamic_fn, args, kwargs = args
        return abstract_fn(dynamic_fn, *args, **kwargs)

    return _abstract_transform_impl


def _jvp_transform(static_fn):
    def _jvp_transform_impl(args):
        args, t_args = args
        tang_is_none = jtu.tree_map(_is_none, t_args, is_leaf=_is_none)
        no_tang, has_tang = partition(args, tang_is_none, is_leaf=_is_none)

        def _to_jvp(_has_tang):
            _args = combine(_has_tang, no_tang)
            return static_fn(_args)

        _, t_out = filter_jvp(_to_jvp, (has_tang,), (t_args,))
        return t_out

    return _jvp_transform_impl


class _MetaTransposeTransform(Module):
    undefined: PyTree[jax.core.ShapedArray]

    def __call__(self, static_fn):
        def _transpose_transform_impl(args):
            defined, cts_out = args

            def _to_transpose(_undefined):
                _args = combine(defined, _undefined)
                return static_fn(_args)

            (cts_undefined,) = jax.linear_transpose(_to_transpose, self.undefined)(
                cts_out
            )
            return cts_undefined

        return _transpose_transform_impl


class _MetaBatchTransform(Module):
    batch_axes: PyTree[Union[batching.NotMapped, int]]  # pyright: ignore

    def __call__(self, static_fn):
        return filter_vmap(static_fn, in_axes=(self.batch_axes,))


@filter_primitive_def
def _noinline_impl(dynamic_index, abstract_fn, transforms, args):
    del abstract_fn
    static_fn = _index_to_fn[dynamic_index.item()]
    for transform in transforms:
        static_fn = transform(static_fn)
    return static_fn(args)


@_only_shapedarrays
@filter_primitive_def
def _noinline_abstract(dynamic_index, abstract_fn, transforms, args):
    del dynamic_index
    dynamic, static = hashable_partition(
        (abstract_fn, transforms, args), _is_shapedarray
    )
    flat_dynamic, treedef_dynamic = jtu.tree_flatten(dynamic)
    key = tuple((x.shape, x.dtype) for x in flat_dynamic), treedef_dynamic, static
    out_struct = _cache_filter_eval_shape(key)
    return jtu.tree_map(_to_shapedarray, out_struct)


@filter_primitive_jvp
def _noinline_jvp(primals, tangents):
    # TODO: add custom partial-eval rule to avoid the double-noinline?
    dynamic_index, abstract_fn, transforms, args = primals
    t_dynamic_index, t_abstract_fn, t_transforms, t_args = tangents
    assert (
        len(jtu.tree_leaves((t_dynamic_index, t_abstract_fn, t_transforms))) == 0
    )  # all none
    del t_dynamic_index, t_abstract_fn, t_transforms
    tangents = jtu.tree_map(materialise_zeros, args, t_args, is_leaf=_is_none)
    primal_outs = filter_primitive_bind(noinline_p, *primals)
    tangent_outs = filter_primitive_bind(
        noinline_p,
        dynamic_index,
        abstract_fn,
        transforms + [_jvp_transform],
        (args, t_args),
    )
    return primal_outs, tangent_outs


@filter_primitive_transpose(materialise_zeros=True)  # pyright: ignore
def _noinline_transpose(inputs, cts_out):
    dynamic_index, abstract_fn, transforms, args = inputs
    assert all(
        not _is_undefined(x)
        for x in jtu.tree_leaves(
            (dynamic_index, abstract_fn, transforms), is_leaf=_is_undefined
        )
    )
    # Note that `defined` may also include non-JAX-arrays
    undefined, defined = partition(args, _is_undefined, is_leaf=_is_undefined)
    undefined = jtu.tree_map(lambda x: x.aval, undefined, is_leaf=_is_undefined)
    cts_args = filter_primitive_bind(
        noinline_p,
        dynamic_index,
        abstract_fn,
        transforms + [_MetaTransposeTransform(undefined)],
        (defined, cts_out),
    )
    cts_rest = jtu.tree_map(lambda _: None, (dynamic_index, abstract_fn, transforms))
    return cts_rest + (cts_args,)


@filter_primitive_batching
def _noinline_batch(inputs, batch_axes):
    dynamic_index, abstract_fn, transforms, args = inputs
    dynamic_index_bdim, abstract_fn_bdim, transforms_bdim, args_bdim = batch_axes
    assert len(jtu.tree_leaves((abstract_fn_bdim, transforms_bdim))) == 0  # all none
    if dynamic_index_bdim is not batching.not_mapped:
        # The batch rule for `lax.cond` with vmap'd predicate simply
        # broadcasts all constants in the branches. In particular it may broadcast
        # this. We simply need to ignore this and return to having a single dynamic
        # index.
        # This is actually a silent error if you do something exceptionally silly, and
        # try to manually combine two different `noinline`d functions. There's no real
        # way to catch this that wouldn't slow things down at runtime though, I don't
        # think.
        assert jnp.ndim(dynamic_index) == 1
        dynamic_index = dynamic_index[0]
    del dynamic_index_bdim, abstract_fn_bdim, transforms_bdim
    args = jtu.tree_map(_move_to_front, args, args_bdim, is_leaf=_is_not_mapped)
    args_bdim = jtu.tree_map(_int_to_zero, args_bdim, is_leaf=_is_not_mapped)
    out = filter_primitive_bind(
        noinline_p,
        dynamic_index,
        abstract_fn,
        transforms + [_MetaBatchTransform(args_bdim)],
        args,
    )
    return out, jtu.tree_map(lambda _: 0, out)


# Not a PyTree
class _MlirWrapper:
    def __init__(self, val):
        self.val = val


def _noinline_mlir(ctx, *dynamic, treedef, static, flatten, **kwargs):
    assert len(kwargs) == 0
    assert flatten.called()
    dynamic = [_MlirWrapper(x) for x in dynamic]
    abstract_dynamic = [_MlirWrapper(x) for x in ctx.avals_in]
    # This is really an internal implementation detail of another component that we're
    # messing with here.
    # Fortunately `noinline` and `primitive` are both inside Equinox, so this isn't too
    # bad.
    flat = _primitive._combine(dynamic, static)
    abstract_flat = _primitive._combine(abstract_dynamic, static)
    index, _, transforms, args = jtu.tree_unflatten(treedef, flat)
    abstract_index, _, abstract_transforms, abstract_args = jtu.tree_unflatten(
        treedef, abstract_flat
    )
    flat, treedef = jtu.tree_flatten((transforms, args))
    abstract_flat, abstract_treedef = jtu.tree_flatten(
        (abstract_transforms, abstract_args)
    )
    assert treedef == abstract_treedef
    dynamic = [x.val for x in flat if type(x) is _MlirWrapper]
    abstract_dynamic = [x.val for x in abstract_flat if type(x) is _MlirWrapper]
    static = tuple(None if type(x) is _MlirWrapper else x for x in flat)
    is_float0 = tuple(x.dtype == jax.dtypes.float0 for x in abstract_dynamic)
    callback = _get_callback(treedef, static, is_float0)

    vals_in = [index.val] + dynamic
    avals_in = [abstract_index.val] + abstract_dynamic
    result, _, keepalive = mlir.emit_python_callback(
        ctx,
        callback,
        None,
        vals_in,
        avals_in,
        ctx.avals_out,
        has_side_effect=False,
        sharding=None,
    )
    ctx.module_context.add_keepalive(keepalive)
    return result


noinline_p = jax.extend.core.Primitive("noinline")
noinline_p.multiple_results = True
noinline_p.def_impl(_noinline_impl)
noinline_p.def_abstract_eval(_noinline_abstract)
ad.primitive_jvps[noinline_p] = _noinline_jvp
ad.primitive_transposes[noinline_p] = _noinline_transpose
batching.primitive_batchers[noinline_p] = _noinline_batch
mlir.register_lowering(noinline_p, _noinline_mlir)


_fn_to_index = {}
_index_to_fn = []


class _NoInlineWrapper(Module):
    dynamic_index: Int[Union[Array, np.ndarray], ""]
    abstract_fn: Callable = field(static=True)
    dynamic_fn: Any

    @property
    def __wrapped__(self):
        return self.abstract_fn

    def __call__(self, *args, **kwargs):
        return filter_primitive_bind(
            noinline_p,
            self.dynamic_index,
            self.abstract_fn,
            [_impl_transform],
            (self.dynamic_fn, args, kwargs),
        )


def noinline(fn: Callable, abstract_fn: Optional[Callable] = None) -> Callable:  # pyright: ignore
    """Marks a function as not being inlined into a larger computation.
    This can help to reduce compile time at the expense of increased runtime.

    `fn` can be any callable `PyTree[Any] -> PyTree[Any]`. In addition `fn`
    itself may be any (callable) PyTree; any JAX arrays in `fn`'s PyTree are
    also treated as inputs to the noinline'd computation.

    `abstract_fn` determines the shapes/dtypes/pytrees of the output. (And must
     return results consistent with `fn`.) If `fn` is called as
    `fn(*args, **kwargs)`, then `abstract_fn` is called as
    ```python
    eqx.filter_eval_shape(
        abstract_fn, eqx.filter(fn, eqx.is_array), *args, **kwargs)
    )
    ```
    If not passed then `abstract_fn` is automatically constructed from `fn`.
    Specifying it is useful as noinline'd functions sharing the same
    `abstract_fn` may be substituted for each other without recompiling the
    enclosing computation graph; see the second example below.

    !!! Example

        ```python
        @noinline
        def f(x, y):
            print("Tracing!")
            return x + y

        @jax.jit
        def g(x, y):
            a = f(x, jnp.array(1))
            b = f(x, jnp.array(2))
            c = f(y, jnp.array(1))
            d = f(y, jnp.array(2))
            return a + b + c + d

        g(1, 1)
        ```

        In this example, `Tracing!` is printed twice. Once just for abstract
        evaluation (only) to figure out out its return shape in `f`'s
        computation graph. (Which is computationally cheap.) And then once
        when it is actually compiled as part of its own computation graph.
        (Which is computationally expensive.)

        Without the `noinline`, we would see `Tracing!` printed four times:
        once for each call, and with it being compiled each time. (Every time
        being computationally expensive.)

        Note how the `1` and the `2` are wrapped in arrays. If they were still
        Python scalars then they would be treated as static arguments, which
        will lead to recompilations of `f`. (E.g. what if there was an
        `if y == 1` command in there?)

    !!! Example

        ```python
        def abstract(_, x, y):
            return jnp.broadcast_arrays(x, y)[0]

        def f(x, y):
            print("Compiling f!")
            return x + y

        def g(x, y):
            print("Compiling g!")
            return x * y

        f = noinline(f, abstract)
        g = noinline(g, abstract)

        @jax.jit
        def call(fn, x, y):
            print("Compiling call!")
            return fn(x, y)

        call(f, 1, 1)  # Compiling call! Compiling f!
        call(g, 1, 1)  # Compiling g!
        ```

        In this example, we see how noinline'd functions with the same
        `abstract_fn` may be passed as inputs to the main call graph,
        and swapped without needing to recompile the main call graph.

    """
    dynamic_fn, static_fn = hashable_partition(fn, is_array)
    if abstract_fn is None:

        def abstract_fn(__dynamic_fn, *args, **kwargs):
            _fn = hashable_combine(__dynamic_fn, static_fn)
            return _fn(*args, **kwargs)

    try:
        dynamic_index = _fn_to_index[static_fn]
    except KeyError:
        dynamic_index = len(_index_to_fn)
        _fn_to_index[static_fn] = dynamic_index
        _index_to_fn.append(static_fn)
    dynamic_index = np.array(dynamic_index)
    noinline_fn = _NoInlineWrapper(dynamic_index, abstract_fn, dynamic_fn)
    return module_update_wrapper(noinline_fn)
