import functools as ft
from typing import Any

import jax
import jax.core
import jax.extend.core
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree

from .._filters import combine, is_array, is_array_like, partition
from .._tree import tree_equal


#
# filter primitives
# -----------------
# As with all filtering in Equinox, this is basically just about putting a
# nicer interface on existing JAX operations; in this case creating custom
# primitives. The inputs and outputs to the primitive can be arbitrary.
#


_like_sentinel = object()
_dummy_none = object()
_missing_dynamic = object()


# Not a PyTree
class _WrappedPrimal:
    def __init__(self, value):
        self.value = value


def _wrap_undefined(x):
    if isinstance(x, ad.UndefinedPrimal):
        return _WrappedPrimal(x)
    else:
        return x


def _unwrap_undefined(x, aval=False):
    if isinstance(x, _WrappedPrimal):
        if aval:
            return x.value.aval
        else:
            return x.value
    else:
        return x


def _is_array_like_internal(x):
    assert type(x) is not ad.UndefinedPrimal
    # Not `type(x) in (...)` as that doesn't handle stuff like ConcreteArrays.
    return is_array(x) or isinstance(x, (jax.core.ShapedArray, _WrappedPrimal))


def _zero_from_primal(p):
    assert type(p) is not ad.UndefinedPrimal
    aval = jax.core.get_aval(p)
    if hasattr(aval, "to_tangent_aval"):
        # JAX >=0.4.34
        aval = aval.to_tangent_aval()  # pyright: ignore
    else:
        # earlier JAX
        aval = aval.at_least_vspace()
    return ad.Zero(aval)


def _combine(dynamic, static):
    iter_dynamic = iter(dynamic)
    out = [next(iter_dynamic) if x is _missing_dynamic else x for x in static]
    assert next(iter_dynamic, None) is None
    return out


def _is_none(x):
    return x is None


def _replace_none(x):
    return _dummy_none if x is None else x


def _get_second(x, y):
    return None if x is None else y


def _make_spec(x, y):
    if y is None:
        return _is_array_like_internal(x)
    else:
        assert x is not None
        return True


class Flatten:
    __slots__ = ("treedef_out", "static_out")

    def called(self):
        return hasattr(self, "treedef_out")

    def get(self):
        return self.treedef_out, self.static_out

    def __call__(self, out, like=_like_sentinel):
        if like is _like_sentinel:
            dynamic_out, static_out = partition(out, _is_array_like_internal)
            flat_out, treedef_out = jtu.tree_flatten(dynamic_out)
            try:
                treedef_out_old = self.treedef_out
                static_out_old = self.static_out
            except AttributeError:
                self.treedef_out = treedef_out
                self.static_out = static_out
            else:
                assert treedef_out_old == treedef_out
                assert tree_equal(static_out_old, static_out)
            return flat_out
        else:
            assert jtu.tree_structure(out, is_leaf=_is_none) == jtu.tree_structure(
                like, is_leaf=_is_none
            )
            spec = jtu.tree_map(_make_spec, out, like, is_leaf=_is_none)
            dynamic_out, static_out = partition(out, spec, is_leaf=_is_none)
            flat_out, treedef_out = jtu.tree_flatten(dynamic_out)
            try:
                treedef_out_old = self.treedef_out
                static_out_old = self.static_out
            except AttributeError:
                self.treedef_out = treedef_out
                self.static_out = static_out
            else:
                assert treedef_out_old == treedef_out
                assert tree_equal(static_out_old, static_out)
            like = jtu.tree_map(_replace_none, like, is_leaf=_is_none)
            like = jtu.tree_map(_get_second, dynamic_out, like, is_leaf=_is_none)
            flat_like, treedef_like = jtu.tree_flatten(like)
            flat_like = [None if x is _dummy_none else x for x in flat_like]
            assert treedef_like == treedef_out
            assert len(flat_out) == len(flat_like)
            return flat_out, flat_like


def filter_primitive_def(rule):
    """For wrapping def_impl and def_abstract_eval.

    These can now take arbitrary inputs and outputs.
    """

    def _wrapper(*dynamic, treedef, static, flatten):
        args = jtu.tree_unflatten(treedef, _combine(dynamic, static))
        out = rule(*args)
        return flatten(out)

    return _wrapper


def filter_primitive_jvp(rule):
    """
    The input tangents (to the wrapped rule) will be a PyTree with the same
    structure as the input primals. `None` indicates symbolic zero tangents,
    in particular for non-JAX-array-likes.

    The output tangents are expected to match the output primals, necessarily
    with `None` for all non-JAX-array-likes.
    """

    def _wrapper(primals, tangents, *, treedef, static, flatten):
        tangents = [None if type(t) is ad.Zero else t for t in tangents]
        tangents_static = [x if x is _missing_dynamic else None for x in static]
        primals = jtu.tree_unflatten(treedef, _combine(primals, static))
        tangents = jtu.tree_unflatten(treedef, _combine(tangents, tangents_static))
        primals_out, tangents_out = rule(primals, tangents)
        flat_primals_out, flat_tangents_out = flatten(primals_out, tangents_out)
        flat_tangents_out = [
            _zero_from_primal(p) if t is None else t
            for p, t in zip(flat_primals_out, flat_tangents_out)
        ]
        return flat_primals_out, flat_tangents_out

    return _wrapper


_sentinel: Any = object()


def filter_primitive_transpose(rule=_sentinel, *, materialise_zeros=False):
    """
    The `inputs` to the transpose rule are a PyTree like the primal
    inputs, with `UndefinedPrimal`s where appropriate.

    The `cts_out` passed to the transpose rule are a PyTree like the
    primal output, with `None` for symbolic zero cotangents, in particular
    for non-JAX-array-likes.

    The output from the rule should be a PyTree like the primal input.
    All leaves which were non-JAX-array-like, or which should have zero
    cotangent, should have cotangent `None`.
    """
    if rule is _sentinel:
        return ft.partial(
            filter_primitive_transpose, materialise_zeros=materialise_zeros
        )

    def _wrapper(cts_out, *dynamic, treedef, static, flatten):
        treedef_out, _ = flatten.get()
        if materialise_zeros:
            cts_out = [ad.instantiate_zeros(ct) for ct in cts_out]
        else:
            cts_out = [None if type(ct) is ad.Zero else ct for ct in cts_out]
        cts_out = jtu.tree_unflatten(treedef_out, cts_out)
        wrapped_dynamic = [_wrap_undefined(x) for x in dynamic]
        wrapped_flat = _combine(wrapped_dynamic, static)
        wrapped_inputs = jtu.tree_unflatten(treedef, wrapped_flat)
        inputs = jtu.tree_map(_unwrap_undefined, wrapped_inputs)
        cts = rule(inputs, cts_out)
        flat_inputs, flat_cts = Flatten()(wrapped_inputs, cts)
        flat_inputs = [_unwrap_undefined(p, aval=True) for p in flat_inputs]
        flat_cts = [
            _zero_from_primal(p) if ct is None else ct
            for p, ct in zip(flat_inputs, flat_cts)
        ]
        assert len(dynamic) == len(flat_cts)
        return flat_cts

    return _wrapper


def filter_primitive_batching(rule):
    """
    The input batch axes (to the wrapped rule) will be a PyTree with the same
    structure as the input primals, with `None` for all non-JAX-arrays.

    The output batch axes are expected to match the output primals, with `None`
    for all non-JAX-arrays.
    """

    def _wrapper(dynamic, batch_axes, *, treedef, static, flatten):
        flat = _combine(dynamic, static)
        inputs = jtu.tree_unflatten(treedef, flat)
        batch_axes = [None if b is batching.not_mapped else b for b in batch_axes]
        batch_axes_static = [x if x is _missing_dynamic else None for x in static]
        batch_axes = _combine(batch_axes, batch_axes_static)
        batch_axes = jtu.tree_unflatten(treedef, batch_axes)
        out, batch_axes = rule(inputs, batch_axes)
        flat_out, flat_batch_axes = flatten(out, batch_axes)
        flat_batch_axes = [
            batching.not_mapped if b is None else b for b in flat_batch_axes
        ]
        return flat_out, flat_batch_axes

    return _wrapper


def filter_primitive_bind(prim: jax.extend.core.Primitive, *args) -> PyTree:
    """Calls a primitive that has had its rules defined using the filter
    functions above.
    """
    assert prim.multiple_results
    # If `args` contains a Jaxpr or ClosedJaxpr in its leaves, then it ends up as a
    # member of the `static` tuple. This is important to ensure that jaxpr-rewriting
    # passes are able to find it.
    # (E.g. if `eqx.filter_closure_convert(...)` is an argument and we apply
    # `jax.core.jaxprs_in_params`.)
    flat, treedef = jtu.tree_flatten(args)
    dynamic = [x for x in flat if is_array(x)]
    static = tuple(_missing_dynamic if is_array(x) else x for x in flat)
    flatten = Flatten()
    flat_out = prim.bind(*dynamic, treedef=treedef, static=static, flatten=flatten)
    treedef_out, static_out = flatten.get()
    return combine(jtu.tree_unflatten(treedef_out, flat_out), static_out)


# Useful helper for JVP rules of higher-order primitives.
def materialise_zeros(primal, tangent, allow_struct=False):
    arraylike = is_array_like(primal)
    if allow_struct:
        arraylike = arraylike or isinstance(primal, jax.ShapeDtypeStruct)
    if tangent is None and arraylike:
        tangent = _zero_from_primal(primal)
        return ad.instantiate_zeros(tangent)
    else:
        return tangent


#
# vprim
# -----
# This allows for creating a primitive without needing to specify its batching rule.
# This is instead automatically obtained by vmap'ing its other rules (assuming that
# its rules are implemented using JAX operations).
#

_vprim_impl_registry = {}
_vprim_abstract_eval_registry = {}
_vprim_jvp_registry = {}
_vprim_transpose_registry = {}


def create_vprim(name: str, impl, abstract_eval, jvp, transpose):
    prim = jax.extend.core.Primitive(name)
    prim.multiple_results = True

    def batch_rule(axis_size, axis_name, trace_type, inputs, batch_axes, **params):
        del trace_type
        if all(b is batching.not_mapped for b in jtu.tree_leaves(batch_axes)):
            out = prim.bind(*inputs, **params)
            batch_axes_out = jtu.tree_map(lambda _: batching.not_mapped, out)
        else:
            # delegates batching to `_vprim_p`
            out = _vprim_p.bind(
                *inputs,
                prim=prim,
                __axis_size=axis_size,
                __axis_name=axis_name,
                __batch_axes=batch_axes,
                params=params,
            )
            batch_axes_out = jtu.tree_map(lambda _: 0, out)
        return out, batch_axes_out

    prim.def_impl(impl)
    prim.def_abstract_eval(abstract_eval)
    ad.primitive_jvps[prim] = jvp
    ad.primitive_transposes[prim] = transpose
    batching.axis_primitive_batchers[prim] = batch_rule
    mlir.register_lowering(prim, mlir.lower_fun(impl, multiple_results=True))
    _vprim_impl_registry[prim] = impl
    _vprim_abstract_eval_registry[prim] = abstract_eval
    _vprim_jvp_registry[prim] = jvp
    _vprim_transpose_registry[prim] = transpose
    return prim


def _vprim_impl(*inputs, prim, __axis_size, __axis_name, __batch_axes, params):
    impl = ft.partial(_vprim_impl_registry[prim], **params)
    impl = jax.vmap(
        impl, in_axes=__batch_axes, axis_size=__axis_size, axis_name=__axis_name
    )
    return impl(*inputs)


if jax.__version_info__ >= (0, 5, 1):

    def _unmapped_aval(axis_size, axis_name, axis, aval):
        del axis_name
        return jax.core.unmapped_aval(axis_size, axis, aval)  # pyright: ignore[reportCallIssue]
else:
    # signature (axis_size, axis_name, axis, aval)
    _unmapped_aval = jax.core.unmapped_aval  # pyright: ignore[reportAssignmentType]


def _vprim_abstract_eval(*inputs, prim, __axis_size, __axis_name, __batch_axes, params):
    assert len(inputs) == len(__batch_axes)
    inputs = [
        jax.core.mapped_aval(__axis_size, b, x) for x, b in zip(inputs, __batch_axes)
    ]
    abstract_eval = _vprim_abstract_eval_registry[prim]
    outs = abstract_eval(*inputs, **params)
    outs = [_unmapped_aval(__axis_size, __axis_name, 0, x) for x in outs]
    return outs


def _resolve_zeros_t(tangent, batch_axis):
    if type(tangent) is ad.Zero and isinstance(batch_axis, int):
        aval = tangent.aval
        # Also accepts ConcreteArrays
        if not isinstance(aval, jax.core.ShapedArray):
            raise NotImplementedError(
                "vprim only currently supports ShapedArrays for symbolic zeros"
            )
        shape = aval.shape[:batch_axis] + aval.shape[batch_axis + 1 :]
        return ad.Zero(jax.core.ShapedArray(shape, aval.dtype))
    else:
        return tangent


def _resolve_zeros_b(tangent, batch_axis):
    if type(tangent) is ad.Zero:
        return None
    else:
        return batch_axis


def _vprim_jvp(
    primals, tangents, *, prim, __axis_size, __axis_name, __batch_axes, params
):
    assert len(primals) == len(__batch_axes)
    assert len(tangents) == len(__batch_axes)
    tangents = [_resolve_zeros_t(t, b) for t, b in zip(tangents, __batch_axes)]
    batch_axes_t = [_resolve_zeros_b(t, b) for t, b in zip(tangents, __batch_axes)]
    jvp = ft.partial(_vprim_jvp_registry[prim], **params)
    jvp = jax.vmap(
        jvp,
        in_axes=(__batch_axes, batch_axes_t),
        axis_size=__axis_size,
        axis_name=__axis_name,
    )
    return jvp(primals, tangents)


def _resolve_undefined_i(input, batch_axis):
    if type(input) is ad.UndefinedPrimal and isinstance(batch_axis, int):
        aval = input.aval
        # Also accepts ConcreteArrays
        if not isinstance(aval, jax.core.ShapedArray):
            raise NotImplementedError(
                "vprim only currently supports ShapedArrays for undefined primals"
            )
        shape = aval.shape[:batch_axis] + aval.shape[batch_axis + 1 :]
        return ad.UndefinedPrimal(jax.core.ShapedArray(shape, aval.dtype))
    else:
        return input


def _resolve_undefined_b(input, batch_axis):
    if type(input) is ad.UndefinedPrimal:
        return None
    else:
        return batch_axis


def _vprim_transpose(
    cts, *inputs, prim, __axis_size, __axis_name, __batch_axes, params
):
    mapped_inputs = [_resolve_undefined_i(i, b) for i, b in zip(inputs, __batch_axes)]
    batch_axes = [_resolve_undefined_b(i, b) for i, b in zip(inputs, __batch_axes)]

    def _transpose(*_inputs):
        _outputs = _vprim_transpose_registry[prim](*_inputs, **params)
        # `Zero` is not a JAX type -- it's an internal AD thing -- so we shouldn't pass
        # it across the `vmap` boundary. In particular JAX won't apply the out batch
        # axis to it.
        # JAX allows for returning `None` to indicate no cotangent, so we use that
        # instead, which is compatible with both `vmap` and `out_axes`.
        return tuple(None if type(o) is ad.Zero else o for o in _outputs)

    transpose = jax.vmap(
        _transpose,
        in_axes=(0, *batch_axes),
        out_axes=__batch_axes,
        axis_size=__axis_size,
        axis_name=__axis_name,
    )
    if prim.multiple_results:
        cts = tuple(None if type(c) is ad.Zero else c for c in cts)
    else:
        cts = None if type(cts) is ad.Zero else cts
    outputs = transpose(cts, *mapped_inputs)
    assert len(inputs) == len(outputs)
    for i, o in zip(inputs, outputs):
        if o is not None:
            # Can't have cotangents on defined variables I think? The point of an
            # `UndefinedPrimal` is to declare what you want cotangents with respect to.
            assert type(i) is ad.UndefinedPrimal
            # We've filtered out all other avals above, with a `NotImplementedError` if
            # required.
            assert isinstance(i.aval, jax.core.ShapedArray)
            assert i.aval.shape == jnp.shape(o)
    return outputs


# _vprim_p is itself a vprim!
_vprim_p = create_vprim(
    "vprim", _vprim_impl, _vprim_abstract_eval, _vprim_jvp, _vprim_transpose
)
