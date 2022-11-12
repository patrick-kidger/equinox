import functools as ft

import jax
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PyTree

from ..filters import combine, is_array, is_array_like, partition
from ..tree import tree_equal


#
# filter primitives
# -----------------
# As with all filtering in Equinox, this is basically just about putting a
# nicer interface on existing JAX operations; in this case creating custom
# primitives. The inputs and outputs to the primitive can be arbitrary.
#


_like_sentinel = object()
_dummy_none = object()


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
    shape = jnp.shape(p)
    dtype = jax.core.primal_dtype_to_tangent_dtype(jnp.result_type(p))
    return ad.Zero(jax.core.ShapedArray(shape, dtype))


def _is_none(x):
    return x is None


def _replace_none(x):
    return _dummy_none if x is None else x


def _get_second(x, y):
    return y


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
            like = jtu.tree_map(_get_second, dynamic_out, like)
            flat_like, treedef_like = jtu.tree_flatten(like)
            flat_like = [None if x is _dummy_none else x for x in flat_like]
            assert treedef_like == treedef_out
            assert len(flat_out) == len(flat_like)
            return flat_out, flat_like


def filter_primitive_def(rule):
    """For wrapping def_impl and def_abstract_eval.

    These can now take arbitrary inputs and outputs.
    """

    def _wrapper(*flat, treedef, static, flatten):
        args = combine(jtu.tree_unflatten(treedef, flat), static)
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
        primals = combine(jtu.tree_unflatten(treedef, primals), static)
        tangents = [None if type(t) is ad.Zero else t for t in tangents]
        tangents = jtu.tree_unflatten(treedef, tangents)
        primals_out, tangents_out = rule(primals, tangents)
        flat_primals_out, flat_tangents_out = flatten(primals_out, tangents_out)
        flat_tangents_out = [
            _zero_from_primal(p) if t is None else t
            for p, t in zip(flat_primals_out, flat_tangents_out)
        ]
        return flat_primals_out, flat_tangents_out

    return _wrapper


def filter_primitive_transpose(rule):
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

    def _wrapper(cts_out, *flat, treedef, static, flatten):
        treedef_out, _ = flatten.get()
        cts_out = [None if type(ct) is ad.Zero else ct for ct in cts_out]
        cts_out = jtu.tree_unflatten(treedef_out, cts_out)
        wrapped_flat = [_wrap_undefined(x) for x in flat]
        wrapped_dynamic = jtu.tree_unflatten(treedef, wrapped_flat)
        wrapped_inputs = combine(wrapped_dynamic, static)
        inputs = jtu.tree_map(_unwrap_undefined, wrapped_inputs)
        cts = rule(inputs, cts_out)
        flat_inputs, flat_cts = Flatten()(wrapped_inputs, cts)
        flat_inputs = [_unwrap_undefined(p, aval=True) for p in flat_inputs]
        flat_cts = [
            _zero_from_primal(p) if ct is None else ct
            for p, ct in zip(flat_inputs, flat_cts)
        ]
        assert len(flat) == len(flat_cts)
        return flat_cts

    return _wrapper


def filter_primitive_batching(rule):
    """
    The input batch axes (to the wrapped rule) will be a PyTree with the same
    structure as the input primals, with `None` for all non-JAX-arrays.

    The output batch axes are expected to match the output primals, with `None`
    for all non-JAX-arrays.
    """

    def _wrapper(flat, batch_axes, *, treedef, static, flatten):
        inputs = combine(jtu.tree_unflatten(treedef, flat), static)
        batch_axes = [None if b is batching.not_mapped else b for b in batch_axes]
        batch_axes = jtu.tree_unflatten(treedef, batch_axes)
        out, batch_axes = rule(inputs, batch_axes)
        flat_out, flat_batch_axes = flatten(out, batch_axes)
        flat_batch_axes = [
            batching.not_mapped if b is None else b for b in flat_batch_axes
        ]
        return flat_out, flat_batch_axes

    return _wrapper


def filter_primitive_bind(prim: jax.core.Primitive, *args) -> PyTree:
    """Calls a primitive that has had its rules defined using the filter
    functions above.
    """
    assert prim.multiple_results
    dynamic, static = partition(args, is_array)
    flat, treedef = jtu.tree_flatten(dynamic)
    flatten = Flatten()
    flat_out = prim.bind(*flat, treedef=treedef, static=static, flatten=flatten)
    treedef_out, static_out = flatten.get()
    return combine(jtu.tree_unflatten(treedef_out, flat_out), static_out)


# Useful helper for JVP rules of higher-order primitives.
def materialise_zeros(primal, tangent):
    if tangent is None and is_array_like(primal):
        shape = jnp.shape(primal)
        dtype = jax.core.primal_dtype_to_tangent_dtype(jnp.result_type(primal))
        if dtype == jax.dtypes.float0:
            return np.broadcast_to(np.zeros((), dtype=dtype), shape)
        else:
            weak_type = hasattr(primal, "weak_type") and primal.weak_type
            if weak_type:
                return lax.broadcast(jnp.array(0, dtype=dtype), shape)
            else:
                return jnp.zeros(shape, dtype=dtype)
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
    prim = jax.core.Primitive(name)
    prim.multiple_results = True

    def batch_rule(inputs, batch_axes, **params):
        # delegates batching to `_vprim_p`
        out = _vprim_p.bind(*inputs, prim=prim, __batch_axes=batch_axes, params=params)
        batch_axes_out = jtu.tree_map(lambda _: 0, out)
        return out, batch_axes_out

    prim.def_impl(impl)
    prim.def_abstract_eval(abstract_eval)
    ad.primitive_jvps[prim] = jvp
    ad.primitive_transposes[prim] = transpose
    batching.primitive_batchers[prim] = batch_rule
    mlir.register_lowering(prim, mlir.lower_fun(impl, multiple_results=True))
    _vprim_impl_registry[prim] = impl
    _vprim_abstract_eval_registry[prim] = abstract_eval
    _vprim_jvp_registry[prim] = jvp
    _vprim_transpose_registry[prim] = transpose
    return prim


def _vprim_impl(*inputs, prim, __batch_axes, params):
    impl = ft.partial(_vprim_impl_registry[prim], **params)
    impl = jax.vmap(impl, in_axes=__batch_axes)
    return impl(*inputs)


def _to_struct(x):
    return jax.ShapeDtypeStruct(x.shape, x.dtype)


def _to_shapedarray(x):
    return jax.core.ShapedArray(x.shape, x.dtype)


def _vprim_abstract_eval(*inputs, prim, __batch_axes, params):
    abstract_eval = ft.partial(_vprim_abstract_eval_registry[prim], **params)
    abstract_eval = jax.vmap(abstract_eval, in_axes=__batch_axes)
    inputs = [_to_struct(x) for x in inputs]
    out = jax.eval_shape(abstract_eval, *inputs)
    return [_to_shapedarray(x) for x in out]


def _resolve_zeros_t(tangent, batch_axis):
    if type(tangent) is ad.Zero and isinstance(batch_axis, int):
        aval = tangent.aval
        if type(aval) is not jax.core.ShapedArray:
            raise NotImplementedError(
                "vprim only currently supports shaped arrays for symbolic zeros"
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


def _vprim_jvp(primals, tangents, *, prim, __batch_axes, params):
    assert len(primals) == len(__batch_axes)
    assert len(tangents) == len(__batch_axes)
    tangents = [_resolve_zeros_t(t, b) for t, b in zip(tangents, __batch_axes)]
    batch_axes_t = [_resolve_zeros_b(t, b) for t, b in zip(tangents, __batch_axes)]
    jvp = ft.partial(_vprim_jvp_registry[prim], **params)
    jvp = jax.vmap(jvp, in_axes=(__batch_axes, batch_axes_t))
    return jvp(primals, tangents)


def _resolve_undefined_i(input, batch_axis):
    if type(input) is ad.UndefinedPrimal and isinstance(batch_axis, int):
        aval = input.aval
        if type(aval) is not jax.core.ShapedArray:
            raise NotImplementedError(
                "vprim only currently supports shaped arrays for undefined primals"
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


def _vprim_transpose(cts, *inputs, prim, __batch_axes, params):
    inputs = [_resolve_undefined_i(i, b) for i, b in zip(inputs, __batch_axes)]
    batch_axes = [_resolve_undefined_b(i, b) for i, b in zip(inputs, __batch_axes)]
    transpose = ft.partial(_vprim_transpose_registry[prim], **params)
    transpose = jax.vmap(transpose, in_axes=(0, *batch_axes))
    return transpose(cts, *inputs)


# _vprim_p is itself a vprim!
_vprim_p = create_vprim(
    "vprim", _vprim_impl, _vprim_abstract_eval, _vprim_jvp, _vprim_transpose
)
