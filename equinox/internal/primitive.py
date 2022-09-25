import functools as ft
from typing import Union

import jax
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, PyTree

from ..filters import combine, is_array, partition
from ..vmap_pmap import filter_vmap


#
# filter primitives
# -----------------
# As with all filtering in Equinox, this is basically just about putting a
# nicer interface on existing JAX operations; in this case creating custom
# primitives. The inputs to the primitive can be arbitrary. The output must
# be a PyTree of JAX arrays.
#

_ArrayLike = Union[Array, np.ndarray, int, float, complex, bool]
_sentinel = object()


def _flat_smuggle(out, smuggle, *, treedef=_sentinel):
    flat_out, treedef_out = jtu.tree_flatten(out)
    if treedef is not _sentinel:
        assert treedef_out == treedef
    if len(smuggle) == 0:
        smuggle.append(treedef_out)
    else:
        assert len(smuggle) == 1
        assert smuggle[0] == treedef_out
    return flat_out


def filter_primitive_def(rule):
    def _wrapper(*flat, treedef, static, smuggle):
        args = combine(jtu.tree_unflatten(treedef, flat), static)
        out = rule(*args)
        return _flat_smuggle(out, smuggle)

    return _wrapper


def filter_primitive_jvp(rule):
    def _wrapper(primals, tangents, *, treedef, static, smuggle):
        primals = combine(jtu.tree_unflatten(treedef, primals), static)
        nones = jtu.tree_map(lambda _: None, static)
        tangents = combine(jtu.tree_unflatten(treedef, tangents), nones)
        primals_out, tangents_out = rule(primals, tangents)
        flat_tangents_out, treedef_tangents_out = jtu.tree_flatten(tangents_out)
        flat_primals_out = _flat_smuggle(
            primals_out, smuggle, treedef=treedef_tangents_out
        )
        return flat_primals_out, flat_tangents_out

    return _wrapper


# Not a PyTree
class _WrappedPrimal:
    def __init__(self, value):
        self.value = value


def _wrap_undefined(x):
    if isinstance(x, ad.UndefinedPrimal):
        return _WrappedPrimal(x)
    else:
        return x


def _unwrap_undefined(x):
    if isinstance(x, _WrappedPrimal):
        return x.value
    else:
        return x


def _is_none(x):
    return x is None


_dummy_none = object()


def _replace_none(x):
    return _dummy_none if x is None else x


def _get_ct(v, ct):
    if isinstance(v, _WrappedPrimal):
        assert ct is not _dummy_none
        assert v.value.aval == ct.aval
    else:
        assert ct is _dummy_none
    return ct


def filter_primitive_transpose(rule):
    def _wrapper(cts_out, *flat, treedef, static, smuggle):
        assert len(smuggle) == 1
        treedef_out = smuggle[0]
        cts_out = jtu.tree_unflatten(treedef_out, cts_out)
        wrapped_flat = [_wrap_undefined(x) for x in flat]
        wrapped_dynamic = jtu.tree_unflatten(treedef, wrapped_flat)
        wrapped_inputs = combine(wrapped_dynamic, static)
        inputs = jtu.tree_map(_unwrap_undefined, wrapped_inputs)
        cts = rule(inputs, cts_out)
        assert jtu.tree_structure(
            wrapped_inputs, is_leaf=_is_none
        ) == jtu.tree_structure(cts, is_leaf=_is_none)
        cts = jtu.tree_map(_replace_none, cts, is_leaf=_is_none)
        cts = jtu.tree_map(_get_ct, wrapped_dynamic, cts)
        cts = jtu.tree_leaves(cts)
        return [None if ct is _dummy_none else ct for ct in cts]

    return _wrapper


def filter_primitive_batching(rule):
    def _wrapper(flat, batch_axes, *, treedef, static, smuggle):
        inputs = combine(jtu.tree_unflatten(treedef, flat), static)
        nones = jtu.tree_map(lambda _: None, static)
        batch_axes = combine(jtu.tree_unflatten(treedef, batch_axes), nones)
        out, batch_axes = rule(inputs, batch_axes)
        flat_batch_axes, treedef_batch_axes = jtu.tree_flatten(batch_axes)
        flat_out = _flat_smuggle(out, smuggle, treedef=treedef_batch_axes)
        return flat_out, flat_batch_axes

    return _wrapper


def filter_primitive_bind(prim: jax.core.Primitive, *args) -> PyTree[_ArrayLike]:
    assert prim.multiple_results
    dynamic, static = partition(args, is_array)
    flat, treedef = jtu.tree_flatten(dynamic)
    smuggle = []
    flat_out = prim.bind(*flat, treedef=treedef, static=static, smuggle=smuggle)
    assert len(smuggle) == 1
    treedef_out = smuggle[0]
    return jtu.tree_unflatten(treedef_out, flat_out)


#
# vprim
# -----
# This allows for creating a primitive without needing to specify its batching rule.
# This is instead automatically obtained by vmap'ing its other rules (assuming that
# its rules are implemented using JAX operations).
#


def create_vprim(name: str, impl, abstract_eval, jvp, transpose):
    prim = jax.core.Primitive(name)
    prim.multiple_results = True

    def batch_rule(inputs, batch_axes, **params):
        # delegates batching to `_vprim_p`
        out = _vprim_p.bind(*inputs, prim=prim, batch_axes=batch_axes, params=params)
        batch_axes_out = jtu.tree_map(lambda _: 0, out)
        return out, batch_axes_out

    prim.def_impl(impl)
    prim.def_abstract_eval(abstract_eval)
    ad.primitive_jvps[prim] = jvp
    ad.primitive_transposes[prim] = transpose
    batching.primitive_batchers[prim] = batch_rule
    mlir.register_lowering(prim, mlir.lower_fun(impl, multiple_results=True))
    return prim


def _vprim_impl(*inputs, prim, batch_axes, params):
    impl = ft.partial(prim.impl, **params)
    return filter_vmap(impl, args=batch_axes)(*inputs)


def _vprim_abstract_eval(*inputs, prim, batch_axes, params):
    abstract_eval = ft.partial(prim.abstract_eval, **params)
    return filter_vmap(abstract_eval, args=batch_axes)(*inputs)


def _vprim_jvp(primals, tangents, *, prim, batch_axes, params):
    jvp = ft.partial(ad.primitive_jvps[prim], **params)
    return filter_vmap(jvp, args=batch_axes)(primals, tangents)


def _vprim_transpose(cts, *inputs, prim, batch_axes, params):
    transpose = ft.partial(ad.primitive_transposes[prim], **params)
    return filter_vmap(transpose, args=batch_axes)(cts, *inputs)


# _vprim_p is itself a vprim!
_vprim_p = create_vprim(
    "vprim", _vprim_impl, _vprim_abstract_eval, _vprim_jvp, _vprim_transpose
)
