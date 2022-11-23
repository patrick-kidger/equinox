from typing import Tuple, Union

import jax
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Int, PyTree, Shaped


# TODO: replace with just a custom batch rule when possible


def _at_set_impl(ys, i, y):
    return lax.dynamic_update_index_in_dim(ys, y, i, 0)


def _at_set_abstract(ys, i, y):
    return ys


def _at_set_jvp(primals, tangents):
    ys, i, y = primals
    t_ys, _, t_y = tangents
    out = at_set(ys, i, y)
    if type(t_ys) is ad.Zero and type(t_y) is ad.Zero:
        t_out = ad.Zero.from_value(out)
    else:
        t_ys = ad.instantiate_zeros(t_ys)
        t_y = ad.instantiate_zeros(t_y)
        t_out = at_set(t_ys, i, t_y)
    return out, t_out


def _at_set_transpose(ct_out, ys, i, y):
    assert not ad.is_undefined_primal(i)
    if type(ct_out) is ad.Zero:
        ct_ys = ad.Zero.from_value(ys.aval) if ad.is_undefined_primal(ys) else None
        ct_y = ad.Zero.from_value(y.aval) if ad.is_undefined_primal(y) else None
    else:
        zeros = jnp.zeros_like(y)
        ct_ys = at_set(ct_out, i, zeros)
        ct_y = ct_out[i]
    return [ct_ys, None, ct_y]


def _at_set_batch(inputs, batch_axes):
    ys, i, y = inputs
    b_ys, b_i, b_y = batch_axes
    if b_i is not batching.not_mapped:
        # If `i` is batched then silently sample an arbitrary element (the first) and
        # use that. If we get here it should only be because some transformation rule
        # has tried to be annoying and force-batch every array (broadcasting out
        # unbatched elements).
        i = lax.index_in_dim(i, 0, axis=b_i, keepdims=False)

    if b_ys is batching.not_mapped:
        if b_y is batching.not_mapped:
            b_out = batching.not_mapped
        else:
            ys = jnp.expand_dims(ys, b_y + 1)
            ys = jnp.broadcast_to(ys, (ys.shape[0],) + y.shape)
            b_out = b_y + 1
    else:
        if b_ys == 0:
            ys = jnp.swapaxes(ys, 0, 1)
            b_ys = 1
        if b_y is batching.not_mapped:
            y = jnp.expand_dims(y, b_ys - 1)
            y = jnp.broadcast_to(y, ys.shape[1:])
        else:
            y = jnp.moveaxis(y, b_y, b_ys - 1)
        b_out = b_ys
    return at_set(ys, i, y), b_out


_at_set_p = jax.core.Primitive("at_set")
_at_set_p.def_impl(_at_set_impl)
_at_set_p.def_abstract_eval(_at_set_abstract)
ad.primitive_jvps[_at_set_p] = _at_set_jvp
ad.primitive_transposes[_at_set_p] = _at_set_transpose
batching.primitive_batchers[_at_set_p] = _at_set_batch
mlir.register_lowering(_at_set_p, mlir.lower_fun(_at_set_impl, multiple_results=False))


def at_set(
    xs: PyTree[Shaped[Array, " dim *_rest"]],
    i: Union[int, Int[Array, ""]],
    x: PyTree[Shaped[Array, " *_rest"]],
):
    """Like `xs.at[i].set(x)`. Used for updating state during a loop.

    !!! warning

        This function will give silently wrong results if you attempt to vmap down `i`.

    Differences to using `xs.at[i].step(x)`:
    - It uses `lax.dynamic_update_index_in_dim` rather than `scatter`. This will be
        lowered to a true in-place update when a scatter sometimes isn't.
    - It assumes that you never vmap down `i`.
    - It operates on PyTrees for `x` and `xs`.
    """

    def _bind(ys, y):
        ys_shape = jnp.shape(ys)
        assert jnp.ndim(ys) == jnp.ndim(y) + 1
        assert ys_shape[0] > 0
        assert ys_shape[1:] == jnp.shape(y)
        return _at_set_p.bind(ys, i, y)

    return jtu.tree_map(_bind, xs, x)


def left_broadcast_to(arr: Array, shape: Tuple[int, ...]) -> Array:
    arr = arr.reshape(arr.shape + (1,) * (len(shape) - arr.ndim))
    return jnp.broadcast_to(arr, shape)


class ContainerMeta(type):
    def __new__(cls, name, bases, dict):
        assert "reverse_lookup" not in dict
        _dict = {}
        reverse_lookup = []
        i = 0
        for key, value in dict.items():
            if key.startswith("__") and key.endswith("__"):
                _dict[key] = value
            else:
                _dict[key] = i
                reverse_lookup.append(value)
                i += 1
        _dict["reverse_lookup"] = reverse_lookup
        return super().__new__(cls, name, bases, _dict)

    def __instancecheck__(cls, instance):
        return isinstance(instance, int) or super().__instancecheck__(instance)

    def __getitem__(cls, item):
        return cls.reverse_lookup[item]

    def __len__(cls):
        return len(cls.reverse_lookup)
