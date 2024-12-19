from typing import cast

import jax
import jax.core
import jax.extend.core
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Int


# unvmap_all

unvmap_all_p = jax.extend.core.Primitive("unvmap_all")


def unvmap_all(x: Bool[ArrayLike, "..."]) -> Bool[Array, ""]:
    """As `jnp.all`, but ignores batch dimensions."""
    return cast(Array, unvmap_all_p.bind(x))


def _unvmap_all_impl(x):
    return jnp.all(x)


def _unvmap_all_abstract_eval(x):
    return jax.core.ShapedArray(shape=(), dtype=jax.numpy.bool_.dtype)


def _unvmap_all_batch(x, batch_axes):
    (x,) = x
    return unvmap_all(x), batching.not_mapped


unvmap_all_p.def_impl(_unvmap_all_impl)
unvmap_all_p.def_abstract_eval(_unvmap_all_abstract_eval)
batching.primitive_batchers[unvmap_all_p] = _unvmap_all_batch
mlir.register_lowering(
    unvmap_all_p,
    mlir.lower_fun(_unvmap_all_impl, multiple_results=False),
)

# unvmap_any

unvmap_any_p = jax.extend.core.Primitive("unvmap_any")


def unvmap_any(x: Bool[ArrayLike, "..."]) -> Bool[Array, ""]:
    """As `jnp.any`, but ignores batch dimensions."""
    return cast(Array, unvmap_any_p.bind(x))


def _unvmap_any_impl(x):
    return jnp.any(x)


def _unvmap_any_abstract_eval(x):
    return jax.core.ShapedArray(shape=(), dtype=jax.numpy.bool_.dtype)


def _unvmap_any_batch(x, batch_axes):
    (x,) = x
    return unvmap_any(x), batching.not_mapped


unvmap_any_p.def_impl(_unvmap_any_impl)
unvmap_any_p.def_abstract_eval(_unvmap_any_abstract_eval)
batching.primitive_batchers[unvmap_any_p] = _unvmap_any_batch
mlir.register_lowering(
    unvmap_any_p,
    mlir.lower_fun(_unvmap_any_impl, multiple_results=False),
)

# unvmap_max

unvmap_max_p = jax.extend.core.Primitive("unvmap_max")


def unvmap_max(x: Int[ArrayLike, "..."]) -> Int[Array, ""]:
    """As `jnp.max`, but ignores batch dimensions."""
    return cast(Array, unvmap_max_p.bind(x))


def _unvmap_max_impl(x):
    return jnp.max(x)


def _unvmap_max_abstract_eval(x):
    return jax.core.ShapedArray(shape=(), dtype=x.dtype)


def _unvmap_max_batch(x, batch_axes):
    (x,) = x
    return unvmap_max(x), batching.not_mapped


unvmap_max_p.def_impl(_unvmap_max_impl)
unvmap_max_p.def_abstract_eval(_unvmap_max_abstract_eval)
batching.primitive_batchers[unvmap_max_p] = _unvmap_max_batch
mlir.register_lowering(
    unvmap_max_p,
    mlir.lower_fun(_unvmap_max_impl, multiple_results=False),
)
