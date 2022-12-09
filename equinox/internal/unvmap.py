import jax
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.interpreters.xla as xla
import jax.numpy as jnp


# unvmap_all

unvmap_all_p = jax.core.Primitive("unvmap_all")


def unvmap_all(x):
    """As `jnp.all`, but ignores batch dimensions."""
    return unvmap_all_p.bind(x)


def _unvmap_all_impl(x):
    return jnp.all(x)


def _unvmap_all_abstract_eval(x):
    return jax.ShapedArray(shape=(), dtype=jax.numpy.bool_.dtype)


def _unvmap_all_batch(x, batch_axes):
    (x,) = x
    return unvmap_all(x), batching.not_mapped


unvmap_all_p.def_impl(_unvmap_all_impl)
unvmap_all_p.def_abstract_eval(_unvmap_all_abstract_eval)
batching.primitive_batchers[unvmap_all_p] = _unvmap_all_batch
if hasattr(xla, "lower_fun"):
    xla.register_translation(
        unvmap_all_p,
        xla.lower_fun(_unvmap_all_impl, multiple_results=False, new_style=True),
    )
mlir.register_lowering(
    unvmap_all_p,
    mlir.lower_fun(_unvmap_all_impl, multiple_results=False),
)

# unvmap_any

unvmap_any_p = jax.core.Primitive("unvmap_any")


def unvmap_any(x):
    """As `jnp.any`, but ignores batch dimensions."""
    return unvmap_any_p.bind(x)


def _unvmap_any_impl(x):
    return jnp.any(x)


def _unvmap_any_abstract_eval(x):
    return jax.ShapedArray(shape=(), dtype=jax.numpy.bool_.dtype)


def _unvmap_any_batch(x, batch_axes):
    (x,) = x
    return unvmap_any(x), batching.not_mapped


unvmap_any_p.def_impl(_unvmap_any_impl)
unvmap_any_p.def_abstract_eval(_unvmap_any_abstract_eval)
batching.primitive_batchers[unvmap_any_p] = _unvmap_any_batch
if hasattr(xla, "lower_fun"):
    xla.register_translation(
        unvmap_any_p,
        xla.lower_fun(_unvmap_any_impl, multiple_results=False, new_style=True),
    )
mlir.register_lowering(
    unvmap_any_p,
    mlir.lower_fun(_unvmap_any_impl, multiple_results=False),
)

# unvmap_max

unvmap_max_p = jax.core.Primitive("unvmap_max")


def unvmap_max(x):
    """As `jnp.max`, but ignores batch dimensions."""
    return unvmap_max_p.bind(x)


def _unvmap_max_impl(x):
    return jnp.max(x)


def _unvmap_max_abstract_eval(x):
    return jax.ShapedArray(shape=(), dtype=x.dtype)


def _unvmap_max_batch(x, batch_axes):
    (x,) = x
    return unvmap_max(x), batching.not_mapped


unvmap_max_p.def_impl(_unvmap_max_impl)
unvmap_max_p.def_abstract_eval(_unvmap_max_abstract_eval)
batching.primitive_batchers[unvmap_max_p] = _unvmap_max_batch
if hasattr(xla, "lower_fun"):
    xla.register_translation(
        unvmap_max_p,
        xla.lower_fun(_unvmap_max_impl, multiple_results=False, new_style=True),
    )
mlir.register_lowering(
    unvmap_max_p,
    mlir.lower_fun(_unvmap_max_impl, multiple_results=False),
)
