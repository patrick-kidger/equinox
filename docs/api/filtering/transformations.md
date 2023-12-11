# Transformations

These offer an alternate API on to JAX transformations, by combining filtering with a JAX transformation.

For example, JAX uses `jax.jit(..., static_argnums=...)` to manually indicate which arguments should be treated dynamically/statically. Meanwhile `equinox.filter_jit` automatically treats all JAX/NumPy arrays dynamically, and everything else statically. Moreover, this is done at the level of individual PyTree leaves, so that unlike `jax.jit`, one argment can have both dynamic (array-valued) and static leaves.

Most users find that this is a simpler API when working with complicated PyTrees, such as produced when using Equinox modules. (But you can also still use Equinox with normal `jax.jit` etc. if you so prefer.)

Likewise, `eqx.filter_grad` will automatically differentiate all floating-point JAX arrays and treat the rest nondifferentiably, etc. Each transformation here just combines [`equinox.partition`][], `jax.{jit, ...}` and [`equinox.combine`][] together.

## Compilation

::: equinox.filter_jit

---

::: equinox.filter_device_put

---

::: equinox.filter_with_sharding_constraint

## Automatic differentiation

::: equinox.filter_grad

---

::: equinox.filter_value_and_grad

---

::: equinox.filter_jvp

---

::: equinox.filter_vjp

---

::: equinox.filter_checkpoint

---

::: equinox.filter_custom_jvp

---

::: equinox.filter_custom_vjp

## Vectorisation and parallelisation

::: equinox.filter_vmap

---

::: equinox.filter_pmap

## Callbacks

::: equinox.filter_pure_callback

# Tracing

---

::: equinox.filter_make_jaxpr

---

::: equinox.filter_eval_shape

---

::: equinox.filter_closure_convert
