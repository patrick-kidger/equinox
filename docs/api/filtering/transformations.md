# Transformations

These offer an alternate API on to JAX transformations. For example, JAX uses `jax.jit(..., static_argnums=...)` to indicate which arguments should be treated dynamically/statically. Meanwhile `equinox.filter_jit` will automatically treat JAX arrays dynamically, and everything else statically. (It does this by doing something like `eqx.partition(args, eqx.is_array)` under-the-hood.)

Generally speaking, this means producing an enhanced version of the JAX transformation, that operates on arbitrary PyTrees instead of specifically just JAX arrays.

## Just-in-time compilation

::: equinox.filter_jit

---

::: equinox.filter_make_jaxpr

---

::: equinox.filter_eval_shape

## Automatic differentiation

::: equinox.filter_grad

---

::: equinox.filter_value_and_grad

---

::: equinox.filter_jvp

---

::: equinox.filter_vjp

---

::: equinox.filter_custom_jvp

---

::: equinox.filter_custom_vjp

---

::: equinox.filter_closure_convert

## Vectorisation and parallelisation

::: equinox.filter_vmap

---

::: equinox.filter_pmap
