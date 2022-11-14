# Filtered transformations

These typically combine [`equinox.partition`][], [`equinox.combine`][], and a JAX transformation, all together.

Generally speaking, this means producing an enhanced version of the JAX transformation, that operates on arbitrary PyTrees instead of specifically just JAX arrays.

Practically speaking these are usually the only kind of filtering you ever have to use. (But it's good to understand what e.g. [`equinox.partition`][] and [`equinox.is_array`][] are doing under the hood, just so that these don't seem too magical.)

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
