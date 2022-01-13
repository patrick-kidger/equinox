# Filtered transformations

These typically combine [`equinox.partition`][], a filter function, and a JAX transformation, all together.

Practically speaking these are usually the only kind of filtering you ever have to use. (But it's good to understand what e.g. [`equinox.partition`][] and [`equinox.is_array`][] are doing under the hood, just so that these don't seem too magical.)

::: equinox.filter_jit

---

::: equinox.filter_grad

---

::: equinox.filter_value_and_grad

---

::: equinox.filter_custom_vjp
    selection:
        members: false
