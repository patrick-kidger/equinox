# Stateful operations

These operations can be used to introduce side-effects in JAX operations, even under JIT.

Use with extreme caution, and frankly only if you know what you're doing. No, really. I won't help you debug your usage of these functions. Many JAX libraries will assume that every function is a pure function, and this breaks that.

Use cases:
- Something like [`equinox.nn.BatchNorm`][], for which we would like to save the running statistics as a side-effect.
- Implicitly passing information between loop iterations -- i.e. rather than explicitly via the `carry` argument to `lax.scan`. Perhaps you're using a third-party library that handles the `lax.scan`, but you want to pass your own information between repeated invocations.

Example:
```python
import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp

index = eqx.StateIndex()
eqx.set_state(index, jnp.array(0))

def scan_fun(_, __):
    val = eqx.get_state(index)
    val = val + 1
    eqx.set_state(index, val)
    return None, val

_, out = lax.scan(scan_fun, None, xs=None, length=5)
print(out)  = [1 2 3 4 5]
```

---

::: equinox.StateIndex
    selection:
        members: false

---

::: equinox.get_state

---

::: equinox.set_state
