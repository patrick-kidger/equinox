# Stateful operations

These operations can be used to introduce save/load JAX arrays as a side-effect of JAX operations, even under JIT.

!!! warning

    This is considered experimental.

Use cases:
- Something like [`equinox.experimental.BatchNorm`][], for which we would like to save the running statistics as a side-effect.
- Implicitly passing information between loop iterations -- i.e. rather than explicitly via the `carry` argument to `lax.scan`. Perhaps you're using a third-party library that handles the `lax.scan`, but you want to pass your own information between repeated invocations.

Example:
```python
import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp

index = eqx.experimental.StateIndex()
init = jnp.array(0)
eqx.experimental.set_state(index, init)

@jax.jit
def scan_fun(_, __):
    val = eqx.experimental.get_state(index, like=init)
    val = val + 1
    eqx.experimental.set_state(index, val)
    return None, val

_, out = lax.scan(scan_fun, None, xs=None, length=5)
print(out)  # [1 2 3 4 5]
```

---

::: equinox.experimental.StateIndex
    selection:
        members: false

---

::: equinox.experimental.get_state

---

::: equinox.experimental.set_state
