# Progress Meters

Equinox provides progress meters that work under JIT. These offer the ability to have some kind of output indicating how far along a process has progressed. For example, to display a text output every now and again, or to fill a [tqdm](https://github.com/tqdm/tqdm) progress bar.

Typically you should write your function as consuming an [`equinox.AbstractProgressMeter`][], and then calling its `.init`, `.step` and `.close` methods at the appropriate times.

!!! Example

    ```python
    import time

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    def sleep(val):
        """Sleeps for 0.005 seconds, under JIT"""
        return jax.pure_callback(lambda x: [time.sleep(0.005), x][1], val, val)

    @eqx.filter_jit
    def my_function(
        val0: jax.Array,
        meter: eqx.AbstractProgressMeter,
        length: int,
    ) -> jax.Array:

        def step(carry, index):
            val, state = carry
            state = meter.step(state, index / length)
            val = sleep(val)
            return (val + 1, state), None

        state0 = meter.init()
        (val1, state1), _ = jax.lax.scan(step, (val0, state0), xs=jnp.arange(length))
        meter.close(state1)
        return val1

    my_function(jnp.array(0), eqx.TqdmProgressMeter(), length=1000)
    ```

---

??? abstract "`equinox.AbstractProgressMeter`"

    ::: equinox.AbstractProgressMeter
        options:
            members:
                - init
                - step
                - close

---

::: equinox.NoProgressMeter
    options:
        members:
            - __init__

::: equinox.TextProgressMeter
    options:
        members:
            - __init__

::: equinox.TqdmProgressMeter
    options:
        members:
            - __init__
