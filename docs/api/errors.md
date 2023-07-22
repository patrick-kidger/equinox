# Runtime errors

Equinox offers support for raising runtime errors.

!!! faq

    **How does this compare to `checkify` in core JAX?**

    JAX's `checkify` stores all errors encountered during the program, and then raises them at the end of the computation. This means, for example, that (the JAX equivalent of) the following pseudocode will still result in an infinite loop:
    ```python
    if h < 0:
        error()
    while t < t_max:
        t += h
    ```
    as the while loop will never terminate, so the "end of the computation" never arrives.

    In contrast, Equinox's errors will be raised eagerly, so the above computation will have the correct behaviour.

!!! warning

    JAX's support for raising runtime errors is technically only experimental. In practice, this nonetheless seems to be stable enough that these are part of the public API for Equinox.

::: equinox.error_if

::: equinox.branched_error_if
