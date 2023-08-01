# Runtime errors

Equinox offers support for raising runtime errors.

!!! faq "How does this compare to `checkify` in core JAX?"

    1. `checkify` is not compatible with operations like `jax.jit` or `jax.lax.scan`. (It must be "functionalised" first, using `jax.experimental.checkify.checkify`, and you then need to pipe a handle to the error through your code). In contrast, Equinox's errors will "just work" without any extra effort.
    
    2. `checkify` stores all errors encountered whilst running your program, and then raises them at the end of the JIT'd region. For example this means that (the JAX equivalent of) the following pseudocode will still end up in the infinite loop (because the end of the computation never arrives):

        ```python
        if h < 0:
            error()
        while t < t_max:
            t += h
        ```

        Meanwhile, Equinox's errors do not wait until the end, so the above computation will have the correct behaviour.

!!! warning

    JAX's support for raising runtime errors is technically only experimental. In practice, this nonetheless seems to be stable enough that these are part of the public API for Equinox.

::: equinox.error_if

::: equinox.branched_error_if
