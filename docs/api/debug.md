# Debugging tools

Both Equinox and JAX provide a number of debugging tools.

## Common sources of NaNs

A common source of NaNs on the forward pass is calling `jnp.log` or `jnp.sqrt` on a negative number, or when dividing by zero. If you get a NaN whilst using these operations, check their inputs carefully (e.g. with `jax.debug.print`).

A common source of NaNs when backpropagating is when using one of the above operations with a `jnp.where`, for example `y = jnp.where(x > 0, jnp.log(x), 0)`. In this case the NaN is created on the forward pass, but is then masked by the `jnp.where`. Unfortunately, when backpropagating, the order of the `log` and the `where` is flipped -- and the NaN is no longer masked! The solution is to use the "double where" trick: bracket your computation by a `where` on both sides. For this example, `safe_x = jnp.where(x > 0, x, 1); y = jnp.where(x > 0, jnp.log(safe_x), 0)`. This ensures that the NaN is never created in the first place at all.

## JAX tools

JAX itself provides the following tools:

- the `jax.debug.print` function, for printing results under JIT.
- the `jax.debug.breakpoint` function, for opening a debugger under JIT.
- the `JAX_DEBUG_NANS=1` environment variable, for halting the computation once a NaN is encountered. This works best for NaNs encountered on the forward pass and outside of loops. If your NaN occurs on the backward pass only, then try [`equinox.debug.backward_nan`][] below. If the NaN occurs inside of a loop, then consider pairing this with `JAX_DISABLE_JIT=1`. (Many loops are implicitly jit'd.)
- the `JAX_DISABLE_JIT=1` environment variable, for running the computation without JIT. This will be *much* slower, so this isn't always practical.
- the `JAX_TRACEBACK_FILTERING=off` environment variable, which means errors and debuggers will include JAX and Equinox internals. (Which by default are filtered out.)

## Equinox tools

::: equinox.debug.announce_transform

---

::: equinox.debug.backward_nan

---

::: equinox.debug.breakpoint_if

---

::: equinox.debug.store_dce

::: equinox.debug.inspect_dce

---

::: equinox.debug.assert_max_traces

::: equinox.debug.get_num_traces

## Runtime errors

If you are getting a runtime error from [`equinox.error_if`][], then you can control the on-error behaviour via the environment variable `EQX_ON_ERROR`. If ran from `jax.jit` then this will be a long error message starting `jaxlib.xla_extension.XlaRuntimeError: INTERNAL: Generated function failed: CpuCallback error: RuntimeError: ...`; if ran from `eqx.filter_jit` then some of the extra boilerplate will be removed from the error message, and it will simply start with `jaxlib.xla_extension.XlaRuntimeError: ...`.

In particular, setting `EQX_ON_ERROR=breakpoint` will open a `jax.debug.breakpoint` where the error arises. See the [runtime errors](./errors.md) for more information and for other values of this environment variable.
