import functools as ft
import types
import typing

import jax

from .filters import combine, is_array, is_inexact_array, partition


def filter_value_and_grad(
    fun, *, filter_spec=is_inexact_array, argnums=None, **gradkwargs
):
    """As [`equinox.filter_grad`][], except that it is `jax.value_and_grad` that is
    wrapped.
    """
    if argnums is not None:
        raise ValueError(
            "`argnums` should not be passed. If you need to differentiate "
            "multiple objects then collect them into a tuple and pass that "
            "as the first argument."
        )

    @ft.partial(jax.value_and_grad, argnums=0, **gradkwargs)
    def fun_value_and_grad(diff_x, nondiff_x, *args, **kwargs):
        x = combine(diff_x, nondiff_x)
        return fun(x, *args, **kwargs)

    def fun_value_and_grad_wrapper(x, *args, **kwargs):
        diff_x, nondiff_x = partition(x, filter_spec)
        return fun_value_and_grad(diff_x, nondiff_x, *args, **kwargs)

    return fun_value_and_grad_wrapper


def filter_grad(fun, *, filter_spec=is_inexact_array, **gradkwargs):
    """Wraps together [`equinox.partition`][] and `jax.grad`.

    **Arguments:**

    - `fun` is a pure function to JIT compile.
    - `filter_spec` is a PyTree whose structure should be a prefix of the structure of
        the **first** argument to `fun`. It behaves as the `filter_spec` argument to
        [`equinox.filter`][]. Truthy values will be differentiated; falsey values will
        not.
    - `**gradkwargs` are any other keyword arguments to `jax.grad`.

    **Returns:**

    A function computing the derivative of `fun` with respect to its first input. Any
    nondifferentiable leaves will have `None` as the gradient. See
    [`equinox.apply_updates`][] for a convenience function that will only attempt to
    apply non-`None` updates.

    !!! info

        A very important special case is to trace all inexact (i.e. floating point)
        JAX arrays and treat all other objects as nondifferentiable.

        This is accomplished with `filter_spec=equinox.is_inexact_array`, which is the
        default.

    !!! tip

        If you need to differentiate multiple objects, then put them together into a
        tuple and pass that through the first argument:
        ```python
        # We want to differentiate `func` with respect to both `x` and `y`.
        def func(x, y):
            ...

        @equinox.filter_grad
        def grad_func(x__y):
            x, y = x__y
            return func(x, y)
        ```
    """

    has_aux = gradkwargs.get("has_aux", False)

    fun_value_and_grad = filter_value_and_grad(
        fun, filter_spec=filter_spec, **gradkwargs
    )

    def fun_grad(*args, **kwargs):
        value, grad = fun_value_and_grad(*args, **kwargs)
        if has_aux:
            _, aux = value
            return grad, aux
        else:
            return grad

    return fun_grad


class filter_custom_vjp:
    """Provides an easier API for `jax.custom_vjp`, by using filtering.

    Usage is:
    ```python
    @equinox.filter_custom_vjp
    def fn(vjp_arg, *args, **kwargs):
        # vjp_arg is some PyTree of arbitrary Python objects.
        # args, kwargs contain arbitrary Python objects.
        ...
        return obj  # some PyTree of arbitrary Python objects.

    def fn_fwd(vjp_arg, *args, **kwargs):
        ...
        # Should return `obj` as before. `residuals` can be any collection of JAX
        # arrays you want to keep around for the backward pass.
        return obj, residuals

    def fn_bwd(residuals, grad_obj, vjp_arg, *args, **kwargs):
        # grad_obj will have `None` as the gradient for any leaves of `obj` that were
        # not JAX arrays
        ...
        # grad_vjp_arg should have `None` as the gradient for any leaves of `vjp_arg`
        # that were not JAX arrays.
        return grad_vjp_arg

    fn.defvjp(fn_fwd, fn_bwd)
    ```

    The key differences to `jax.custom_vjp` are that:

    - Only the gradient of the first argument, `vjp_arg`, should be computed on the
        backward pass. Everything else will automatically have zero gradient.
    - You do not need to distinguish differentiable from nondifferentiable manually.
        Instead you should return gradients for all inexact JAX arrays in the first
        argument. (And just put `None` on every other leaf of the PyTree.)
    - As a convenience, all of the inputs from the forward pass are additionally made
        available to you on the backward pass.

    !!! tip

        If you need gradients with respect to multiple arguments, then just pack them
        together as a tuple via the first argument `vjp_arg`. (See also
        [`equinox.filter_grad`][] for a similar trick.)
    """

    def __init__(self, fn):
        self.fn = fn
        self.fn_wrapped = None

    def defvjp(self, fn_fwd, fn_bwd):
        def fn_wrapped(
            nonarray_vjp_arg,
            nonarray_args_kwargs,
            diff_array_vjp_arg,
            nondiff_array_vjp_arg,
            array_args_kwargs,
        ):
            vjp_arg = combine(
                nonarray_vjp_arg, diff_array_vjp_arg, nondiff_array_vjp_arg
            )
            args, kwargs = combine(nonarray_args_kwargs, array_args_kwargs)
            return self.fn(vjp_arg, *args, **kwargs)

        def fn_fwd_wrapped(
            nonarray_vjp_arg,
            nonarray_args_kwargs,
            diff_array_vjp_arg,
            nondiff_array_vjp_arg,
            array_args_kwargs,
        ):
            vjp_arg = combine(
                nonarray_vjp_arg, diff_array_vjp_arg, nondiff_array_vjp_arg
            )
            args, kwargs = combine(nonarray_args_kwargs, array_args_kwargs)
            out, residuals = fn_fwd(vjp_arg, *args, **kwargs)
            return out, (
                residuals,
                diff_array_vjp_arg,
                nondiff_array_vjp_arg,
                array_args_kwargs,
            )

        def fn_bwd_wrapped(nonarray_vjp_arg, nonarray_args_kwargs, residuals, grad_out):
            (
                residuals,
                diff_array_vjp_arg,
                nondiff_array_vjp_arg,
                array_args_kwargs,
            ) = residuals
            vjp_arg = combine(
                nonarray_vjp_arg, diff_array_vjp_arg, nondiff_array_vjp_arg
            )
            args, kwargs = combine(nonarray_args_kwargs, array_args_kwargs)
            out = fn_bwd(residuals, grad_out, vjp_arg, *args, **kwargs)
            if jax.tree_structure(out) != jax.tree_structure(diff_array_vjp_arg):
                raise RuntimeError(
                    "custom_vjp gradients must have the same structure as "
                    "`equinox.filter(vjp_arg, equinox.is_inexact_array)`, where "
                    "`vjp_arg` is the first argument used in the forward pass."
                )
            # None is the gradient through nondiff_array_vjp_arg and array_args_kwargs
            return out, None, None

        fn_wrapped = jax.custom_vjp(fn_wrapped, nondiff_argnums=(0, 1))
        fn_wrapped.defvjp(fn_fwd_wrapped, fn_bwd_wrapped)
        self.fn_wrapped = fn_wrapped

    def __call__(__self, __vjp_arg, *args, **kwargs):
        # Try and avoid name collisions with the arguments of the wrapped function.
        # TODO: once we switch to Python 3.8, use (self, vjp_arg, /, *args, **kwargs).
        self = __self
        vjp_arg = __vjp_arg
        del __self, __vjp_arg
        if self.fn_wrapped is None:
            raise RuntimeError(f"defvjp not yet called for {self.fn.__name__}")
        array_vjp_arg, nonarray_vjp_arg = partition(vjp_arg, is_array)
        diff_array_vjp_arg, nondiff_array_vjp_arg = partition(
            array_vjp_arg, is_inexact_array
        )
        array_args_kwargs, nonarray_args_kwargs = partition((args, kwargs), is_array)
        return self.fn_wrapped(
            nonarray_vjp_arg,
            nonarray_args_kwargs,
            diff_array_vjp_arg,
            nondiff_array_vjp_arg,
            array_args_kwargs,
        )


if getattr(typing, "GENERATING_DOCUMENTATION", False):
    _filter_custom_vjp_doc = filter_custom_vjp.__doc__

    def defvjp(fn_fwd, fn_bwd):
        pass

    def filter_custom_vjp(fn):
        return types.SimpleNamespace(defvjp=defvjp)

    filter_custom_vjp.__doc__ = _filter_custom_vjp_doc
