import functools as ft
import types
import typing
import warnings
from typing import Any, Callable, Dict

import jax

from .custom_types import BoolAxisSpec, PyTree, sentinel
from .doc_utils import doc_strip_annotations
from .filters import combine, is_array, is_inexact_array, partition
from .module import Module, module_update_wrapper


class _ValueAndGradWrapper(Module):
    _fun: Callable
    _arg: PyTree[BoolAxisSpec]
    _gradkwargs: Dict[str, Any]

    # Try to avoid clashes with existing argument names.
    # TODO: use "/" once we're on Python 3.8.
    def __call__(__self, __x, *args, **kwargs):
        @ft.partial(jax.value_and_grad, argnums=0, **__self._gradkwargs)
        def fun_value_and_grad(_diff_x, _nondiff_x, *_args, **_kwargs):
            _x = combine(_diff_x, _nondiff_x)
            return __self._fun(_x, *_args, **_kwargs)

        diff_x, nondiff_x = partition(__x, __self._arg)
        return fun_value_and_grad(diff_x, nondiff_x, *args, **kwargs)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return jax.tree_util.Partial(self, instance)


class _GradWrapper(Module):
    _fun_value_and_grad: _ValueAndGradWrapper
    _has_aux: bool

    def __call__(__self, *args, **kwargs):
        value, grad = __self._fun_value_and_grad(*args, **kwargs)
        if __self._has_aux:
            _, aux = value
            return grad, aux
        else:
            return grad

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return jax.tree_util.Partial(self, instance)


@doc_strip_annotations
def filter_value_and_grad(
    fun: Callable = sentinel,
    *,
    arg: PyTree[BoolAxisSpec] = is_inexact_array,
    **gradkwargs,
) -> Callable:
    """As [`equinox.filter_grad`][], except that it is `jax.value_and_grad` that is
    wrapped.
    """

    if fun is sentinel:
        return ft.partial(filter_value_and_grad, arg=arg, **gradkwargs)

    filter_spec = gradkwargs.pop("filter_spec", None)
    if filter_spec is not None:
        warnings.warn("For brevity the `filter_spec` argument has been renamed `arg`")
        arg = filter_spec

    argnums = gradkwargs.pop("argnums", None)
    if argnums is not None:
        raise ValueError(
            "`argnums` should not be passed. If you need to differentiate "
            "multiple objects then collect them into a tuple and pass that "
            "as the first argument."
        )

    return module_update_wrapper(_ValueAndGradWrapper(fun, arg, gradkwargs), fun)


@doc_strip_annotations
def filter_grad(
    fun: Callable = sentinel,
    *,
    arg: PyTree[BoolAxisSpec] = is_inexact_array,
    **gradkwargs,
):
    """Wraps together [`equinox.partition`][] and `jax.grad`.

    !!! info

        By default, all inexact (floating-point) JAX arrays are differentiated. Any
        nondifferentiable leaves will have `None` as the gradient.


    **Arguments:**

    - `fun` is a pure function to JIT compile.
    - `arg` is a PyTree whose structure should be a prefix of the structure of
        the **first** argument to `fun`. It behaves as the `filter_spec` argument to
        [`equinox.filter`][]. Truthy values will be differentiated; falsey values will
        not.
    - `**gradkwargs` are any other keyword arguments to `jax.grad`.

    **Returns:**

    A function computing the derivative of `fun` with respect to its first input. Any
    nondifferentiable leaves will have `None` as the gradient. See
    [`equinox.apply_updates`][] for a convenience function that will only attempt to
    apply non-`None` updates.

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

    if fun is sentinel:
        return ft.partial(filter_grad, arg=arg, **gradkwargs)

    has_aux = gradkwargs.get("has_aux", False)

    fun_value_and_grad = filter_value_and_grad(fun, arg=arg, **gradkwargs)
    return module_update_wrapper(_GradWrapper(fun_value_and_grad, has_aux), fun)


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
