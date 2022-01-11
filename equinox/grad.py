import functools as ft

import jax

from .deprecated import deprecated
from .filters import (
    combine,
    is_array,
    is_inexact_array,
    merge,
    partition,
    split,
    validate_filters,
)


def filter_value_and_grad(
    fun, *, filter_spec=is_inexact_array, argnums=None, **gradkwargs
):
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


def filter_grad(fun, *, filter_spec=is_inexact_array, has_aux=False, **gradkwargs):
    fun_value_and_grad = filter_value_and_grad(
        fun, filter_spec=filter_spec, has_aux=has_aux, **gradkwargs
    )

    def fun_grad(*args, **kwargs):
        value, grad = fun_value_and_grad(*args, **kwargs)
        if has_aux:
            value, aux = value
            return aux, grad
        else:
            return grad

    return fun_grad


class filter_custom_vjp:
    def __init__(self, fn):
        self.fn = fn
        self.fn_wrapped = None

    def defvjp(self, fn_fwd, fn_bwd):
        def fn_wrapped(
            nonarray_vjp_arg, nonarray_args_kwargs, array_vjp_arg, array_args_kwargs
        ):
            vjp_arg = combine(nonarray_vjp_arg, array_vjp_arg)
            args, kwargs = combine(nonarray_args_kwargs, array_args_kwargs)
            return self.fn(vjp_arg, *args, **kwargs)

        def fn_fwd_wrapped(
            nonarray_vjp_arg, nonarray_args_kwargs, array_vjp_arg, array_args_kwargs
        ):
            vjp_arg = combine(nonarray_vjp_arg, array_vjp_arg)
            args, kwargs = combine(nonarray_args_kwargs, array_args_kwargs)
            out, residuals = fn_fwd(vjp_arg, *args, **kwargs)
            return out, (residuals, array_vjp_arg, array_args_kwargs)

        def fn_bwd_wrapped(nonarray_vjp_arg, nonarray_args_kwargs, residuals, grad_out):
            residuals, array_vjp_arg, array_args_kwargs = residuals
            vjp_arg = combine(nonarray_vjp_arg, array_vjp_arg)
            args, kwargs = combine(nonarray_args_kwargs, array_args_kwargs)
            out = fn_bwd(residuals, grad_out, vjp_arg, *args, **kwargs)
            return out, None  # None is the gradient through array_args_kwargs

        fn_wrapped = jax.custom_vjp(fn_wrapped, nondiff_argnums=(0, 1))
        fn_wrapped.defvjp(fn_fwd_wrapped, fn_bwd_wrapped)
        self.fn_wrapped = fn_wrapped

    def __call__(self, vjp_arg, /, *args, **kwargs):
        if self.fn_wrapped is None:
            raise RuntimeError(f"defvjp not yet called for {self.fn.__name__}")
        array_vjp_arg, nonarray_vjp_arg = partition(vjp_arg, is_array)
        array_args_kwargs, nonarray_args_kwargs = partition((args, kwargs), is_array)
        return self.fn_wrapped(
            nonarray_vjp_arg, nonarray_args_kwargs, array_vjp_arg, array_args_kwargs
        )


#
# Deprecated
#


@deprecated(in_favour_of=filter_value_and_grad)
def value_and_grad_f(fun, *, filter_fn=None, filter_tree=None, argnums=0, **gradkwargs):
    if isinstance(argnums, int):
        unwrap = True
        argnums = (argnums,)
        if filter_tree is not None:
            filter_tree = (filter_tree,)
    else:
        unwrap = False

    validate_filters("value_and_grad_f", filter_fn, filter_tree)

    @ft.partial(jax.value_and_grad, argnums=argnums, **gradkwargs)
    def f_value_and_grad(*args, **kwargs):
        *args, notes = args
        args = list(args)
        for i, (arg_nograd, which, treedef) in notes.items():
            arg_grad = args[i]
            arg = merge(arg_grad, arg_nograd, which, treedef)
            args[i] = arg
        return fun(*args, **kwargs)

    def f_value_and_grad_wrapper(*args, **kwargs):
        args = list(args)
        notes = {}
        for j, i in enumerate(argnums):
            arg = args[i]
            if filter_fn is None:
                # implies filter_tree is not None
                arg_grad, arg_nograd, which, treedef = split(
                    arg, filter_tree=filter_tree[j]
                )
            else:
                arg_grad, arg_nograd, which, treedef = split(arg, filter_fn=filter_fn)
            args[i] = arg_grad
            notes[i] = (arg_nograd, which, treedef)
        value, grad = f_value_and_grad(*args, notes, **kwargs)
        grad = list(grad)
        for j, i in enumerate(argnums):
            g = grad[j]
            arg_nograd, which, treedef = notes[i]
            none_grad = [None for _ in arg_nograd]
            grad[j] = merge(g, none_grad, which, treedef)
        if unwrap:
            (grad,) = grad
        return value, grad

    return f_value_and_grad_wrapper


@deprecated(in_favour_of=filter_grad)
def gradf(fun, *, has_aux=False, **gradkwargs):
    f_value_and_grad = value_and_grad_f(fun, has_aux=has_aux, **gradkwargs)

    def f_grad(*args, **kwargs):
        value, grad = f_value_and_grad(*args, **kwargs)
        if has_aux:
            value, aux = value
            return aux, grad
        else:
            return grad

    return f_grad
