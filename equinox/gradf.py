import functools as ft

import jax

from .deprecated import deprecated
from .filters import combine, merge, partition, split, validate_filters, is_inexact_array


def filter_value_and_grad(fun, *, filter_spec=is_inexact_array, argnums=None, **gradkwargs):
    if argnums is not None:
        raise ValueError("`argnums` should not be passed. If you need to differentiate "
                         "multiple objects then collect them into a tuple and pass that "
                         "as the first argument.")

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
