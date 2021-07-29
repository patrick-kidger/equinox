import functools as ft
import jax

from .filters import split, merge


def value_and_grad_f(fun, *, filter_fn, argnums=0, **gradkwargs):
    if isinstance(argnums, int):
        unwrap = True
        argnums = (argnums,)
    else:
        unwrap = False

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
        for i in argnums:
            arg = args[i]
            arg_grad, arg_nograd, which, treedef = split(arg, filter_fn)
            args[i] = arg_grad
            notes[i] = (arg_nograd, which, treedef)
        value, grad = f_value_and_grad(*args, notes, **kwargs)
        grad = list(grad)
        for j, i in enumerate(argnums):
            g = grad[j]
            arg_nograd, which, treedef = notes[i]
            zero = [0 for _ in arg_nograd]
            grad[j] = merge(g, zero, which, treedef)
        if unwrap:
            grad, = grad
        return value, grad

    return f_value_and_grad_wrapper


def gradf(fun, *, has_aux=False, **gradkwargs):
    f_value_and_grad = value_and_grad_f(fun, **gradkwargs)

    def f_grad(*args, **kwargs):
        value, grad = f_value_and_grad(*args, **kwargs)
        if has_aux:
            value, aux = value
            return aux, grad
        else:
            return grad

    return f_grad
