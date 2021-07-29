import functools as ft
import jax
import jax.numpy as jnp

from .helpers import is_inexact_array, split, merge


def auto_value_and_grad(fun, *, argnums=0, filter_fn=is_inexact_array, **gradkwargs):
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
            zero = [jnp.zeros_like(x) for x in arg_nograd]
            grad[j] = merge(g, zero, which, treedef)
        if unwrap:
            grad, = grad
        return value, grad

    return f_value_and_grad_wrapper


def autograd(fun, *, has_aux=False, **gradkwargs):
    f_value_and_grad = auto_value_and_grad(fun, **gradkwargs)

    def f_grad(*args, **kwargs):
        value, grad = f_value_and_grad(*args, **kwargs)
        if has_aux:
            value, aux = value
            return aux, grad
        else:
            return grad

    return f_grad
