import functools as ft
from typing import Callable

import jax

from .compile_utils import Static
from .filters import combine, is_array_like, partition


def _filter(x):
    return isinstance(x, jax.ShapeDtypeStruct) or is_array_like(x)


def filter_eval_shape(fun: Callable, *args, **kwargs):
    """As `jax.eval_shape`, but allows any Python object as inputs and outputs.

    (`jax.eval_shape` is constrained to only work with JAX arrays, Python float/int/etc.)
    """

    def _fn(_static, _dynamic):
        _args, _kwargs = combine(_static, _dynamic)
        _out = fun(*_args, **_kwargs)
        _dynamic_out, _static_out = partition(_out, _filter)
        return _dynamic_out, Static(_static_out)

    dynamic, static = partition((args, kwargs), _filter)
    dynamic_out, static_out = jax.eval_shape(ft.partial(_fn, static), dynamic)
    return combine(dynamic_out, static_out.value)
