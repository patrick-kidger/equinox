import functools as ft
from typing import Any, Callable, Union

import jax
from jaxtyping import PyTree

from .filters import combine, is_array, partition
from .module import Static


def _filter(x):
    return isinstance(x, jax.ShapeDtypeStruct) or is_array(x)


def filter_eval_shape(
    fun: Callable[..., Any], *args, **kwargs
) -> PyTree[Union[jax.ShapeDtypeStruct, Any]]:
    """As `jax.eval_shape`, but allows any Python object as inputs and outputs.

    (`jax.eval_shape` is constrained to only work with JAX arrays, Python
    float/int/etc.)
    """

    def _fn(_static, _dynamic):
        _fun, _args, _kwargs = combine(_static, _dynamic)
        _out = _fun(*_args, **_kwargs)
        _dynamic_out, _static_out = partition(_out, _filter)
        return _dynamic_out, Static(_static_out)

    dynamic, static = partition((fun, args, kwargs), _filter)
    dynamic_out, static_out = jax.eval_shape(ft.partial(_fn, static), dynamic)
    return combine(dynamic_out, static_out.value)
