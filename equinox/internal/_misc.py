from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree, Scalar

from .._eval_shape import filter_eval_shape
from .._filters import is_array
from ._nontraceable import nonbatchable


# Old; deprecated
class ContainerMeta(type):
    reverse_lookup: dict

    def __new__(cls, name, bases, dict):
        assert "reverse_lookup" not in dict
        _dict = {}
        reverse_lookup = []
        i = 0
        for key, value in dict.items():
            if key.startswith("__") and key.endswith("__"):
                _dict[key] = value
            else:
                _dict[key] = i
                reverse_lookup.append(value)
                i += 1
        _dict["reverse_lookup"] = reverse_lookup
        return super().__new__(cls, name, bases, _dict)

    def __instancecheck__(cls, instance):
        if is_array(instance):
            return instance.shape == () and jnp.issubdtype(instance.dtype, jnp.integer)
        else:
            return isinstance(instance, int) or super().__instancecheck__(instance)

    def __getitem__(cls, item):
        return cls.reverse_lookup[item]

    def __len__(cls):
        return len(cls.reverse_lookup)


_X = TypeVar("_X")


def scan_trick(fn: Callable, intermediates: Sequence[Callable], init: _X) -> _X:
    def body(carry, step):
        out = fn(carry)
        step = nonbatchable(step)
        out = lax.switch(step, intermediates, out)
        return out, None

    intermediates = list(intermediates) + [lambda x: x]
    out, _ = lax.scan(body, init, xs=jnp.arange(len(intermediates)))
    return out


def eval_empty(fn: Callable, *inputs: PyTree[Any]) -> PyTree[Array]:
    out = filter_eval_shape(fn, *inputs)
    return jtu.tree_map(lambda x: jnp.empty(x.shape, x.dtype), out)


def eval_zero(fn: Callable, *inputs: PyTree[Any]) -> PyTree[Array]:
    out = filter_eval_shape(fn, *inputs)
    return jtu.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), out)


def eval_full(fn: Callable, *inputs: PyTree[Any], fill_value: Scalar) -> PyTree[Any]:
    out = filter_eval_shape(fn, *inputs)
    return jtu.tree_map(lambda x: jnp.full(x.shape, fill_value, x.dtype), out)
