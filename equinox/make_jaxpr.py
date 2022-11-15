from typing import Callable

import jax
import jax.tree_util as jtu

from .filters import combine, is_array, partition
from .module import Module, Static


def _is_struct(x):
    return is_array(x) or (hasattr(x, "shape") and hasattr(x, "dtype"))


class _MakeJaxpr(Module):
    fn: Callable

    def __call__(self, *args, **kwargs):
        dynamic, static = partition((args, kwargs), _is_struct)
        dynamic_flat, dynamic_treedef = jtu.tree_flatten(dynamic)

        def _fn(*_dynamic_flat):
            _dynamic = jtu.tree_unflatten(dynamic_treedef, _dynamic_flat)
            _args, _kwargs = combine(_dynamic, static)
            _out = self.fn(*_args, **_kwargs)
            _out_dynamic, _out_static = partition(_out, is_array)
            return _out_dynamic, Static(_out_static)

        jaxpr, out_struct = jax.make_jaxpr(_fn, return_shape=True)(*dynamic_flat)
        dynamic_out_struct, static_out = out_struct
        static_out = static_out.value
        return jaxpr, dynamic_out_struct, static_out


def filter_make_jaxpr(fun):
    """As `jax.make_jaxpr`, but accepts arbitrary PyTrees as input and output.

    **Arguments:**

    - `fun`: The function `fun(*arg, **kwargs)` whose jaxpr is to be computed. Its
        positional and keyword arguments may be anything, as can its return value.

    **Returns:**

    A wrapped version of `fun`, that when applied to example arguments
    `*args, **kwargs`, will return a 3-tuple of:
    - A `ClosedJaxpr` representing the evaluation of that function on those arguments.
    - A `PyTree[jax.ShapeDtypeStruct]` representing the output shape and dtype of the
        result.
    - A `PyTree[Any]` representing any non-array outputs from `fun`.

    The example arguments may be either JAX/NumPy arrays, or anything with `.shape` and
    `.dtype` fields (typically `jax.ShapeDtypeStruct`s). Python builtins (`int`,
    `float`, `bool`, `complex`) are treated as static inputs; wrap them in JAX/NumPy
    arrays if you would like them to be traced.
    """
    return _MakeJaxpr(fun)
