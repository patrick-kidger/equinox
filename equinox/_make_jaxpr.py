from collections.abc import Callable
from typing import Any
from typing_extensions import ParamSpec

import jax
import jax._src.traceback_util as traceback_util
import jax.extend.core
import jax.tree_util as jtu
from jaxtyping import PyTree

from ._filters import combine, is_array, partition
from ._module import Module, module_update_wrapper, Static


traceback_util.register_exclusion(__file__)


_P = ParamSpec("_P")


def _is_struct(x):
    return is_array(x) or isinstance(x, jax.ShapeDtypeStruct)


class _MakeJaxpr(Module):
    fn: Callable

    @property
    def __wrapped__(self):
        return self.fn

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


def filter_make_jaxpr(
    fun: Callable[_P, Any],
) -> Callable[
    _P, tuple[jax.extend.core.ClosedJaxpr, PyTree[jax.ShapeDtypeStruct], PyTree[Any]]
]:
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

    The example arguments to be traced may be anything with `.shape` and `.dtype`
    fields (typically JAX arrays, NumPy arrays, of `jax.ShapeDtypeStruct`s). All
    other arguments are treated statically. In particular, Python builtins (`bool`,
    `int`, `float`, `complex`) are treated as static inputs; wrap them in JAX/NumPy
    arrays if you would like them to be traced.
    """
    return module_update_wrapper(_MakeJaxpr(fun))
