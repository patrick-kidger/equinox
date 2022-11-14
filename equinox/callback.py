import jax

from .filters import combine, is_array, partition
from .tree import tree_equal


def _is_struct(x):
    return hasattr(x, "shape") and hasattr(x, "dtype")


def filter_pure_callback(
    callback, *args, result_shape_dtypes, vectorized=False, **kwargs
):
    """Calls a Python function inside a JIT region. As `jax.pure_callback` but accepts
    arbitrary Python objects as inputs and outputs. (Not just JAXable types.)

    **Arguments:**

        - `callback`: The Python function to call.
        - `args`, `kwargs`: The function will be called as `callback(*args, **kwargs)`.
            These may be arbitrary Python objects.
        - `result_shape_dtypes`: A PyTree specifying the output of `callback`. It should
            have a `jax.ShapeDtypeStruct` in place of any JAX arrays.
        - `vectorized`: If `True` then `callback` is batched(when transformed by `vmap`)
            by calling it directly on the batched arrays. If `False` then `callback` is
            called on each batch element individually.

    **Returns:**

    The result of `callback(*args, **kwargs)`, valid for use under JIT.
    """
    dynamic, static = partition((args, kwargs), is_array)
    dynamic_struct, static_struct = partition(result_shape_dtypes, _is_struct)

    def _callback(_dynamic):
        _args, _kwargs = combine(_dynamic, static)
        _out = callback(*_args, **_kwargs)
        _dynamic_out, _static_out = partition(_out, is_array)
        if not tree_equal(_static_out, static_struct):
            raise ValueError("Callback did not return matching static elements")
        return _dynamic_out

    dynamic_out = jax.pure_callback(
        _callback, dynamic_struct, dynamic, vectorized=vectorized
    )
    return combine(dynamic_out, static_struct)
