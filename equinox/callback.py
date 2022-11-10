import jax

from .filters import combine, is_array, partition
from .tree import tree_equal


def _is_struct(x):
    return hasattr(x, "shape") and hasattr(x, "dtype")


def filter_pure_callback(
    callback, *args, result_shape_dtypes, vectorized=False, **kwargs
):
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
