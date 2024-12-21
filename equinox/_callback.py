import jax

from ._filters import combine, is_array, partition
from ._tree import tree_equal


def _is_struct(x):
    return hasattr(x, "shape") and hasattr(x, "dtype")


def filter_pure_callback(
    callback,
    *args,
    result_shape_dtypes,
    sharding=None,
    vmap_method=None,
    vectorized=None,
    **kwargs,
):
    """Calls a Python function inside a JIT region. As `jax.pure_callback` but accepts
    arbitrary Python objects as inputs and outputs. (Not just JAXable types.)

    Note that unlike `jax.pure_callback`, then the `result_shape_dtypes` argument must
    be passed as a keyword argument.

    **Arguments:**

    - `callback`: The Python function to call.
    - `*args`, `**kwargs`: The function will be called as `callback(*args, **kwargs)`.
        These may be arbitrary Python objects.
    - `result_shape_dtypes`: A PyTree specifying the output of `callback`. It should
        have a `jax.ShapeDtypeStruct` in place of any JAX arrays. Note that unlike
        `jax.pure_callback`, this must be passed as a keyword-only argument.
    - `sharding`: optional sharding that specifies the device from which the callback
        should be invoked.
    - `vmap_method`, `vectorized`: these specify how the callback transforms under
        `vmap()` as described in the documentation for `jax.pure_callback`.

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

    keywords = {}
    if sharding is not None:
        keywords["sharding"] = sharding
    if vectorized is not None:
        keywords["vectorized"] = vectorized
    if vmap_method is not None:
        keywords["vmap_method"] = vmap_method
    dynamic_out = jax.pure_callback(_callback, dynamic_struct, dynamic, **keywords)
    return combine(dynamic_out, static_struct)
