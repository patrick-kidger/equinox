import jax.numpy as jnp

from ._finalise_jaxpr import finalise_fn


def to_onnx(fn):
    """Export a JAX function to ONNX.

    !!! Warning

        This is experimental and may be removed or changed.

    !!! Example

        ```python
        import equinox.internal as eqxi
        import jax.numpy as jnp

        def f(x, y):
            return x + y

        onnx_f = eqxi.to_onnx(f)
        result = onnx_f(jnp.array(1), jnp.array(2))
        ```
    """
    import jax.experimental.jax2tf as jax2tf
    import tensorflow as tf  # pyright: ignore[reportMissingImports]
    import tf2onnx  # pyright: ignore[reportMissingImports]

    def _to_onnx(*args):
        finalised_fn = finalise_fn(fn)
        tf_fn = tf.function(jax2tf.convert(finalised_fn, enable_xla=False))
        tf_args = [tf.TensorSpec(jnp.shape(x), jnp.result_type(x)) for x in args]  # pyright: ignore
        onnx_fn = tf2onnx.convert.from_function(tf_fn, input_signature=tf_args)
        return onnx_fn

    return _to_onnx
