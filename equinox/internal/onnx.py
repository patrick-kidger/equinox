import jax.numpy as jnp

from .finalise_jaxpr import finalise_fn


def to_onnx(fn):
    import jax.experimental.jax2tf as jax2tf
    import tensorflow as tf
    import tf2onnx

    def _to_onnx(*args):
        finalised_fn = finalise_fn(fn)
        tf_fn = tf.function(jax2tf.convert(finalised_fn, enable_xla=False))
        tf_args = [tf.TensorSpec(jnp.shape(x), jnp.result_type(x)) for x in args]
        onnx_fn = tf2onnx.convert.from_function(tf_fn, input_signature=tf_args)
        return onnx_fn

    return _to_onnx
