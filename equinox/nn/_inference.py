import jax.tree_util as jtu
from jaxtyping import PyTree

from .._tree import tree_at


def _inferences(pytree):
    is_leaf = lambda x: hasattr(x, "inference") and x is not pytree

    out = [pytree.inference] if hasattr(pytree, "inference") else []

    leaves = [x for x in jtu.tree_leaves(pytree, is_leaf=is_leaf) if is_leaf(x)]
    # Nodes with an inference flag might have sub-nodes with an inference flag.

    for x in leaves:
        out.extend(_inferences(x))
    return out


def inference_mode(pytree: PyTree, value: bool = True) -> PyTree:
    """Convenience function for setting all `inference` attributes.

    `inference` flags are used to toggle the behaviour of a number of the pre-built
    neural network layers, such as [`equinox.nn.Dropout`][] or
    [`equinox.nn.BatchNorm`][].

    !!! Example

        ```python
        class Model(eqx.Module):
            norm: eqx.nn.BatchNorm
            dropout: eqx.nn.Dropout
            linear: eqx.nn.Linear

            def __init__(self, key):
                key1, key2 = jax.random.split(key)
                self.norm = eqx.nn.BatchNorm(3, "batch", key=key1)
                self.dropout = eqx.nn.Dropout(0.4)
                self.linear = eqx.nn.Linear(3, 1, key=key2)

            def __call__(self, x, ctx, *, key):
                x, ctx = self.norm(x, ctx)
                x = self.dropout(x, key=key)
                x = self.linear(x)
                return x, ctx

        training_model = Model(jax.random.PRNGKey(0))
        inference_model = eqx.nn.inference_mode(training_model)
        training_model_again = eqx.nn.inference_mode(inference_model, value=False)
        ```

    This function is essentially equivalent to:
    ```python
    has_inference = lambda leaf: hasattr(leaf, "inference")

    def where(pytree):
        return tuple(x.inference
                     for x in jtu.tree_leaves(pytree, is_leaf=has_inference)
                     if has_inference(x))

    inference_pytree = equinox.tree_at(where, pytree, replace_fn=lambda _: value)
    ```

    **Arguments:**

    - `pytree`: the PyTree to modify.
    - `value`: the value to set all `inference` attributes to. Defaults to `True`, i.e.
        inference mode.

    **Returns:**

    A copy of `pytree` with all `inference` flags set to `value`.
    """
    return tree_at(_inferences, pytree, replace_fn=lambda _: value)
