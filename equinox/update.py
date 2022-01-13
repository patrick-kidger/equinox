import jax

from .custom_types import PyTree


def _apply_update(u, p):
    if u is None:
        return p
    else:
        return p + u


def _is_none(x):
    return x is None


def apply_updates(model: PyTree, updates: PyTree) -> PyTree:
    """A `jax.tree_map`-broadcasted version of
    ```python
    model = model if update is None else model + update
    ```

    This is often useful when updating a model's parameters via stochastic gradient
    descent. (This function is essentially the same as `optax.apply_updates`, except
    that it understands `None`.)

    **Arguments:**

    - `model`: An arbitrary PyTree.
    - `updates`: Any PyTree that is a prefix of `model`.

    **Returns:**

    The updated model.
    """
    # Assumes that updates is a prefix of model
    return jax.tree_map(_apply_update, updates, model, is_leaf=_is_none)
