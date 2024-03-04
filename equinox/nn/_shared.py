from collections.abc import Callable

from jaxtyping import PyTree

from .._eval_shape import filter_eval_shape
from .._module import Module
from .._tree import tree_at, tree_equal


class SharedNode:
    """Placeholder value for nodes that have been removed by `eqx.nn.Shared`."""

    def __repr__(self):
        return "SharedNode"


class Shared(Module, strict=True):
    """Used to tie together multiple nodes across a PyTree.

    Note that Equinox modules are Py**Trees** -- so the same layer, appearing in two
    difference parts of the tree, will be treated as two copies of this layer. For
    example,
    ```python
    class SubModel(eqx.Module):
        linear: eqx.nn.Linear

    class Model(eqx.Module):
        linear: eqx.nn.Linear
        submodel: SubModel

        def __init__(self):
            linear = eqx.nn.Linear(...)
            self.linear = linear
            self.submodel = SubModel(linear)
    ```
    is used to declare `model.linear` and `model.submodel.linear` as two separate
    layers. They will start with the same initial parameter values, and then update
    independently during training.

    For when we really do want to share layers or weights across different parts of a
    model, then `eqx.nn.Shared` exists as a way to easily express this in the PyTree
    paradigm.

    !!! Example

        It is common in many language models to have an initial embedding matrix at the
        start, and then to reuse this as the weight of the final linear transformation.

        ```python
        import equinox as eqx
        import jax.numpy as jnp
        from jaxtyping import Array, Int

        class LanguageModel(eqx.Module):
            shared: eqx.nn.Shared

            def __init__(self):
                embedding = eqx.nn.Embedding(...)
                linear = eqx.nn.Linear(...)
                # These two weights will now be tied together.
                where = lambda embed_and_lin: embed_and_lin[1].weight
                get = lambda embed_and_lin: embed_and_lin[0].weight
                self.shared = eqx.nn.Shared((embedding, linear), where, get)

            def __call__(self, tokens: Int[Array, "sequence"]):
                # Expand back out so we can evaluate these layers.
                embedding, linear = self.shared()
                assert embedding.weight is linear.weight  # same parameter!
                # Now go ahead and evaluate your language model.
                values = jax.vmap(embedding)(tokens)
                ...  # other layers, probably
                return jax.vmap(linear)(values)
        ```

        _(Side note: you will sometimes see some authors referring to transposing
        the embedding matrix prior to the final linear layer. This is because some
        other libraries store the weight matrices of linear layers the other way
        around. If that had been necessary here then we could have done it with
        `get = lambda embed_and_lin: jnp.transpose(embed_and_lin[0].weight)`.)_

    """

    pytree: PyTree
    where: Callable
    get: Callable

    def __init__(self, pytree: PyTree, where: Callable, get: Callable):
        """**Arguments:**

        - `pytree`: The PyTree to share some nodes across.
        - `where`: a function specifying either a single node, or a sequence of nodes,
            as with `eqx.tree_at(where, pytree, ...)`.
        - `get`: a function, which when evaluated on `pytree`, returns either a single
            value (if `where` does), or a sequence of values (if `where` does, and in
            this case this must be a sequence of the same length as `where`).

        The node(s) of `get(pytree)` and the corresponding value(s) of `where(pytree)`
        will be tied together.

        !!! info

            To explain how this works. The implementation is just:
            ```python
            class Shared(eqx.Module):
                pytree: PyTree
                where: Callable
                get: Callable

                def __init__(self, pytree, where, get):
                    # `0` is just some dummy value
                    self.pytree = eqx.tree_at(where, pytree, replace_fn=lambda _: 0)
                    self.where = where
                    self.get = get

                def __call__(self):
                    return eqx.tree_at(self.where, self.pytree, self.get(self.pytree))
            ```
            so that at `__init__` time, the duplicate nodes specified in `where` are
            removed from the PyTree. We no longer have a separate copy updating during
            training.

            And then at `__call__` time, references to the values returned by
            `get(pytree)` are put in their place. We end up with a pytree of the same
            structure as what we started with, which we can now use (evaluate as a
            layer etc.) as normal.

        !!! tip

            If you need to apply any transform (e.g. transposing a matrix), then this
            can be done as part of `get`. For example,
            `get = lambda pair: jnp.transpose(pair[1].weight)`.
        """

        source_struct = filter_eval_shape(get, pytree)
        dest_struct = filter_eval_shape(where, pytree)
        if tree_equal(source_struct, dest_struct) is not True:
            raise ValueError(
                "Every node being shared together must have the same pytree "
                "structure, shape+dtype of arrays, etc., as each other. Got:\n"
                f"{source_struct}\n"
                "and\n"
                f"{dest_struct}"
            )
        self.pytree = tree_at(where, pytree, replace_fn=lambda _: SharedNode())
        self.where = where
        self.get = get

    def __call__(self):
        """**Arguments:**

        None.

        **Returns:**

        A PyTree of the same structure as the original `pytree`, with `get(pytree)` in
        the place of the nodes at `where(pytree)`.
        """
        return tree_at(self.where, self.pytree, self.get(self.pytree))
