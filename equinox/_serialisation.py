import pathlib
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, BinaryIO, Optional, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PyTree


def _ordered_tree_map(
    f: Callable[..., Any],
    tree: Any,
    *rest: Any,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> Any:
    """Like jax.tree_util.tree_map, but guaranteed to iterate over the tree
    in fixed order. (Namely depth-first left-to-right.)
    """
    # Discussion: https://github.com/patrick-kidger/equinox/issues/136
    leaves, treedef = jtu.tree_flatten(tree, is_leaf)
    all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in rest]
    return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))


def default_serialise_filter_spec(f: BinaryIO, x: Any) -> None:
    """Default filter specification for serialising a leaf.

    **Arguments**

    -   `f`: file-like object
    -   `x`: The leaf to be saved on the disk.

    **Returns**

    Nothing.

    !!! info

        This function can be extended to customise the serialisation behaviour for
        leaves.

    !!! example

        Skipping saving of jax.Array.

        ```python
        import jax.numpy as jnp
        import equinox as eqx

        tree = (jnp.array([1,2,3]), [4,5,6])
        new_filter_spec = lambda f,x: (
            None if isinstance(x, jax.Array) else eqx.default_serialise_filter_spec(f, x)
        )
        eqx.tree_serialise_leaves("some_filename.eqx", tree, filter_spec=new_filter_spec)
        ```
    """  # noqa: E501
    if isinstance(x, jax.Array):
        jnp.save(f, x)
    elif isinstance(x, np.ndarray):
        np.save(f, x)
    elif isinstance(x, (bool, float, complex, int)):
        np.save(f, x)
    else:
        pass


def default_deserialise_filter_spec(f: BinaryIO, x: Any) -> Any:
    """Default filter specification for deserialising saved data.

    **Arguments**

    -   `f`: file-like object
    -   `x`: The leaf for which the data needs to be loaded.

    **Returns**

    The new value for datatype `x`.

    !!! info

        This function can be extended to customise the deserialisation behaviour for
        leaves.

    !!! example

        Skipping loading of jax.Array.

        ```python
        import jax.numpy as jnp
        import equinox as eqx

        tree = (jnp.array([4,5,6]), [1,2,3])
        new_filter_spec = lambda f,x: (
            x if isinstance(x, jax.Array) else eqx.default_deserialise_filter_spec(f, x)
        )
        new_tree = eqx.tree_deserialise_leaves("some_filename.eqx", tree, filter_spec=new_filter_spec)
        ```
    """  # noqa: E501
    if isinstance(x, jax.Array):
        return jnp.load(f)
    elif isinstance(x, np.ndarray):
        return np.load(f)
    elif isinstance(x, (bool, float, complex, int)):
        return np.load(f).item()
    else:
        return x


def _with_suffix(path):
    path = pathlib.Path(path)
    if path.suffix == "":
        return path.with_suffix(".eqx")
    else:
        return path


@contextmanager
def _maybe_open(path_or_file: Union[str, pathlib.Path, BinaryIO], mode: str):
    """A function that unifies handling of file objects and path-like objects
    by opening the latter."""
    if isinstance(path_or_file, (str, pathlib.Path)):
        file = open(_with_suffix(path_or_file), mode)
        try:
            yield file
        finally:
            file.close()
    else:  # file-like object
        yield path_or_file


def _assert_same(new, old):
    if type(new) is not type(old):
        raise RuntimeError(
            f"Deserialised leaf has changed type from {type(old)} in `like` to "
            f"{type(new)} on disk."
        )
    if isinstance(new, (np.ndarray, jax.Array)):
        if new.shape != old.shape:
            raise RuntimeError(
                f"Deserialised leaf has changed shape from {old.shape} in `like` to "
                f"{new.shape} on disk."
            )
        if new.dtype != old.dtype:
            raise RuntimeError(
                f"Deserialised leaf has changed dtype from {old.dtype} in `like` to "
                f"{new.dtype} on disk."
            )


def tree_serialise_leaves(
    path_or_file: Union[str, pathlib.Path, BinaryIO],
    pytree: PyTree,
    filter_spec=default_serialise_filter_spec,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> None:
    """Save the leaves of a PyTree to file.

    **Arguments:**

    - `path_or_file`: The file location to save values to or a binary file-like object.
    - `pytree`: The PyTree whose leaves will be saved.
    - `filter_spec`: Specifies how to save each kind of leaf. By default all JAX
        arrays, NumPy arrays, Python bool/int/float/complexes are saved,
        and all other leaf types are ignored. (See
        [`equinox.default_serialise_filter_spec`][].)
    - `is_leaf`: Called on every node of `pytree`; if `True` then this node will be
        treated as a leaf.

    **Returns:**

    Nothing.

    !!! example

        This can be used to save a model to file.

        ```python
        import equinox as eqx
        import jax.random as jr

        model = eqx.nn.MLP(2, 2, 2, 2, key=jr.PRNGKey(0))
        eqx.tree_serialise_leaves("some_filename.eqx", model)
        ```

    !!! info

        `filter_spec` should typically be a function `(File, Any) -> None`, which takes
        a file handle and a leaf to save, and either saves the leaf to the file or
        does nothing.

        It can also be a PyTree of such functions, in which case the PyTree structure
        should be a prefix of `pytree`, and each function will be mapped over the
        corresponding sub-PyTree of `pytree`.
    """

    with _maybe_open(path_or_file, "wb") as f:

        def _serialise(spec, x):
            def __serialise(y):
                spec(f, y)

            _ordered_tree_map(__serialise, x, is_leaf=is_leaf)

        _ordered_tree_map(_serialise, filter_spec, pytree)


def tree_deserialise_leaves(
    path_or_file: Union[str, pathlib.Path, BinaryIO],
    like: PyTree,
    filter_spec=default_deserialise_filter_spec,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> PyTree:
    """Load the leaves of a PyTree from a file.

    **Arguments:**

    - `path_or_file`: The file location to load values from or a binary file-like
        object.
    - `like`: A PyTree of same structure, and with leaves of the same type, as the
        PyTree being loaded. Those leaves which are loaded will replace the
        corresponding leaves of `like`.
    - `filter_spec`: Specifies how to load each kind of leaf. By default all JAX
        arrays, NumPy arrays, Python bool/int/float/complexes are loaded, and
        all other leaf types are not loaded, and will retain their
        value from `like`. (See [`equinox.default_deserialise_filter_spec`][].)
    - `is_leaf`: Called on every node of `like`; if `True` then this node will be
        treated as a leaf.

    **Returns:**

    The loaded PyTree, formed by iterating over `like` and replacing some of its leaves
    with the leaves saved in `path`.

    !!! example

        This can be used to load a model from file.

        ```python
        import equinox as eqx
        import jax.random as jr

        model_original = eqx.nn.MLP(2, 2, 2, 2, key=jr.PRNGKey(0))
        eqx.tree_serialise_leaves("some_filename.eqx", model_original)
        model_loaded = eqx.tree_deserialise_leaves("some_filename.eqx", model_original)

        # To partially load weights: in this case load everything except the final layer.
        model_partial = eqx.tree_at(lambda mlp: mlp.layers[-1], model_loaded, model_original)
        ```
    !!! info

        `filter_spec` should typically be a function `(File, Any) -> Any`, which takes
        a file handle and a leaf from `like`, and either returns the corresponding
        loaded leaf, or retuns the leaf from `like` unchanged.

        It can also be a PyTree of such functions, in which case the PyTree structure
        should be a prefix of `pytree`, and each function will be mapped over the
        corresponding sub-PyTree of `pytree`.
    """  # noqa: E501
    with _maybe_open(path_or_file, "rb") as f:

        def _deserialise(spec, x):
            def __deserialise(y):
                return spec(f, y)

            return _ordered_tree_map(__deserialise, x, is_leaf=is_leaf)

        out = _ordered_tree_map(_deserialise, filter_spec, like)
    jtu.tree_map(_assert_same, out, like, is_leaf=is_leaf)
    return out
