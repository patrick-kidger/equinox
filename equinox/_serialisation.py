import functools as ft
import pathlib
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, BinaryIO, Optional, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PyTree

from ._filters import is_array_like


class TreePathError(RuntimeError):
    path: tuple


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
    paths_and_leaves, treedef = jtu.tree_flatten_with_path(tree, is_leaf)
    all_leaves = list(zip(*paths_and_leaves)) + [treedef.flatten_up_to(r) for r in rest]

    @ft.wraps(f)
    def _f(path, *xs):
        try:
            return f(*xs)
        except TreePathError as e:
            combo_path = path + e.path
            exc = TreePathError(f"Error at leaf with path {combo_path}")
            exc.path = combo_path
            raise exc from e
        except Exception as e:
            exc = TreePathError(f"Error at leaf with path {path}")
            exc.path = path
            raise exc from e

    return treedef.unflatten(_f(*xs) for xs in zip(*all_leaves))


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
        # Important to use `np` here so that we don't cast NumPy arrays to JAX arrays.
        np.save(f, x)
    elif is_array_like(x):
        # Important to use `jnp` here to handle `bfloat16`.
        jnp.save(f, x)
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
    if isinstance(x, (jax.Array, jax.ShapeDtypeStruct)):
        return jnp.load(f)
    elif isinstance(x, np.ndarray):
        # Important to use `np` here to avoid promoting NumPy arrays to JAX.
        return np.load(f)
    elif is_array_like(x):
        # np.generic gets deserialised directly as an array, so convert back to a scalar
        # type here.
        # See also https://github.com/google/jax/issues/17858
        out = np.load(f)
        if isinstance(x, jax.dtypes.bfloat16):
            out = out.view(jax.dtypes.bfloat16)
        return type(x)(out.item())
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


def _assert_same(array_impl_type):
    def _assert_same_impl(path, new, old):
        typenew = type(new)
        typeold = type(old)
        if typeold is jax.ShapeDtypeStruct:
            typeold = array_impl_type
        if typenew is not typeold:
            raise RuntimeError(
                f"Deserialised leaf at path '{jtu.keystr(path)}' has changed type from "
                f"{type(old)} in `like` to {type(new)} on disk."
            )
        if isinstance(new, (np.ndarray, jax.Array)):
            if new.shape != old.shape:
                raise RuntimeError(
                    f"Deserialised leaf at path {path} has changed shape from "
                    f"{old.shape} in `like` to {new.shape} on disk."
                )
            if new.dtype != old.dtype:
                raise RuntimeError(
                    f"Deserialised leaf at path {path} has changed dtype from "
                    f"{old.dtype} in `like` to {new.dtype} on disk."
                )

    return _assert_same_impl


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

        # To partially load weights, do model surgery. In this case load everything
        # except the final layer.
        model_partial = eqx.tree_at(lambda mlp: mlp.layers[-1], model_loaded, model_original)
        ```

    !!! example

        A common pattern is the following:

        ```python
        def run(..., load_path=None):
            if load_path is None:
                model = Model(...hyperparameters...)
            else:
                model = eqx.filter_eval_shape(Model, ...hyperparameters...)
                model = eqx.tree_deserialise_leaves(load_path, model)
        ```
        in which either a model is created directly (e.g. at the start of training), or
        a suitable `like` is constructed (e.g. when resuming training), where
        [`equinox.filter_eval_shape`][] is used to avoid creating spurious short-lived
        arrays taking up memory.

    !!! info

        `filter_spec` should typically be a function `(File, Any) -> Any`, which takes
        a file handle and a leaf from `like`, and either returns the corresponding
        loaded leaf, or returns the leaf from `like` unchanged.

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
    with jax.ensure_compile_time_eval():
        # ArrayImpl isn't a public type, so this is how we get access to it instead.
        # `ensure_compile_time_eval` just in case someone is doing deserialisation
        # inside JIT. Which would be weird, but still.
        array_impl_type = type(jnp.array(0))
    jtu.tree_map_with_path(_assert_same(array_impl_type), out, like, is_leaf=is_leaf)
    return out
