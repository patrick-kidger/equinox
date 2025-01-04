import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.tree_util as jtu
import wadler_lindig as wl


@dataclasses.dataclass
class _Partial:
    func: Callable
    args: tuple[Any, ...]
    keywords: dict[str, Any]


_Partial.__name__ = jtu.Partial.__name__
_Partial.__qualname__ = jtu.Partial.__qualname__
_Partial.__module__ = jtu.Partial.__module__


def _false(_):
    return False


def tree_pformat(
    pytree: Any,
    *,
    width: int = 80,
    indent: int = 2,
    short_arrays: bool = True,
    struct_as_array: bool = False,
    truncate_leaf: Callable[[Any], bool] = _false,
) -> str:
    """Pretty-formats a PyTree as a string, whilst abbreviating JAX arrays.

    (This is the function used in `__repr__` of [`equinox.Module`][].)

    As [`equinox.tree_pprint`][], but returns the string instead of printing it.
    """

    def custom(obj):
        if truncate_leaf(obj):
            return wl.TextDoc(f"{type(obj).__name__}(...)")

        if short_arrays:
            if isinstance(obj, jax.Array) or (
                struct_as_array and isinstance(obj, jax.ShapeDtypeStruct)
            ):
                dtype = obj.dtype.name
                # Added in JAX 0.4.32 to `ShapeDtypeStruct`
                if getattr(obj, "weak_type", False):
                    dtype = f"weak_{dtype}"
                return wl.array_summary(obj.shape, dtype, kind=None)

        if isinstance(obj, (jax.custom_jvp, jax.custom_vjp)):
            return wl.pdoc(obj.__wrapped__)

        if isinstance(obj, jtu.Partial):
            obj = _Partial(obj.func, obj.args, obj.keywords)
            return wl.pdoc(obj)

    return wl.pformat(
        pytree, width=width, indent=indent, short_arrays=short_arrays, custom=custom
    )


def tree_pprint(
    pytree: Any,
    *,
    width: int = 80,
    indent: int = 2,
    short_arrays: bool = True,
    struct_as_array: bool = False,
    truncate_leaf: Callable[[Any], bool] = _false,
) -> None:
    """Pretty-prints a PyTree as a string, whilst abbreviating JAX arrays.

    All JAX arrays in the PyTree are condensed down to a short string representation
    of their dtype and shape.

    !!! example

        A 32-bit floating-point JAX array of shape `(3, 4)` is printed as `f32[3,4]`.

    **Arguments:**

    - `pytree`: The PyTree to pretty-print.
    - `width`: The desired maximum number of characters per line of output. If a
        structure cannot be formatted within this constraint then a best effort will
        be made.
    - `indent`: The amount of indentation each nesting level.
    - `short_arrays`: Toggles the abbreviation of JAX arrays.
    - `struct_as_array`: Whether to treat `jax.ShapeDtypeStruct`s as arrays.
    - `truncate_leaf`: A function `Any -> bool`. Applied to all nodes in the PyTree;
        all truthy nodes will be truncated to just `f"{type(node).__name__}(...)"`.

    **Returns:**

    Nothing. (The result is printed to stdout instead.)
    """
    print(
        tree_pformat(
            pytree,
            width=width,
            indent=indent,
            short_arrays=short_arrays,
            struct_as_array=struct_as_array,
            truncate_leaf=truncate_leaf,
        )
    )
