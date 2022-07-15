import dataclasses
import functools as ft
import types
from typing import Any, Callable, Dict, List, NamedTuple, Tuple

import jax
import jax._src.pretty_printer as pp
import jax.numpy as jnp
import numpy as np

from .custom_types import Array, PyTree


Dataclass = Any
PrettyPrintAble = PyTree


_comma_sep = pp.concat([pp.text(","), pp.brk()])


def _nest(n: int, doc: pp.Doc) -> pp.Doc:
    return pp.nest(n, pp.concat([pp.brk(""), doc]))


def _pformat_list(obj: List, **kwargs) -> pp.Doc:
    indent = kwargs["indent"]
    return pp.group(
        pp.concat(
            [
                pp.text("["),
                _nest(
                    indent, pp.join(_comma_sep, [_pformat(x, **kwargs) for x in obj])
                ),
                pp.brk(""),
                pp.text("]"),
            ]
        )
    )


def _pformat_tuple(obj: Tuple, **kwargs) -> pp.Doc:
    indent = kwargs["indent"]
    if len(obj) == 1:
        entries = pp.concat([_pformat(obj[0], **kwargs), pp.text(",")])
    else:
        entries = pp.join(_comma_sep, [_pformat(x, **kwargs) for x in obj])
    return pp.group(
        pp.concat([pp.text("("), _nest(indent, entries), pp.brk(""), pp.text(")")])
    )


def _dict_entry(key: PrettyPrintAble, value: PrettyPrintAble, **kwargs) -> pp.Doc:
    return pp.concat(
        [_pformat(key, **kwargs), pp.text(":"), pp.brk(), _pformat(value, **kwargs)]
    )


def _pformat_dict(obj: Dict, **kwargs) -> pp.Doc:
    indent = kwargs["indent"]
    entries = [_dict_entry(key, value, **kwargs) for key, value in obj.items()]
    return pp.group(
        pp.concat(
            [
                pp.text("{"),
                _nest(indent, pp.join(_comma_sep, entries)),
                pp.brk(""),
                pp.text("}"),
            ]
        )
    )


def _named_entry(name: str, value: Any, **kwargs) -> pp.Doc:
    return pp.concat([pp.text(name), pp.text("="), _pformat(value, **kwargs)])


def _pformat_namedtuple(obj: NamedTuple, **kwargs) -> pp.Doc:
    indent = kwargs["indent"]
    entries = [_named_entry(name, getattr(obj, name), **kwargs) for name in obj._fields]
    return pp.group(
        pp.concat(
            [
                pp.text(obj.__class__.__name__),
                pp.text("("),
                _nest(indent, pp.join(_comma_sep, entries)),
                pp.brk(""),
                pp.text(")"),
            ]
        )
    )


def _pformat_dataclass(obj: Dataclass, **kwargs) -> pp.Doc:
    indent = kwargs["indent"]
    entries = [
        _named_entry(f.name, getattr(obj, f.name), **kwargs)
        for f in dataclasses.fields(obj)
        if f.repr
    ]
    return pp.group(
        pp.concat(
            [
                pp.text(obj.__class__.__name__),
                pp.text("("),
                _nest(indent, pp.join(_comma_sep, entries)),
                pp.brk(""),
                pp.text(")"),
            ]
        )
    )


def _pformat_array(obj: Array, **kwargs) -> pp.Doc:
    short = kwargs["short_arrays"]
    if short:
        dtype_str = (
            obj.dtype.name.replace("float", "f")
            .replace("uint", "u")
            .replace("int", "i")
            .replace("complex", "c")
        )
        shape_str = ",".join(map(str, obj.shape))
        backend = "(numpy)" if isinstance(obj, np.ndarray) else ""
        return pp.text(f"{dtype_str}[{shape_str}]{backend}")
    else:
        return pp.text(repr(obj))


def _pformat_function(obj: types.FunctionType, **kwargs) -> pp.Doc:
    if kwargs.get("wrapped", False):
        fn = "wrapped function"
    else:
        fn = "function"
    return pp.text(f"<{fn} {obj.__name__}>")


# TODO: offer user-customisable override mechanism if they wish.
# Ideally this would be something tied to being a PyTree.
def _pformat(obj: PrettyPrintAble, **kwargs) -> pp.Doc:
    follow_wrapped = kwargs["follow_wrapped"]
    truncate_leaf = kwargs["truncate_leaf"]
    if truncate_leaf(obj):
        return pp.text(f"{type(obj).__name__}(...)")
    elif dataclasses.is_dataclass(obj):
        return _pformat_dataclass(obj, **kwargs)
    elif isinstance(obj, list):
        return _pformat_list(obj, **kwargs)
    elif isinstance(obj, dict):
        return _pformat_dict(obj, **kwargs)
    elif isinstance(obj, tuple):
        if hasattr(obj, "_fields"):
            return _pformat_namedtuple(obj, **kwargs)
        else:
            return _pformat_tuple(obj, **kwargs)
    elif isinstance(obj, (np.ndarray, jnp.ndarray)):
        return _pformat_array(obj, **kwargs)
    elif isinstance(obj, (jax.custom_jvp, jax.custom_vjp)):
        return _pformat(obj.__wrapped__, **kwargs)
    elif hasattr(obj, "__wrapped__") and follow_wrapped:
        kwargs["wrapped"] = True
        return _pformat(obj.__wrapped__, **kwargs)
    elif isinstance(obj, ft.partial) and follow_wrapped:
        kwargs["wrapped"] = True
        return _pformat(obj.func, **kwargs)
    elif isinstance(obj, types.FunctionType):
        return _pformat_function(obj, **kwargs)
    else:  # int, str, float, complex, bool, etc.
        return pp.text(repr(obj))


def _false(_):
    return False


def tree_pformat(
    pytree: PrettyPrintAble,
    width: int = 80,
    indent: int = 2,
    short_arrays: bool = True,
    follow_wrapped: bool = True,
    truncate_leaf: Callable[[PrettyPrintAble], bool] = _false,
) -> str:
    """Pretty-formats a PyTree as a string, whilst abbreviating JAX arrays.

    (This is the function used in `__repr__` of [`equinox.Module`][].)

    All JAX arrays in the PyTree are condensed down to a short string representation
    of their dtype and shape.

    !!! example

        A 32-bit floating-point JAX array of shape `(3, 4)` is printed as `f32[3,4]`.

    **Arguments:**

    - `pytree`: The PyTree to pretty-format.
    - `width`: The desired maximum number of characters per line of output. If a
        structure cannot be formatted within this constraint then a best effort will
        be made.
    - `indent`: The amount of indentation each nesting level.
    - `short_arrays`: Toggles the abbreviation of JAX arrays.
    - `follow_wrapped`: Whether to unwrap `functools.partial` and `functools.wraps`.
    - `truncate_leaf`: A function `Any -> bool`. Applied to all nodes in the PyTree;
        all truthy nodes will be truncated to just `f"{type(node).__name__}(...)"`.

    **Returns:**

    A string.
    """

    return _pformat(
        pytree,
        indent=indent,
        short_arrays=short_arrays,
        follow_wrapped=follow_wrapped,
        truncate_leaf=truncate_leaf,
    ).format(width=width)
