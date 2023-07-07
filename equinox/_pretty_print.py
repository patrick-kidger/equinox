import dataclasses
import functools as ft
import types
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

import jax
import jax._src.pretty_printer as pp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PyTree


Dataclass = Any
NamedTuple = Any  # workaround typeguard bug
PrettyPrintAble = PyTree

# Re-export
text = pp.text


_comma_sep = pp.concat([pp.text(","), pp.brk()])


def bracketed(
    name: Optional[pp.Doc],
    indent: int,
    objs: Sequence[pp.Doc],
    lbracket: str,
    rbracket: str,
) -> pp.Doc:
    nested = pp.concat(
        [
            pp.nest(indent, pp.concat([pp.brk(""), pp.join(_comma_sep, objs)])),
            pp.brk(""),
        ]
    )
    concated = []
    if name is not None:
        concated.append(name)
    concated.extend([pp.text(lbracket), nested, pp.text(rbracket)])
    return pp.group(pp.concat(concated))


def named_objs(pairs, **kwargs):
    return [
        pp.concat([pp.text(key + "="), tree_pp(value, **kwargs)])
        for key, value in pairs
    ]


def _pformat_list(obj: list, **kwargs) -> pp.Doc:
    return bracketed(
        name=None,
        indent=kwargs["indent"],
        objs=[tree_pp(x, **kwargs) for x in obj],
        lbracket="[",
        rbracket="]",
    )


def _pformat_tuple(obj: tuple, **kwargs) -> pp.Doc:
    if len(obj) == 1:
        objs = [pp.concat([tree_pp(obj[0], **kwargs), pp.text(",")])]
    else:
        objs = [tree_pp(x, **kwargs) for x in obj]
    return bracketed(
        name=None, indent=kwargs["indent"], objs=objs, lbracket="(", rbracket=")"
    )


def _pformat_namedtuple(obj: NamedTuple, **kwargs) -> pp.Doc:
    objs = named_objs([(name, getattr(obj, name)) for name in obj._fields], **kwargs)
    return bracketed(
        name=pp.text(obj.__class__.__name__),
        indent=kwargs["indent"],
        objs=objs,
        lbracket="(",
        rbracket=")",
    )


def _dict_entry(key: Any, value: Any, **kwargs) -> pp.Doc:
    return pp.concat(
        [tree_pp(key, **kwargs), pp.text(":"), pp.brk(), tree_pp(value, **kwargs)]
    )


def _pformat_dict(obj: dict, **kwargs) -> pp.Doc:
    objs = [_dict_entry(key, value, **kwargs) for key, value in obj.items()]
    return bracketed(
        name=None,
        indent=kwargs["indent"],
        objs=objs,
        lbracket="{",
        rbracket="}",
    )


def pformat_short_array_text(shape: tuple[int, ...], dtype: str) -> str:
    short_dtype = (
        dtype.replace("float", "f")
        .replace("uint", "u")
        .replace("int", "i")
        .replace("complex", "c")
    )
    short_shape = ",".join(map(str, shape))
    return f"{short_dtype}[{short_shape}]"


def _pformat_short_array(
    shape: tuple[int, ...], dtype: str, kind: Optional[str]
) -> pp.Doc:
    out = pformat_short_array_text(shape, dtype)
    if kind is not None:
        out = out + f"({kind})"
    return pp.text(out)


def _pformat_array(obj: Union[jax.Array, np.ndarray], **kwargs) -> pp.Doc:
    short_arrays = kwargs["short_arrays"]
    if short_arrays:
        kind = "numpy" if isinstance(obj, np.ndarray) else None
        return _pformat_short_array(obj.shape, obj.dtype.name, kind)
    else:
        return pp.text(repr(obj))


def _pformat_function(obj: types.FunctionType, **kwargs) -> pp.Doc:
    if kwargs.get("wrapped", False):
        fn = "wrapped function"
    else:
        fn = "function"
    return pp.text(f"<{fn} {obj.__name__}>")


def _pformat_dataclass(obj, **kwargs) -> pp.Doc:
    # <uninitialised> can happen when typechecking an `eqx.Module`'s `__init__` method
    # with beartype, and printing args using pytest. We haven't yet actually assigned
    # values to the module so the repr fails.
    objs = named_objs(
        [
            (field.name, getattr(obj, field.name, "<uninitialised>"))
            for field in dataclasses.fields(obj)
            if field.repr
        ],
        **kwargs,
    )
    return bracketed(
        name=pp.text(obj.__class__.__name__),
        indent=kwargs["indent"],
        objs=objs,
        lbracket="(",
        rbracket=")",
    )


@dataclasses.dataclass
class _Partial:
    func: Callable
    args: tuple[Any, ...]
    keywords: dict[str, Any]


_Partial.__name__ = jtu.Partial.__name__
_Partial.__qualname__ = jtu.Partial.__qualname__
_Partial.__module__ = jtu.Partial.__module__


def tree_pp(obj: PrettyPrintAble, **kwargs) -> pp.Doc:
    follow_wrapped = kwargs["follow_wrapped"]
    truncate_leaf = kwargs["truncate_leaf"]
    if truncate_leaf(obj):
        return pp.text(f"{type(obj).__name__}(...)")
    if hasattr(obj, "__tree_pp__"):
        custom_pp = obj.__tree_pp__(**kwargs)
        if custom_pp is not NotImplemented:
            return pp.group(custom_pp)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
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
    elif isinstance(obj, (np.ndarray, jax.Array)):
        return _pformat_array(obj, **kwargs)
    elif isinstance(obj, (jax.custom_jvp, jax.custom_vjp)):
        return tree_pp(obj.__wrapped__, **kwargs)
    elif hasattr(obj, "__wrapped__") and follow_wrapped:
        kwargs["wrapped"] = True
        return tree_pp(obj.__wrapped__, **kwargs)  # pyright: ignore
    elif isinstance(obj, jtu.Partial) and follow_wrapped:
        obj = _Partial(obj.func, obj.args, obj.keywords)
        return _pformat_dataclass(obj, **kwargs)
    elif isinstance(obj, ft.partial) and follow_wrapped:
        kwargs["wrapped"] = True
        return tree_pp(obj.func, **kwargs)
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

    As [`equinox.tree_pprint`][], but returns the string instead of printing it.
    """

    return tree_pp(
        pytree,
        indent=indent,
        short_arrays=short_arrays,
        follow_wrapped=follow_wrapped,
        truncate_leaf=truncate_leaf,
    ).format(width=width)


def tree_pprint(
    pytree: PrettyPrintAble,
    width: int = 80,
    indent: int = 2,
    short_arrays: bool = True,
    follow_wrapped: bool = True,
    truncate_leaf: Callable[[PrettyPrintAble], bool] = _false,
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
    - `follow_wrapped`: Whether to unwrap `functools.partial` and `functools.wraps`.
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
            follow_wrapped=follow_wrapped,
            truncate_leaf=truncate_leaf,
        )
    )
