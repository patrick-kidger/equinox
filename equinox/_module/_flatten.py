"""Utilities for generating optimized flatten/unflatten functions for Module."""

import dataclasses
import textwrap
from enum import Enum
from typing import Any, Final, Literal, TypeVar

import jax.tree_util as jtu


# Names for generated functions
FLAT_FUNC_NAME = "<generated_flatten_{0}>"
FLAT_KEYS_NAME = "<generated_flatten_with_keys_{0}>"
UNFLAT_NAME = "<generated_unflatten_{0}>"

# Fields of Module wrappers to always include
_WRAPPER_FIELD_NAMES: Final = (
    "__module__",
    "__name__",
    "__qualname__",
    "__doc__",
    "__annotations__",
)

# Indentation for generated code
INDENT: Final = " " * 4


# Sentinel values for flattening/unflattening
class Sentinel(Enum):
    """Sentinel values for flattening/unflattening."""

    MISSING = "MISSING"


MISSING = Sentinel.MISSING

# Code template for flattening the wrapper fields
FLAT_WRAPPERS = ", ".join(
    [f"getattr(self, {name!r}, MISSING)" for name in _WRAPPER_FIELD_NAMES]
)

# Code template for flatten function
FLAT_CODE = f'''
def {{func_name}}(obj: module_cls) -> {{return_annotation}}:
    """Generated {{func_name}} function for {{qualname}}.
    
    Dynamic fields: {{dynamic_fs}}
    Static fields: {{static_fs}}
    Wrapper fields: {_WRAPPER_FIELD_NAMES}
    """
    return (
        {{dynamic_vals}},
        (
            MISSING if '__module__' not in obj.__dict__ else ({FLAT_WRAPPERS}),
            {{static_vals}}
        )
    )
'''

# Code template for setting wrapper fields during unflattening
SET_WRAPPERS: Final = "\n".join(
    f"self.__dict__[{name}] = waux[{i}]" for i, name in enumerate(_WRAPPER_FIELD_NAMES)
)

# Code template for setting dynamic fields during unflattening
SET_DYNAMIC = """
if data[{i}] is not MISSING:
    object.__setattr__(self, {name!r}, data[{i}])
"""[1:-1]  # (trim leading and trailing newlines)

# Code template for setting static fields during unflattening
SET_STATIC = """
if aux[{i}] is not MISSING:
    object.__setattr__(self, {name!r}, aux[{i}])
"""[1:-1]  # (trim leading and trailing newlines)

# Code template for unflatten function
UNFLAT_FUNC = f'''
def unflatten(
    module_cls: type[T],
    aux: {{aux_type}},
    data: {{dynamic_type}},
) -> T:
    """Generated unflatten function for {{qualname}}.

    Dynamic fields: {{dynamic_fs}}
    Static fields: {{static_fs}}
    Wrapper fields: {_WRAPPER_FIELD_NAMES}
    """
    self = object.__new__(module_cls)
    # Set fields directly by index
{{setters_dynamic}}
    if aux[0] is not MISSING:
        waux = aux[0]
{textwrap.indent(SET_WRAPPERS, INDENT * 2)}
{{setters_static}}
    return self
'''

NS_BASE = {
    "object": object,
    "Any": Any,
    "tuple": tuple,
    "Literal": Literal,
    "MISSING": MISSING,
}


def make_tuple_type(count: int, element_type: str = "Any") -> str:
    """Generate a tuple type annotation string for a given count of elements."""
    if count == 0:
        return "tuple[()]"
    elif count == 1:
        return f"tuple[{element_type}]"
    else:
        return f"tuple[{', '.join([element_type] * count)}]"


def _generate_flatten_functions(cls: type, fields: tuple[dataclasses.Field[Any], ...]):
    """Generate optimized flatten/unflatten functions for a specific field config."""
    # Separate dynamic and static fields
    _dynamic_fs, _static_fs = [], []
    for f in fields:
        if f.metadata.get("static", False):
            _static_fs.append(f.name)
        else:
            _dynamic_fs.append(f.name)
    dynamic_fs, static_fs = tuple(_dynamic_fs), tuple(_static_fs)

    # Extract the generated functions from respective namespaces to
    # set the proper module reference.
    module_name = getattr(cls, "__module__", "equinox._module._module")

    # -------------------------------------------
    # Generate flatten function

    # Directly access dynamic fields by name
    if dynamic_fs:
        dynamic_exprs = [f"getattr(obj,{name!r},MISSING)" for name in dynamic_fs]
        dynamic_vals = f"({', '.join(dynamic_exprs)},)"
    else:
        dynamic_vals = "()"

    # For static fields, we need to store their values in aux data
    static_exprs = [f"getattr(obj,{k!r},MISSING)" for k in static_fs]
    static_vals = f"{', '.join(static_exprs)}"

    # Build return type annotation
    dynamic_type = make_tuple_type(len(dynamic_fs))
    wrapper_type = "Literal[MISSING]|tuple[Any, ...]"
    static_type = f"tuple[{wrapper_type}, {', '.join(['Any'] * len(static_fs))}]"
    clsname = cls.__qualname__

    # Generate flatten function code
    flat_code = FLAT_CODE.format(
        func_name="flatten",
        return_annotation=f"tuple[{dynamic_type}, {static_type}]",
        qualname=clsname,
        dynamic_fs=dynamic_fs,
        static_fs=static_fs,
        dynamic_vals=dynamic_vals,
        static_vals=static_vals,
    )

    # make flatten func
    flat_ns = NS_BASE | {"module_cls": cls}
    exec(compile(flat_code, FLAT_FUNC_NAME.format(clsname), "exec"), flat_ns)
    flat_fn = flat_ns["flatten"]
    flat_fn.__module__ = module_name
    object.__setattr__(flat_fn, "__source__", flat_code)

    # -------------------------------------------
    # Generate flatten_with_keys function

    if dynamic_fs:
        key_exprs = [f"(jtu.GetAttrKey({name!r}), obj.{name})" for name in dynamic_fs]
        dynamic_key_vals = f"({', '.join(key_exprs)},)"
    else:
        dynamic_key_vals = "()"

    keys_dynamic_type = make_tuple_type(len(dynamic_fs), "tuple[jtu.GetAttrKey, str]")

    # Generate flatten_with_keys function code
    flat_k_code = FLAT_CODE.format(
        func_name="flatten_with_keys",
        return_annotation=f"tuple[{keys_dynamic_type}, {static_type}]",
        qualname=clsname,
        dynamic_fs=dynamic_fs,
        static_fs=static_fs,
        dynamic_vals=dynamic_key_vals,
        static_vals=static_vals,
    )

    # flatten with keys func
    flat_k_ns = NS_BASE | {"jtu": jtu, "module_cls": cls}
    exec(compile(flat_k_code, FLAT_KEYS_NAME.format(clsname), "exec"), flat_k_ns)
    flat_k_fn = flat_k_ns["flatten_with_keys"]
    flat_k_fn.__module__ = module_name
    object.__setattr__(flat_k_fn, "__source__", flat_k_code)

    # -------------------------------------------
    # Generate unflatten function - directly set fields by index
    # Extract types from flatten return type: tuple[dynamic_type, static_type]

    # Set dynamic fields by index
    unflat_dynamic = [SET_DYNAMIC.format(i=i, name=k) for i, k in enumerate(dynamic_fs)]
    # Set static fields by index. Offset by 1 for wrapper auxiliary field.
    unflat_aux = [
        SET_STATIC.format(i=i, name=k) for i, k in enumerate(static_fs, start=1)
    ]

    # Generate unflatten function code
    unflat_code = UNFLAT_FUNC.format(
        aux_type=static_type,
        dynamic_type=dynamic_type,
        qualname=clsname,
        dynamic_fs=dynamic_fs,
        static_fs=static_fs,
        setters_dynamic=textwrap.indent("\n".join(unflat_dynamic), INDENT),
        setters_static=textwrap.indent("\n".join(unflat_aux), INDENT),
    )

    # Namespace for unflatten function (takes module_cls as parameter)
    unflat_ns = NS_BASE | {"type": type, "T": TypeVar("T")}
    # unflatten
    exec(compile(unflat_code, UNFLAT_NAME.format(clsname), "exec"), unflat_ns)
    unflat_fn = unflat_ns["unflatten"]
    unflat_fn.__module__ = module_name
    object.__setattr__(unflat_fn, "__source__", unflat_code)

    # -------------------------------------------

    return flat_fn, flat_k_fn, unflat_fn
