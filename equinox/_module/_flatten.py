"""Utilities for generating optimized flatten/unflatten functions for Module."""

import dataclasses
import textwrap
from typing import Any, Final, TypeVar

import jax.tree_util as jtu


MISSING = object()

_WRAPPER_FIELD_NAMES: Final = (
    "__module__",
    "__name__",
    "__qualname__",
    "__doc__",
    "__annotations__",
)

INDENT: Final = 4

FIELDS_INFO = f"""
    Dynamic fields: {{dynamic}}
    Static fields: {{static}}
    Wrapper fields: {_WRAPPER_FIELD_NAMES}
"""[1:-1]  # (trim leading and trailing newlines)


FLAT_FUNC_NAME = "<generated_flatten_{0}>"

FLAT_CODE_BASE = '''
def flatten(obj: module_cls) -> {return_annotation}:
    """Generated flatten function for {qualname}.
    
    {fields_info}
    """
    return (
        {dynamic_vals},
        {aux}
    )
'''

FLAT_KEYS_NAME = "<generated_flatten_with_keys_{0}>"

FLAT_WITH_KEYS_CODE_BASE = '''
def flatten_with_keys(obj: module_cls) -> {return_annotation}:
    """Generated flatten_with_keys function for {qualname}.
    
    {fields_info}
    """
    return (
        {key_tuple},
        {aux}
    )
'''

UNFLAT_NAME = "<generated_unflatten_{0}>"

UNFLAT_FUNC_BASE = '''
def unflatten(
    module_cls: type[T],
    aux: {aux_type},
    data: {dynamic_type},
) -> T:
    """Generated unflatten function for {qualname}.

    {fields_info}
    """
    self = object.__new__(module_cls)
    {setters}
    return self
'''

SET_DYNAMIC_BASE = """
if data[{i}] is not MISSING:
    object.__setattr__(self, {name!r}, data[{i}])
"""[1:-1]  # (trim leading and trailing newlines)

SET_AUX_BASE = """
if aux[{i}] is not MISSING:
    object.__setattr__(self, {name!r}, aux[{i}])
"""[1:-1]  # (trim leading and trailing newlines)

SET_WRAPPER_LINES = f"""
if aux[0] is not MISSING:
    waux = aux[0]
    for i, name in enumerate({_WRAPPER_FIELD_NAMES!r}):
        self.__dict__[name] = waux[i]
"""[1:-1]  # (trim leading and trailing newlines)

NS_BASE = {"object": object, "Any": Any, "tuple": tuple, "MISSING": MISSING}


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
    # aux_fs = _WRAPPER_FIELD_NAMES + static_fs

    # Build field info for docs
    fields_info = FIELDS_INFO.format(dynamic=dynamic_fs, static=static_fs)[INDENT:]

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

    # # For wrapper fields we only need to
    wrapper_exprs = ", ".join(
        [f"getattr(self, {name!r}, MISSING)" for name in _WRAPPER_FIELD_NAMES]
    )

    # For static fields, we need to store their values in aux data
    static_exprs = [f"getattr(obj, {k!r}, MISSING)" for k in static_fs]
    aux_vals = (
        "("
        f"MISSING if '__module__' not in obj.__dict__ else ({wrapper_exprs}), "
        + f"{', '.join(static_exprs)}"
        + ")"
    )

    # Build return type annotation
    dynamic_type = make_tuple_type(len(dynamic_fs))
    static_type = make_tuple_type(len(static_fs) + 1)  # +1 for wrapper aux
    clsname = cls.__qualname__

    flat_code = FLAT_CODE_BASE.format(
        return_annotation=f"tuple[{dynamic_type}, {static_type}]",
        qualname=clsname,
        fields_info=fields_info,
        dynamic_vals=dynamic_vals,
        aux=aux_vals,
    )

    # Namespace for flatten functions (they need module_cls)
    flat_ns = NS_BASE | {"jtu": jtu, "module_cls": cls}

    # make flatten func
    exec(compile(flat_code, FLAT_FUNC_NAME.format(clsname), "exec"), flat_ns)
    flat_fn = flat_ns["flatten"]
    flat_fn.__module__ = module_name
    flat_fn.__source__ = flat_code

    # -------------------------------------------
    # Generate flatten_with_keys function

    if dynamic_fs:
        key_exprs = [f"(jtu.GetAttrKey({name!r}), obj.{name})" for name in dynamic_fs]
        dynamic_key_vals = f"({', '.join(key_exprs)},)"
    else:
        dynamic_key_vals = "()"

    keys_dynamic_type = make_tuple_type(len(dynamic_fs), "tuple[jtu.GetAttrKey, str]")

    flat_w_keys_code = FLAT_WITH_KEYS_CODE_BASE.format(
        return_annotation=f"tuple[{keys_dynamic_type}, {static_type}]",
        qualname=clsname,
        fields_info=fields_info,
        key_tuple=dynamic_key_vals,
        aux=aux_vals,
    )

    # flatten with keys func
    exec(compile(flat_w_keys_code, FLAT_KEYS_NAME.format(clsname), "exec"), flat_ns)
    flat_w_keys_fn = flat_ns["flatten_with_keys"]
    flat_w_keys_fn.__module__ = module_name
    flat_w_keys_fn.__source__ = flat_w_keys_code

    # -------------------------------------------
    # Generate unflatten function - directly set fields by index
    # Extract types from flatten return type: tuple[dynamic_type, static_type]

    unflatten_lines: list[str] = []
    if dynamic_fs or static_fs:
        unflatten_lines.append("# Set dynamic fields directly by index")
    # Set dynamic fields directly by index
    unflatten_lines.extend(
        SET_DYNAMIC_BASE.format(i=i, name=k) for i, k in enumerate(dynamic_fs)
    )
    # # Set wrapper fields from aux_data
    unflatten_lines.append(SET_WRAPPER_LINES)
    # Set static fields from aux_data
    unflatten_lines.extend(
        SET_AUX_BASE.format(i=i, name=k) for i, k in enumerate(static_fs, start=1)
    )

    unflat_code = UNFLAT_FUNC_BASE.format(
        aux_type=static_type,
        dynamic_type=dynamic_type,
        qualname=clsname,
        fields_info=fields_info,
        setters=textwrap.indent("\n".join(unflatten_lines), " " * INDENT)[INDENT:],
    )

    # Namespace for unflatten function (takes module_cls as parameter)
    unflat_ns = NS_BASE | {"type": type, "T": TypeVar("T")}
    # unflatten
    exec(compile(unflat_code, UNFLAT_NAME.format(clsname), "exec"), unflat_ns)
    unflat_fn = unflat_ns["unflatten"]
    unflat_fn.__module__ = module_name
    unflat_fn.__source__ = unflat_code

    # -------------------------------------------

    return flat_fn, flat_w_keys_fn, unflat_fn
