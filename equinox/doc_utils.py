import inspect
import typing
from typing import TYPE_CHECKING, TypeVar


# Inherits from type so that _WithRepr instances are types and can be used as
# e.g. Sequence[_WithRepr(...)]
class _WithRepr(type):
    def __new__(self, string):
        out = super().__new__(self, string, (), {})
        # prevent the custom typing repr from doing the wrong thing
        out.__module__ = "builtins"
        return out

    def __init__(self, string):
        self.string = string

    def __repr__(self):
        return self.string


_T = TypeVar("_T")


def doc_repr(obj: _T, string: str) -> _T:
    if TYPE_CHECKING:
        return obj
    else:
        if getattr(typing, "GENERATING_DOCUMENTATION", False):
            return _WithRepr(string)
        else:
            return obj


def doc_remove_args(*args):
    def doc_remove_args_impl(fn):
        sig = inspect.signature(fn)
        new_params = []
        for param in sig.parameters.values():
            if param.name not in args:
                new_params.append(param)
        sig = sig.replace(parameters=new_params)
        fn.__signature__ = sig
        return fn

    return doc_remove_args_impl
