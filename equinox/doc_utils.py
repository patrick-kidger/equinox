import typing
from types import FunctionType
from typing import Any


class _WithRepr:
    def __init__(self, string):
        self.string = string

    def __repr__(self):
        return self.string


def doc_repr(obj: Any, string: str):
    if getattr(typing, "GENERATING_DOCUMENTATION", False):
        return _WithRepr(string)
    else:
        return obj


def doc_strip_annotations(fn: FunctionType) -> FunctionType:
    if getattr(typing, "GENERATING_DOCUMENTATION", False):
        fn.__annotations__ = None
    return fn
