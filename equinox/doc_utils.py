import typing
from types import FunctionType


class WithRepr:
    def __init__(self, string):
        self.string = string

    def __repr__(self):
        return self.string


if getattr(typing, "GENERATING_DOCUMENTATION", False):

    def doc_fn(fn: FunctionType) -> WithRepr:
        name = fn.__name__
        if name.startswith("_"):
            name = name[1:]
        return WithRepr(f"<function {name}>")

else:

    def doc_fn(fn: FunctionType) -> FunctionType:
        return fn


def doc_strip_annotations(fn: FunctionType) -> FunctionType:
    if getattr(typing, "GENERATING_DOCUMENTATION", False):
        fn.__annotations__ = None
    return fn
