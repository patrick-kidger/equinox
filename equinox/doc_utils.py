import typing
from types import FunctionType


class WithRepr:
    def __init__(self, string):
        self.string = string

    def __repr__(self):
        return self.string


def doc_strip_annotations(fn: FunctionType) -> FunctionType:
    if getattr(typing, "GENERATING_DOCUMENTATION", False):
        fn.__annotations__ = None
    return fn
