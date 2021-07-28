from typing import Any


# The [...] argument is used for documentation purposes to state the size of the array.
class Array:
    def __class_getitem__(cls, item):
        return Any
