import abc
from typing import Any


abstractmethod = abc.abstractmethod
_abstractattribute = object()


def abstractattribute(typ: type) -> Any:
    del typ
    return _abstractattribute


class ABCMeta(abc.ABCMeta):
    def register(cls, subclass):
        raise ValueError

    def __call__(cls, *args, **kwargs):
        self = super(ABCMeta, cls).__call__(*args, **kwargs)
        abstract_attributes = {
            name
            for name in dir(self)
            if getattr(self, name, None) is _abstractattribute
        }
        if len(abstract_attributes) > 0:
            raise NotImplementedError(
                f"Can't instantiate abstract class {cls.__name__} with"
                f" abstract attributes {abstract_attributes}"
            )
        return self
