import abc
from dataclasses import dataclass, fields
import jax


# dataclasses.astuple operates recursively, which destroys information about
# nested tree_dataclasses. In contrast this is just a shallow tuplification.
def _dataclass_astuple(datacls):
    return tuple(getattr(datacls, field.name) for field in fields(datacls))


# Inherits from ABCMeta as a convenience for anyone wanting to create abstract modules,
# to avoid the metaclass conflict.
# We don't use the ABC-ness of it ourselves.
class ModuleMeta(abc.ABCMeta):
    def __new__(mcs, name, bases, dict_):
        try:
            user_provided_init = dict_['__init__']
        except KeyError:
            reinstate_init = False
        else:
            reinstate_init = True
            del dict_['__init__']
        cls = super().__new__(mcs, name, bases, dict_)
        cls = dataclass(eq=False, frozen=True)(cls)

        assert '__dataclass_setattr__' not in cls.__dict__
        cls.__dataclass_init__ = cls.__init__
        cls.__dataclass_setattr__ = cls.__setattr__
        if reinstate_init:
            cls.__init__ = user_provided_init

        def flatten(self):
            return _dataclass_astuple(self), None

        def unflatten(_, fields):
            self = cls.__new__(cls, *fields)
            cls.__dataclass_init__(self, *fields)
            return self

        jax.tree_util.register_pytree_node(cls, flatten, unflatten)
        return cls

    def __call__(cls, *args, **kwargs):
        self = cls.__new__(cls, *args, **kwargs)
        # Defreeze it during __init__
        cls.__setattr__ = object.__setattr__
        cls.__init__(self, *args, **kwargs)
        cls.__setattr__ = cls.__dataclass_setattr__
        return self


class Module(metaclass=ModuleMeta):
    # Exists purely to match the PyTorch API, for those who prefer it.
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
