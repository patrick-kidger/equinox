from dataclasses import dataclass, fields
import jax

from .tree import tree_equal


# dataclasses.astuple operates recursively, which destroys information about
# nested tree_dataclasses. In contrast this is just a shallow tuplification.
def _dataclass_astuple(cls):
    return tuple(getattr(cls, field.name) for field in fields(cls))


def _allow_setattr(fields):
    def __setattr__(self, name, value):
        if name in fields:
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(f"Cannot set attribute {name}")

    return __setattr__


class ModuleMeta(type):
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

        assert '__dataclass_init__' not in cls.__dict__
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
        # Defreeze it during __init__. TODO: this isn't thread-safe.
        field_names = [field.name for field in fields(cls)]
        cls.__setattr__ = _allow_setattr(field_names)
        cls.__init__(self, *args, **kwargs)
        cls.__setattr__ = cls.__dataclass_setattr__
        missing_names = {name for name in field_names if name not in dir(self)}
        if len(missing_names):
            raise ValueError(f"The following fields were not initialised during __init__: {missing_names}")
        return self


class Module(metaclass=ModuleMeta):
    def __eq__(self, other):
        return tree_equal(self, other)
