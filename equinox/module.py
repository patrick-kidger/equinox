import abc
import functools as ft
from dataclasses import dataclass, fields

import jax

from .tree import tree_equal


# dataclasses.astuple operates recursively, which destroys information about
# nested tree_dataclasses. In contrast this is just a shallow tuplification.
def _dataclass_astuple(cls):
    return tuple(getattr(cls, field.name) for field in fields(cls) if field.init)


@ft.lru_cache(maxsize=128)
def _make_initable(cls):
    field_names = {field.name for field in fields(cls)}

    class _InitableModule(cls):
        pass

    # Done like this to avoid dataclasses complaining about overriding setattr on a
    # frozen class.
    def __setattr__(self, name, value):
        if name in field_names:
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(f"Cannot set attribute {name}")

    _InitableModule.__setattr__ = __setattr__

    return _InitableModule, field_names


# Inherits from abc.ABCMeta as a convenience for a common use-case.
# It's not a feature we use ourselve.
class _ModuleMeta(abc.ABCMeta):
    def __new__(mcs, name, bases, dict_):
        try:
            user_provided_init = dict_["__init__"]
        except KeyError:
            reinstate_init = False
        else:
            reinstate_init = True
            del dict_["__init__"]

        cls = super().__new__(mcs, name, bases, dict_)
        cls = dataclass(eq=False, frozen=True)(cls)

        assert "__dataclass_init__" not in cls.__dict__
        cls.__dataclass_init__ = cls.__init__
        if not reinstate_init:
            # Override the default dataclass init if our parent has an init
            for kls in cls.__mro__[1:-1]:
                if (
                    isinstance(kls, _ModuleMeta)
                    and kls.__init__ is not kls.__dataclass_init__
                ):
                    reinstate_init = True
                    user_provided_init = kls.__init__
                    break
        if reinstate_init:
            cls.__init__ = user_provided_init

        jax.tree_util.register_pytree_node_class(cls)
        return cls

    def __call__(cls, *args, **kwargs):
        self = cls.__new__(cls, *args, **kwargs)

        # Defreeze it during __init__
        initable_cls, field_names = _make_initable(cls)
        object.__setattr__(self, "__class__", initable_cls)
        cls.__init__(self, *args, **kwargs)
        object.__setattr__(self, "__class__", cls)

        missing_names = {name for name in field_names if name not in dir(self)}
        if len(missing_names):
            raise ValueError(
                f"The following fields were not initialised during __init__: {missing_names}"
            )
        return self


class Module(metaclass=_ModuleMeta):
    def __hash__(self):
        return hash(tuple(jax.tree_leaves(self)))

    def __eq__(self, other):
        return tree_equal(self, other)

    def tree_flatten(self):
        return _dataclass_astuple(self), None

    @classmethod
    def tree_unflatten(cls, _, fields):
        self = cls.__new__(cls, *fields)
        cls.__dataclass_init__(self, *fields)
        return self
