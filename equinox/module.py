import abc
from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp

from equinox.custom_types import Array

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
        assert "__dataclass_setattr__" not in cls.__dict__
        cls.__dataclass_init__ = cls.__init__
        cls.__dataclass_setattr__ = cls.__setattr__
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
        # Defreeze it during __init__. TODO: this isn't thread/recursion-safe.
        field_names = {field.name for field in fields(cls)}
        cls.__setattr__ = _allow_setattr(field_names)
        cls.__init__(self, *args, **kwargs)
        cls.__setattr__ = cls.__dataclass_setattr__
        missing_names = {name for name in field_names if name not in dir(self)}
        if len(missing_names):
            raise ValueError(
                f"The following fields were not initialised during __init__: {missing_names}"
            )
        return self


class Module(metaclass=_ModuleMeta):
    def __eq__(self, other):
        return tree_equal(self, other)

    def parameters(self):
        cls = self.__class__
        mask = []
        for field in fields(cls):
            f = getattr(self, field.name)
            if isinstance(f, Module):
                mask.append(f.parameters())
            else:
                mask.append(False)
        ans = cls.__new__(cls, *mask)
        cls.__dataclass_init__(ans, *mask)
        return ans

    def remove_unjitable_fields(self):
        """remove all fields of type int, float, str, etc."""
        def filter_fn(x): return x if isinstance(x, (jnp.DeviceArray, jax.core.Tracer)) else None
        return jax.tree_map(filter_fn, self)

    def update(self, updates):
        """update non-None components"""
        return jax.tree_multimap(lambda s, u: s if u is None else u, self, updates)


class Parameter(Module):
    value: Array

    def parameters(self):
        return Parameter(True)
