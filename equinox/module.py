import abc
import functools as ft
from dataclasses import dataclass, fields, field

import jax

from .tree import tree_equal


def static_field(**kwargs):
    try:
        metadata = dict(kwargs['metadata'])
    except KeyError:
        metadata = kwargs['metadata'] = {}
    if 'static' in metadata:
        raise ValueError("Cannot use metadata with `static` already set.")
    metadata['static'] = True
    return field(**kwargs)


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

    return _InitableModule


# Inherits from abc.ABCMeta as a convenience for a common use-case.
# It's not a feature we use ourselve.
class _ModuleMeta(abc.ABCMeta):
    def __new__(mcs, name, bases, dict_):
        cls = super().__new__(mcs, name, bases, dict_)
        cls = dataclass(eq=False, frozen=True)(cls)
        jax.tree_util.register_pytree_node_class(cls)
        return cls

    def __call__(cls, *args, **kwargs):
        self = cls.__new__(cls, *args, **kwargs)

        # Defreeze it during __init__
        initable_cls = _make_initable(cls)
        object.__setattr__(self, "__class__", initable_cls)
        cls.__init__(self, *args, **kwargs)
        object.__setattr__(self, "__class__", cls)

        missing_names = {
            field.name
            for field in fields(cls)
            if field.init and field.name not in dir(self)
        }
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
        dynamic_field_names = []
        dynamic_field_values = []
        static_field_names = []
        static_field_values = []
        for field in fields(self):
            name = field.name
            try:
                value = self.__dict__[name]
            except KeyError:
                continue
            if field.metadata.get('static', False):
                static_field_names.append(name)
                static_field_values.append(value)
            else:
                dynamic_field_names.append(name)
                dynamic_field_values.append(value)
        return tuple(dynamic_field_values), (tuple(dynamic_field_names), tuple(static_field_names), tuple(static_field_values))

    @classmethod
    def tree_unflatten(cls, aux, dynamic_field_values):
        self = cls.__new__(cls)
        dynamic_field_names, static_field_names, static_field_values = aux
        for name, value in zip(dynamic_field_names, dynamic_field_values):
            object.__setattr__(self, name, value)
        for name, value in zip(static_field_names, static_field_values):
            object.__setattr__(self, name, value)
        return self
