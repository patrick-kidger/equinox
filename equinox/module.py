import abc
import functools as ft
import inspect
from dataclasses import dataclass, field, fields

import jax

from .tree import tree_equal


def static_field(**kwargs):
    try:
        metadata = dict(kwargs["metadata"])
    except KeyError:
        metadata = kwargs["metadata"] = {}
    if "static" in metadata:
        raise ValueError("Cannot use metadata with `static` already set.")
    metadata["static"] = True
    return field(**kwargs)


class _wrap_method:
    def __init__(self, method):
        self.method = method
        if getattr(self.method, "__isabstractmethod__", False):
            self.__isabstractmethod__ = self.method.__isabstractmethod__

    def __get__(self, instance, owner):
        if instance is None:
            return self.method
        return jax.tree_util.Partial(self.method, instance)


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


def _has_dataclass_init(cls):
    if "__init__" in cls.__dict__:
        return False
    return cls._has_dataclass_init


def _not_magic(k):
    return not (k.startswith("__") and k.endswith("__"))


# Inherits from abc.ABCMeta as a convenience for a common use-case.
# It's not a feature we use ourselve.
class _ModuleMeta(abc.ABCMeta):
    def __new__(mcs, name, bases, dict_):
        dict_ = {
            k: _wrap_method(v) if _not_magic(k) and inspect.isfunction(v) else v
            for k, v in dict_.items()
        }
        cls = super().__new__(mcs, name, bases, dict_)
        # Do override subclasses' dataclass-__init__-s. (None of which call super, so
        # they must be overriden.)
        # Don't override custom __init__'s, which leads to poor ergonomics:
        # e.g. if `B` has a custom init then `class A(B): pass` would otherwise set a
        # dataclass init that overrides the custom __init__.
        _init = cls._has_dataclass_init = _has_dataclass_init(cls)
        cls = dataclass(eq=False, frozen=True, init=_init)(cls)
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
    _has_dataclass_init = True

    def __hash__(self):
        return hash(tuple(jax.tree_leaves(self)))

    def __eq__(self, other):
        return tree_equal(self, other)

    def tree_flatten(self):
        dynamic_field_names = []
        dynamic_field_values = []
        static_field_names = []
        static_field_values = []
        for field_ in fields(self):
            name = field_.name
            try:
                value = self.__dict__[name]
            except KeyError:
                continue
            if field_.metadata.get("static", False):
                static_field_names.append(name)
                static_field_values.append(value)
            else:
                dynamic_field_names.append(name)
                dynamic_field_values.append(value)
        return tuple(dynamic_field_values), (
            tuple(dynamic_field_names),
            tuple(static_field_names),
            tuple(static_field_values),
        )

    @classmethod
    def tree_unflatten(cls, aux, dynamic_field_values):
        self = cls.__new__(cls)
        dynamic_field_names, static_field_names, static_field_values = aux
        for name, value in zip(dynamic_field_names, dynamic_field_values):
            object.__setattr__(self, name, value)
        for name, value in zip(static_field_names, static_field_values):
            object.__setattr__(self, name, value)
        return self
