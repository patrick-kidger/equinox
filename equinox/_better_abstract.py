"""
Introduces `AbstractVar` and `AbstractClassVar` type annotations, which can be used to
mark abstract instance attributes and abstract class attributes.

Provides enhanced versions of `abc.ABCMeta` and `dataclasses.dataclass` that respect
this type annotation. (In practice the sole consumer of these is `equinox.Module`, which
is the public API. But the two pieces aren't directly related under-the-hood.)

## Usage

```
from typing import ClassVar
from equinox import Module
from equinox.internal import AbstractVar, AbstractClassVar

class AbstractX(Module):
    attr1: AbstractVar[bool]
    attr2: AbstractClassVar[bool]
```

Subsequently, these can be implemented in downstream classes, e.g.:
```
class ConcreteX(AbstractX):
    attr1: bool
    attr2: ClassVar[bool] = False

ConcreteX(attr1=True)
```

An `AbstractVar[T]` can be overridden by an attribute annotated with `AbstractVar[T]`,
`AbstractClassVar[T]`, `ClassVar[T]`, `T`, or a property returning `T`.

An `AbstractClassVar[T]` can be overridden by an attribute annotated with
`AbstractClassVar[T]`, or `ClassVar[T]`.

Note that `AbstractVar` and `AbstractClassVar` do not create dataclass fields. This
affects the order of `__init__` argments. E.g.
```
class AbstractX(Module):
    attr1: AbstractVar[bool]

class ConcreteX(AbstractX):
    attr2: str
    attr1: bool
```
should be called as `ConcreteX(attr2, attr1)`.

## Known issues

Due to a Pyright bug (#4965), these must be imported as:
```
if TYPE_CHECKING:
    from typing import ClassVar as AbstractVar
    from typing import ClassVar as AbstractClassVar
else:
    from equinox.internal import AbstractVar, AbstractClassVar
```
"""
import abc
import dataclasses
from typing import ClassVar, Generic, get_args, get_origin, TYPE_CHECKING, TypeVar
from typing_extensions import dataclass_transform


_T = TypeVar("_T")


if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar, ClassVar as AbstractVar
else:

    class AbstractVar(Generic[_T]):
        pass

    # We can't just combine `ClassVar[AbstractVar[...]]`. At static checking time we
    # fake `AbstractVar` as `ClassVar` to prevent it from appearing in __init__
    # signatures. This means that static type checkers think they see
    # `ClassVar[ClassVar[...]]` which is not allowed.
    class AbstractClassVar(Generic[_T]):
        pass


def _process_annotation(annotation):
    is_abstract = False
    is_class = False
    bad_class_var_annotation = False
    if isinstance(annotation, str):
        if annotation.startswith("AbstractVar["):
            is_abstract = True
            is_class = False
        elif annotation.startswith("AbstractClassVar["):
            is_abstract = True
            is_class = True
        elif annotation.startswith("AbstractVar") or annotation.startswith(
            "AbstractClassVar"
        ):
            bad_class_var_annotation = True
    else:
        if get_origin(annotation) is AbstractVar:
            is_abstract = True
            is_class = False
        elif get_origin(annotation) is AbstractClassVar:
            is_abstract = True
            is_class = True
        elif annotation in (AbstractVar, AbstractClassVar):
            bad_class_var_annotation = True
    if bad_class_var_annotation:
        raise TypeError("Cannot use unsubscripted `AbstractVar` or `AbstractClassVar`.")
    return is_abstract, is_class


_sentinel = object()


def _is_concretisation(sub, super):
    if isinstance(sub, str) != isinstance(super, str):
        raise NotImplementedError(
            "Cannot mix stringified and non-stringified abstract " "type annotations"
        )
    if isinstance(sub, str):
        if super.startswith("AbstractVar["):
            return sub == super or sub == super[len("AbstractVar[") : -1]
        elif super.startswith("AbstractClassVar["):
            return sub == super or sub == super[len("AbstractClassVar[") : -1]
        else:
            assert False
    else:
        if get_origin(super) is AbstractVar:
            if get_origin(sub) in (AbstractVar, AbstractClassVar, ClassVar):
                return get_args(sub) == get_args(super)
            else:
                return (sub,) == get_args(super)
        elif get_origin(super) is AbstractClassVar:
            if get_origin(sub) in (AbstractClassVar, ClassVar):
                return get_args(sub) == get_args(super)
            else:
                return False
        else:
            assert False


class ABCMeta(abc.ABCMeta):
    def register(cls, subclass):
        raise ValueError

    def __new__(mcs, name, bases, namespace, /, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        abstract_vars = dict()
        abstract_class_vars = dict()
        for attr, group in [
            ("__abstractvars__", abstract_vars),
            ("__abstractclassvars__", abstract_class_vars),
        ]:
            for base in bases:
                for name, annotation in getattr(base, attr, dict()).items():
                    try:
                        existing_annotation = group[name]
                    except KeyError:
                        pass
                    else:
                        if isinstance(annotation, str) != isinstance(
                            existing_annotation, str
                        ):
                            raise NotImplementedError(
                                "Cannot mix stringified and non-stringified abstract "
                                "type annotations"
                            )
                        if not (
                            _is_concretisation(annotation, existing_annotation)
                            or _is_concretisation(existing_annotation, annotation)
                        ):
                            raise TypeError(
                                "Base classes have mismatched type annotations for "
                                f"{name}"
                            )
                    if hasattr(cls, "__annotations__"):
                        try:
                            new_annotation = cls.__annotations__[name]
                        except KeyError:
                            pass
                        else:
                            if not _is_concretisation(new_annotation, annotation):
                                raise TypeError(
                                    "Base class and derived class have mismatched type "
                                    f"annotations for {name}"
                                )
                    if name not in namespace:
                        group[name] = annotation
        if hasattr(cls, "__annotations__"):
            for name, annotation in cls.__annotations__.items():
                is_abstract, is_class = _process_annotation(annotation)
                if is_abstract:
                    if name in namespace:
                        if is_class:
                            raise TypeError(
                                "Abstract class attribute cannot have value"
                            )
                        else:
                            raise TypeError("Abstract attribute cannot have value")
                    if is_class:
                        abstract_class_vars[name] = annotation
                    else:
                        abstract_vars[name] = annotation
        cls.__abstractvars__ = abstract_vars  # pyright: ignore
        cls.__abstractclassvars__ = abstract_class_vars  # pyright: ignore
        return cls

    def __call__(cls, *args, **kwargs):
        if len(cls.__abstractclassvars__) > 0:  # pyright: ignore
            abstract_class_vars = set(cls.__abstractclassvars__)  # pyright: ignore
            raise TypeError(
                f"Can't instantiate abstract class {cls.__name__} with abstract class "
                f"attributes {abstract_class_vars}"
            )
        self = super().__call__(*args, **kwargs)
        abstract_vars = set()
        for name in cls.__abstractvars__:  # pyright: ignore
            # Deliberately not doing `if name in self.__dict__` to allow for use of
            # properties (which are actually class attributes) to override abstract
            # instance variables.
            if getattr(self, name, _sentinel) is _sentinel:
                abstract_vars.add(name)
        if len(abstract_vars) > 0:
            raise TypeError(
                f"Can't instantiate abstract class {cls.__name__} with abstract "
                f"attributes {abstract_vars}"
            )
        return self


@dataclass_transform()
def dataclass(**kwargs):
    def make_dataclass(cls):
        try:
            annotations = cls.__annotations__
        except AttributeError:
            cls = dataclasses.dataclass(**kwargs)(cls)
        else:
            new_annotations = dict(annotations)
            for name, annotation in annotations.items():
                is_abstract, _ = _process_annotation(annotation)
                if is_abstract:
                    new_annotations.pop(name)
            cls.__annotations__ = new_annotations
            cls = dataclasses.dataclass(**kwargs)(cls)
            cls.__annotations__ = annotations
        return cls

    return make_dataclass
