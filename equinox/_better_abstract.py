"""Introduces `AbstractVar` and `AbstractClassVar` type annotations, which can be used
to mark abstract instance attributes and abstract class attributes.

Provides enhanced versions of `abc.ABCMeta` and `dataclasses.dataclass` that respect
this type annotation. (In practice the sole consumer of these is `equinox.Module`, which
is the public API. But the two pieces aren't directly related under-the-hood.)
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
        """Used to mark an abstract instance attribute, along with its type. Used as:
        ```python
        class Foo(eqx.Module):
            attr: AbstractVar[bool]
        ```

        An `AbstractVar[T]` must be overridden by an attribute annotated with
        `AbstractVar[T]`, `AbstractClassVar[T]`, `ClassVar[T]`, `T`, or a property
        returning `T`.

        This makes `AbstractVar` useful when you just want to assert that you can access
        `self.attr` on a subclass, regardless of whether it's an instance attribute,
        class attribute, property, etc.

        Attempting to instantiate a module with an unoveridden `AbstractVar` will raise
        an error.

        !!! Example

            ```python
            import equinox as eqx

            class AbstractX(eqx.Module):
                attr1: int
                attr2: AbstractVar[bool]

            class ConcreteX(AbstractX):
                attr2: bool

            ConcreteX(attr1=1, attr2=True)
            ```

        !!! Info

            `AbstractVar` does not create a dataclass field. This affects the order of
            `__init__` argments. E.g.
            ```python
            class AbstractX(Module):
                attr1: AbstractVar[bool]

            class ConcreteX(AbstractX):
                attr2: str
                attr1: bool
            ```
            should be called as `ConcreteX(attr2, attr1)`.

        ## Known issues

        Due to a Pyright bug
        ([#4965](https://github.com/microsoft/pyright/issues/4965)), this must be
        imported as:
        ```python
        if TYPE_CHECKING:
            from typing import ClassVar as AbstractVar
        else:
            from equinox import AbstractVar
        ```
        """

    # We can't just combine `ClassVar[AbstractVar[...]]`. At static checking time we
    # fake `AbstractVar` as `ClassVar` to prevent it from appearing in __init__
    # signatures. This means that static type checkers think they see
    # `ClassVar[ClassVar[...]]` which is not allowed.
    class AbstractClassVar(Generic[_T]):
        """Used to mark an abstract class attribute, along with its type. Used as:
        ```python
        class Foo(eqx.Module):
            attr: AbstractClassVar[bool]
        ```

        An `AbstractClassVar[T]` can be overridden by an attribute annotated with
        `AbstractClassVar[T]`, or `ClassVar[T]`. This makes `AbstractClassVar` useful
        when you want to assert that you can access `cls.attr` on a subclass.

        Attempting to instantiate a module with an unoveridden `AbstractClassVar` will
        raise an error.

        !!! Example

            ```python
            import equinox as eqx
            from typing import ClassVar

            class AbstractX(eqx.Module):
                attr1: int
                attr2: AbstractClassVar[bool]

            class ConcreteX(AbstractX):
                attr2: ClassVar[bool] = True

            ConcreteX(attr1=1)
            ```

        !!! Info

            `AbstractClassVar` does not create a dataclass field. This affects the order
            of `__init__` argments. E.g.
            ```python
            class AbstractX(Module):
                attr1: AbstractClassVar[bool]

            class ConcreteX(AbstractX):
                attr2: str
                attr1: ClassVar[bool] = True
            ```
            should be called as `ConcreteX(attr2)`.

        ## Known issues


        Due to a Pyright bug
        ([#4965](https://github.com/microsoft/pyright/issues/4965)), this must be
        imported as:
        ```python
        if TYPE_CHECKING:
            from typing import ClassVar as AbstractClassVar
        else:
            from equinox import AbstractClassVar
        ```
        """


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
                for name, annotation in base.__dict__.get(attr, dict()).items():
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
                    if "__annotations__" in cls.__dict__:
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
                    # Not just `if name not in namespace`, as `cls.__dict__` may be
                    # slightly bigger from `__init_subclass__`.
                    if name not in cls.__dict__:
                        group[name] = annotation
        if "__annotations__" in cls.__dict__:
            for name, annotation in cls.__annotations__.items():
                is_abstract, is_class = _process_annotation(annotation)
                if is_abstract:
                    if name in namespace:
                        if is_class:
                            raise TypeError(
                                f"Abstract class attribute {name} cannot have value"
                            )
                        else:
                            raise TypeError(
                                f"Abstract attribute {name} cannot have value"
                            )
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
            annotations = cls.__dict__["__annotations__"]
        except KeyError:
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
