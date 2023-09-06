"""Introduces `AbstractVar` and `AbstractClassVar` type annotations, which can be used
to mark abstract instance attributes and abstract class attributes.

Provides enhanced versions of `abc.ABCMeta` and `dataclasses.dataclass` that respect
this type annotation. (In practice the sole consumer of these is `equinox.Module`, which
is the public API. But the two pieces aren't directly related under-the-hood.)
"""
import abc
import dataclasses
from typing import (
    Annotated,
    ClassVar,
    Generic,
    get_args,
    get_origin,
    TYPE_CHECKING,
    TypeVar,
)
from typing_extensions import dataclass_transform, TypeAlias


_T = TypeVar("_T")


if TYPE_CHECKING:
    AbstractVar: TypeAlias = Annotated[_T, "AbstractVar"]
    from typing import ClassVar as AbstractClassVar
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
    if isinstance(annotation, str):
        if annotation.startswith("AbstractVar[") or annotation.startswith(
            "AbstractClassVar["
        ):
            raise NotImplementedError(
                "Stringified abstract annotations are not supported"
            )
        else:
            return False, False
    else:
        if annotation in (AbstractVar, AbstractClassVar):
            raise TypeError(
                "Cannot use unsubscripted `AbstractVar` or `AbstractClassVar`."
            )
        elif get_origin(annotation) is AbstractVar:
            if len(get_args(annotation)) != 1:
                raise TypeError("`AbstractVar` can only have a single argument.")
            is_abstract = True
            is_class = False
        elif get_origin(annotation) is AbstractClassVar:
            if len(get_args(annotation)) != 1:
                raise TypeError("`AbstractClassVar` can only have a single argument.")
            is_abstract = True
            is_class = True
        else:
            is_abstract = False
            is_class = False
        return is_abstract, is_class


_sentinel = object()


# try:
#     import beartype
# except ImportError:
#     def is_subhint(subhint, superhint) -> bool:
#         return True  # no checking in this case
# else:
#     from beartype.door import is_subhint


# TODO: reinstate once https://github.com/beartype/beartype/issues/271 is resolved.
def is_subhint(subhint, superhint) -> bool:
    return True


def _is_concretisation(sub, super):
    if isinstance(sub, str) or isinstance(super, str):
        raise NotImplementedError("Stringified abstract annotations are not supported")
    elif get_origin(super) is AbstractVar:
        if get_origin(sub) in (AbstractVar, AbstractClassVar, ClassVar):
            (sub_args,) = get_args(sub)
            (sup_args,) = get_args(super)
        else:
            sub_args = sub
            (sup_args,) = get_args(super)
    elif get_origin(super) is AbstractClassVar:
        if get_origin(sub) in (AbstractClassVar, ClassVar):
            (sub_args,) = get_args(sub)
            (sup_args,) = get_args(super)
        else:
            return False
    else:
        assert False
    return is_subhint(sub_args, sup_args)


def _is_just_static_field(field):
    # These are just the default kwargs of eqx.field apart from static
    valid_props = (
        ("default", dataclasses.MISSING),
        ("default_factory", dataclasses.MISSING),
        ("kw_only", dataclasses.MISSING),
        ("init", True),
    )
    is_valid = True
    for attr, v_allowed in valid_props:
        v = getattr(field, attr)
        if v_allowed != v:
            is_valid = False
    if not field.metadata.get("static", False):
        is_valid = False
    if not is_valid:
        raise ValueError(
            "Only eqx.field(static=True) is allowed for abstract attributes"
        )
    return True


class ABCMeta(abc.ABCMeta):
    def register(cls, subclass):
        raise ValueError

    def __new__(mcs, clsname, bases, namespace, /, **kwargs):
        # Empty dictionary of annotations that will be filled up from all baseclasses.
        abstract_vars = dict()
        abstract_class_vars = dict()
        static_abstractvars = set()
        for attr, group in [
            ("__abstractvars__", abstract_vars),
            ("__abstractclassvars__", abstract_class_vars),
        ]:
            for base in bases:
                # For this baseclass check that existing annotations are compatible with
                # with annotations found in previous baseclasses.
                for name, annotation in base.__dict__.get(attr, dict()).items():
                    try:
                        existing_annotation = group[name]
                    except KeyError:
                        # No previous annotation, so skip.
                        pass
                    else:
                        # Found previous annotation, check compatibility.
                        if not (
                            _is_concretisation(annotation, existing_annotation)
                            or _is_concretisation(existing_annotation, annotation)
                        ):
                            raise TypeError(
                                "Base classes have mismatched type annotations for "
                                f"{name}"
                            )
                    if "__annotations__" in namespace:
                        try:
                            new_annotation = namespace["__annotations__"][name]
                        except KeyError:
                            pass
                        else:
                            if not _is_concretisation(new_annotation, annotation):
                                raise TypeError(
                                    "Base class and derived class have mismatched type "
                                    f"annotations for {name}"
                                )
                            else:
                                # It is a concretization, so check wether the abstract
                                # variable was declared as static in the base class.
                                was_declared_static = False
                                try:
                                    was_declared_static = (
                                        name in base.__static_abstractvars__
                                    )
                                except AttributeError:
                                    pass
                                # If so, keep record of that name for further processing
                                if was_declared_static:
                                    static_abstractvars.add(name)
                    # FIXME: is this important?
                    # Not just `if name not in namespace`, as `namespace` may be
                    # slightly bigger from `__init_subclass__`.
                    if name not in namespace:
                        # This is true, if no default value was set yet
                        group[name] = annotation
        # Mark attributes static that have been flagged so via a static abstractvar in
        # a base class
        for name in static_abstractvars:
            # Three options
            try:
                # Check wether default value was specified
                default = namespace[name]
            except KeyError:
                # 1) No default value. So set one as static field.
                namespace[name] = dataclasses.field(metadata={"static": True})
            else:
                if isinstance(default, dataclasses.Field):
                    # 2) There was a default value specified and its a field. Make sure
                    # it is static
                    # FIXME: don't now if the ignore here is needed or one could just
                    # change the assigmnent to something that doesn't complain.
                    default.metadata["static"] = True  # pyright: ignore
                else:
                    # 3) There was a defauled value, but its not a field. So wrap field
                    # around it
                    namespace[name] = dataclasses.field(
                        default=default, metadata={"static": True}
                    )
        # Now go through all new annotations. When they are abstract, they should either
        # not have a value OR have a static field without default as a value
        if "__annotations__" in namespace:
            for name, annotation in namespace["__annotations__"].items():
                is_abstract, is_class = _process_annotation(annotation)
                if is_abstract:
                    if name in namespace:
                        default = namespace[name]
                        if isinstance(
                            default, dataclasses.Field
                        ) and _is_just_static_field(default):
                            # The default value is just a static field. So add it to the
                            # static fields
                            static_abstractvars.add(name)
                            # Remove the value from the namespace
                            namespace.pop(name)
                            print("DEBUG: Found static field", name)
                            print("DEBUG:", namespace["__annotations__"])
                        else:
                            # Abstract values with other default values are not allowed
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
        cls = super().__new__(mcs, clsname, bases, namespace, **kwargs)
        cls.__abstractvars__ = abstract_vars  # pyright: ignore
        cls.__abstractclassvars__ = abstract_class_vars  # pyright: ignore
        cls.__static_abstractvars__ = static_abstractvars  # pyright: ignore
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
