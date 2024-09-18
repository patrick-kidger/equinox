"""Implements the core Module abstraction.

This includes both the core dataclass+pytree combo, plus various related pieces of
functionality.
"""

import abc
import dataclasses
import functools as ft
import inspect
import types
import warnings
import weakref
from collections.abc import Callable
from typing import Any, cast, Optional, Protocol, TYPE_CHECKING, TypeVar, Union
from typing_extensions import dataclass_transform, ParamSpec

import jax
import jax._src.traceback_util as traceback_util
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, Bool, PyTreeDef

from ._better_abstract import ABCMeta, dataclass
from ._caches import cache_clears
from ._doc_utils import doc_repr
from ._filters import is_array, is_array_like
from ._pretty_print import tree_pformat
from ._tree import tree_equal


traceback_util.register_exclusion(__file__)


_P = ParamSpec("_P")
_T = TypeVar("_T")


#
# Part 1: fields. We extend `dataclasses.field` with a wrapper supporting extra features
# like type converters, and marking a field as static metadata.
#


def static_field(**kwargs):
    """Deprecated in favour of [`equinox.field`][], i.e. `eqx.field(static=True)`."""
    return field(**kwargs, static=True)


_converter_sentinel: Any = doc_repr(object(), "lambda x: x")


def field(
    *,
    converter: Callable[[Any], Any] = _converter_sentinel,
    static: bool = False,
    **kwargs,
):
    """Equinox supports extra functionality on top of the default dataclasses.

    **Arguments:**

    - `converter`: a function to call on this field when the model is initialised. For
        example, `field(converter=jax.numpy.asarray)` to convert
        `bool`/`int`/`float`/`complex` values to JAX arrays. This is ran after the
        `__init__` method (i.e. when using a user-provided `__init__`), and before
        `__post_init__` (i.e. when using the default dataclass initialisation).
    - `static`: whether the field should not interact with any JAX transform at all (by
        making it part of the PyTree structure rather than a leaf).
    - `**kwargs`: All other keyword arguments are passed on to `dataclass.field`.

    !!! example "Example for `converter`"

        ```python
        class MyModule(eqx.Module):
            foo: Array = eqx.field(converter=jax.numpy.asarray)

        mymodule = MyModule(1.0)
        assert isinstance(mymodule.foo, jax.Array)
        ```

    !!! example "Example for `static`"

        ```python
        class MyModule(eqx.Module):
            normal_field: int
            static_field: int = eqx.field(static=True)

        mymodule = MyModule("normal", "static")
        leaves, treedef = jax.tree_util.tree_flatten(mymodule)
        assert leaves == ["normal"]
        assert "static" in str(treedef)
        ```

    `static=True` means that this field is not a node of the PyTree, so it does not
    interact with any JAX transforms, like JIT or grad. This means that it is usually a
    bug to make JAX arrays be static fields. `static=True` should very rarely be used.
    It is preferred to just filter out each field with `eqx.partition` whenever you need
    to select only some fields.
    """
    try:
        metadata = dict(kwargs.pop("metadata"))  # safety copy
    except KeyError:
        metadata = {}
    if "converter" in metadata:
        raise ValueError("Cannot use metadata with `static` already set.")
    if "static" in metadata:
        raise ValueError("Cannot use metadata with `static` already set.")
    # We don't just use `lambda x: x` as the default, so that this works:
    # ```
    # class Abstract(eqx.Module):
    #     x: int = eqx.field()
    #
    # class Concrete(Abstract):
    #    @property
    #    def x(self):
    #        pass
    # ```
    # otherwise we try to call the default converter on a property without a setter,
    # and an error is raised.
    # Oddities like the above are to be discouraged, of course, but in particular
    # `field(init=False)` was sometimes used to denote an abstract field (prior to the
    # introduction of `AbstractVar`), so we do want to support this.
    if converter is not _converter_sentinel:
        metadata["converter"] = converter
    if static:
        metadata["static"] = True
    return dataclasses.field(metadata=metadata, **kwargs)


#
# Part 2: Modules!
# This is the core of Equinox.
#


@dataclass(frozen=True)
class StrictConfig:
    """Used to configure strict Equinox modules.

    Passing this as the `strict` parameter of a `Module` marks that the `Module` is
    strict, and additionally provides some extra custom behaviour.

    Example usage is:
    ```python
    class Foo(eqx.Module, strict=StrictConfig(force_abstract=True)):
        pass
    ```
    """

    force_abstract: bool = False
    allow_abstract_name: bool = False
    allow_method_override: bool = False


StrictConfig.__init__.__doc__ = """**Arguments:**

- `force_abstract`: marks that a class without abstract methods or attributes should
    still be treated as abstract. Useful for e.g.
    ```python
    class AbstractFoo(eqx.Module, strict=True):
        pass
    ```
    which would otherwise be treated as concrete.
- `allow_abstract_name`: grants this class an exemption from the rule that abstract
    classes must begin their names with 'Abstract' or '_Abstract', and that concrete
    classes must not begin their names with 'Abstract' or '_Abstract'.
- `allow_method_override`: grants this class an exemption from the rule that it cannot
    override concrete methods.

Both `allow_*` options should be used with care! They exist only to make it easier to
transition a codebase from non-strict to strict `Module`s in a backward-compatible
manner. If you are starting a new codebase you should not have need of them.
"""


# Inherits from ABCMeta to support `eqx.{AbstractVar, AbstractClassVar}` and
# `abc.abstractmethod`.
class _ActualModuleMeta(ABCMeta):
    # This method is called whenever you definite a module: `class Foo(eqx.Module): ...`
    def __new__(
        mcs,
        name,
        bases,
        dict_,
        /,
        strict: Union[bool, StrictConfig] = False,
        **kwargs,
    ):
        if isinstance(strict, bool):
            strict_config = StrictConfig()
        elif isinstance(strict, StrictConfig):
            strict_config = strict
            strict = True
        else:
            raise TypeError(
                "The `strict` parameter in `class Foo(equinox.Module, strict=...)` "
                "must either be a `bool` or `equinox.StrictConfig` object."
            )

        # [Step 1] Create the class as normal.
        cls = super().__new__(mcs, name, bases, dict_, **kwargs)
        # [Step 2] Arrange for bound methods to be treated as PyTrees as well. This
        # ensures that
        # ```
        # @jax.jit
        # def run(fn):
        #     ...
        # run(SomeModule().some_method)
        # ```
        # works.
        for k, v in cls.__dict__.items():
            if _not_magic(k) and inspect.isfunction(v):
                setattr(cls, k, _wrap_method(v))

        # [Step 3] Handle initialisation.
        #
        # For context, with any Python dataclass, there are three possible scenarios for
        # initialisation:
        # (a) a user-provided `__init__` method is supplied;
        # (b) dataclasses creates `__init__` , without a user-provided `__post_init__`
        # (c) dataclasses creates `__init__` , with a user-provided `__post_init__`

        # Create a dataclass `__init__` method if a user method isn't provided.
        # If a user passed one on this class, then we definitely have a custom __init__.
        # Else just use whatever our superclass does. Note that this is different to
        # default dataclass behaviour. Given
        # ```
        # @dataclass
        # class Foo: def __init__(...): ...
        # @dataclass
        # class Bar(Foo): pass
        # ```
        # then `Bar` will end up with a dataclass-provided `__init__`. That ends up
        # being ergonomically very annoying, so we disable it.
        added_custom_init = "__init__" in cls.__dict__
        if added_custom_init:
            has_dataclass_init = False
        else:
            for kls in cls.__mro__[1:-1]:
                try:
                    has_dataclass_init = _has_dataclass_init[kls]
                except KeyError:
                    # Non-Module superclasses.
                    if kls.__init__ is not object.__init__:
                        has_dataclass_init = False
                        break
                else:
                    break
            else:
                assert name == "Module"
                has_dataclass_init = True  # eqx.Module itself

        # Check for a common error. (Check for `_Initable` to avoid duplicate warnings.)
        if (
            not has_dataclass_init
            and hasattr(cls, "__post_init__")
            and not issubclass(cls, _Initable)
        ):
            warnings.warn(
                f"Class `{cls.__module__}.{cls.__qualname__}` has both an "
                "`__init__` method and a `__post_init__` method. This means that "
                "the `__post_init__` method will not be run!\n"
                "The reason for this is that `__post_init__` is intended to be "
                "used with the automatically-generated `__init__` method provided "
                "by Python dataclasses, which are generated of the form:\n"
                "```\n"
                "def __init__(self, field1, field2)\n"
                "    self.field1 = field1\n"
                "    self.field2 = field2\n"
                "    self.__post_init__()\n"
                "```\n"
                "and as such a user-provided `__init__` overrides both the setting "
                "of fields, and the calling of `__post_init__`.\n"
                "The above is how Python dataclasses work, and has nothing "
                "to do with Equinox!\n"
                "If you are using `__post_init__` to check that certain invariants "
                "hold, then consider using `__check_init__` instead. This is an "
                "Equinox-specific extension that is always ran. See here for more "
                "details: "
                "https://docs.kidger.site/equinox/api/module/advanced_fields/#checking-invariants",  # noqa: E501
                stacklevel=2,
            )

        # Add support for `eqx.field(converter=...)` when using `__post_init__`.
        # (Scenario (c) above. Scenarios (a) and (b) are handled later.)
        if has_dataclass_init and "__post_init__" in cls.__dict__:
            post_init = cls.__post_init__

            @ft.wraps(post_init)  # pyright: ignore
            def __post_init__(self, *args, **kwargs):
                # This `if` is to handle `super()` correctly.
                # We want to only convert once, at the top level.
                #
                # This check is basically testing whether or not the function we're in
                # now (`cls`) is at the top level (`self.__class__`). If we are, do
                # conversion. If we're not, it's presumably because someone is calling
                # us via `super()` in the middle of their own `__post_init__`. No
                # conversion then; their own version of this wrapper will do it at the
                # appropriate time instead.
                #
                # This top-level business means that this is very nearly the same as
                # doing conversion in `_ModuleMeta.__call__`. The differences are that
                # (a) that wouldn't allow us to convert fields before the user-provided
                # `__post_init__`, and (b) it allows other libraries (i.e. jaxtyping)
                # to later monkey-patch `__init__`, and we have our converter run before
                # their own monkey-patched-in code.
                if self.__class__ is _make_initable_wrapper(cls):
                    # Convert all fields currently available.
                    _convert_fields(self, init=True)
                post_init(self, *args, **kwargs)  # pyright: ignore
                if self.__class__ is _make_initable_wrapper(cls):
                    # Convert all the fields filled in by `__post_init__` as well.
                    _convert_fields(self, init=False)

            cls.__post_init__ = __post_init__  # pyright: ignore
        else:
            post_init = None

        # Fairly common to write `Superclass.__init__.__doc__ = "..."` with
        # dataclass-provided inits; here we look through the class hierarchy and will
        # copy this doc forward.
        if has_dataclass_init:
            init_doc = cls.__init__.__doc__

        # [Step 4] Register as a dataclass.
        cls = dataclass(eq=False, repr=False, frozen=True, init=has_dataclass_init)(
            cls  # pyright: ignore
        )
        # [Step 3b] -- finish off building `__init__` methods. Until we'd done
        # dataclass'ification then we didn't necessarily have our `__init__` method.

        # Set annotation to the converter input. This is useful for runtime type
        # checkers.
        # Note that mutating the `__init__.__annotations__` is okay, as it was created
        # by the dataclass decorator on the previous line, so nothing else owns it.
        for f in dataclasses.fields(cls):
            if f.name not in cls.__init__.__annotations__:
                continue  # Odd behaviour, so skip.
            try:
                converter = f.metadata["converter"]
            except KeyError:
                pass
            else:
                try:
                    signature = inspect.signature(converter)
                except ValueError:
                    # e.g. `inspect.signature(str)` fails
                    converter_annotation = Any
                else:
                    parameters = list(signature.parameters.values())
                    if len(parameters) == 0:
                        # No idea what happened, but play it safe.
                        converter_annotation = Any
                    else:
                        converter_annotation = parameters[0].annotation
                        if converter_annotation is inspect.Signature.empty:
                            converter_annotation = Any
                cls.__init__.__annotations__[f.name] = converter_annotation

        # Registering here records that the `dataclass(...)` call has happened.
        _has_dataclass_init[cls] = has_dataclass_init

        # Now handle conversion for cases (a) and (b) above, in which there is no
        # `__post_init__`.
        if post_init is None:
            init = cls.__init__

            @ft.wraps(init)
            def __init__(self, *args, **kwargs):
                __tracebackhide__ = True
                init(self, *args, **kwargs)
                # Same `if` trick as with `__post_init__`.
                if self.__class__ is _make_initable_wrapper(cls):
                    _convert_fields(self, init=True)
                    _convert_fields(self, init=False)

            cls.__init__ = __init__

        # Assign `__doc__` in case it has been manually overridden:
        # ```
        # class Foo(eqx.Module):
        #     x: int
        #
        # Foo.__init__.__doc__ = "Foo should be called with with an integer `x`."
        #
        # class Bar(Foo):
        #     pass
        #
        # # Now we try to access `Bar.__init__.__doc__`. (E.g. during docgen.)
        # ```
        if has_dataclass_init:
            cls.__init__.__doc__ = init_doc  # pyright: ignore
            # TODO: is this next line still necessary?
            cls.__init__.__module__ = cls.__module__

        # [Step 5] We support an optional `strict` mode for Rust-like strictness in the
        # type checking.
        # In practice this is probably too much for your average user, but it's a great
        # way to build robust libraries.
        _is_force_abstract[cls] = strict_config.force_abstract
        _is_strict[cls] = strict
        if strict:
            for base in bases:
                if base is Module:
                    # We shouldn't check `Module` itself: we can't make it strict as to
                    # be able to inherit from it we would need to make it abstract. To
                    # be able to make it abstract its name must start with `Abstract`.
                    # That we cannot do for backward compatibility.
                    continue
                if _is_special_form(base):
                    # Skip `typing.Generic` etc.
                    continue
                # Invariant: all base classes are also strict modules.
                if not issubclass(base, Module):
                    raise TypeError(
                        "Strict `eqx.Module`s must only inherit from other strict "
                        f"`eqx.Module`s. `{cls.__module__}.{cls.__qualname__}` is a "
                        "strict Module inheriting from "
                        f"`{base.__module__}.{base.__qualname__}`, which is not a "
                        "`Module` at all."
                    )
                if not _is_strict[base]:
                    raise TypeError(
                        "Strict `eqx.Module`s must only inherit from other strict "
                        f"`eqx.Module`s. `{cls.__module__}.{cls.__qualname__}` is a "
                        "strict Module inheriting from "
                        f"`{base.__module__}.{base.__qualname__}`, which is a "
                        "`Module`, but not a strict `Module`."
                    )
                # Invariant: concrete means final.
                if not _is_abstract(base):
                    raise TypeError(
                        "Every strict `eqx.Module` must be either abstract or final. "
                        "This means that it is not possible to inherit from a concrete "
                        f"strict `eqx.Module`. `{cls.__module__}.{cls.__qualname__}` "
                        "is a strict Module inheriting from "
                        f"`{base.__module__}.{base.__qualname__}`, which is a concrete "
                        "strict `Module`."
                    )
                # Invariant: field definitions and __init__ methods must all appear on
                # just one class in a hierarchy.
                base_num_fields = len(dataclasses.fields(base))
                if (base_num_fields > 0) or (not _has_dataclass_init[base]):
                    # If we've added more fields, or added a custom init method, then
                    # error.
                    if len(dataclasses.fields(cls)) != base_num_fields:
                        raise TypeError(
                            "For readability, any custom `__init__` method, and all "
                            "fields, must all be defined on the same strict Module. "
                            f"{cls.__module__}.{cls.__qualname__} is a strict Module "
                            "that is attempting to add fields, and inherit from "
                            f"{base.__module__}.{base.__qualname__}. However the "
                            "latter already has fields or a custom `__init__` defined."
                        )
                    if added_custom_init:
                        raise TypeError(
                            "For readability, any custom `__init__` method, and all "
                            "fields, must all be defined on the same strict Module. "
                            f"{cls.__module__}.{cls.__qualname__} is a strict Module "
                            "that is attempting to define a custom `__init__` method, "
                            f"and inherit from {base.__module__}.{base.__qualname__}. "
                            "However the latter already has fields "
                            "or a custom `__init__` defined."
                        )
            if not strict_config.allow_abstract_name:
                has_abstract_name = cls.__name__.startswith(
                    "Abstract"
                ) or cls.__name__.startswith("_Abstract")
                if _is_abstract(cls):
                    if not has_abstract_name:
                        # Invariant: abstract classes have names beginning with
                        # `Abstract`.
                        main = (
                            "Abstract strict `eqx.Module`s must be named starting "
                            f"with 'Abstract' or '_Abstract'. Got {name} when defining "
                            f"{cls.__module__}.{cls.__qualname__}."
                        )
                        if _is_force_abstract[cls]:
                            raise TypeError(main)
                        inner = []
                        if len(cls.__abstractmethods__) > 0:
                            inner.append(
                                f"abstract methods: {list(cls.__abstractmethods__)}"
                            )
                        if len(cls.__abstractvars__) > 0:
                            inner.append(
                                f"abstract variables: {list(cls.__abstractvars__)}"
                            )
                        if len(cls.__abstractclassvars__) > 0:
                            inner.append(
                                (
                                    "abstract class variables: "
                                    f"{list(cls.__abstractclassvars__)}"
                                )
                            )
                        inner = ", ".join(inner)
                        inner = " " + inner + "."
                        raise TypeError(main + inner)
                else:
                    if has_abstract_name:
                        # Invariant: concrete classes do not have names beginning with
                        # `Abstract`.
                        raise TypeError(
                            "Concrete strict `eqx.Module`s should not have names "
                            f"starting with 'Abstract' or '_Abstract'. Got '{name}' "
                            f"when defining {cls.__module__}.{cls.__qualname__}.",
                        )
            for k, v in cls.__dict__.items():
                if isinstance(v, _wrap_method):
                    v = v.method
                    # Invariant: concrete methods are not overridden.
                    if not getattr(v, "__isabstractmethod__", False):
                        for base in bases:
                            old_v = getattr(base, k, _dummy_abstract)
                            if not inspect.isfunction(old_v):
                                raise TypeError(
                                    "Strict `eqx.Module`s cannot override non-methods "
                                    "with methods. "
                                    f"`{cls.__module__}.{cls.__qualname__}.{k}` is "
                                    "attempting to override "
                                    f"`{base.__module__}.{base.__qualname__}.{k}`."
                                )
                            if not strict_config.allow_method_override and not getattr(
                                old_v, "__isabstractmethod__", False
                            ):
                                raise TypeError(
                                    "Strict `eqx.Module`s cannot override concrete "
                                    "methods. "
                                    f"`{cls.__module__}.{cls.__qualname__}.{k}` is "
                                    "attempting to override "
                                    f"`{base.__module__}.{base.__qualname__}.{k}`."
                                )
        # [Step 6] Register as a pytree.
        jtu.register_pytree_with_keys(
            cls,
            flatten_with_keys=ft.partial(_flatten_module, with_keys=True),  # pyright: ignore
            flatten_func=ft.partial(_flatten_module, with_keys=False),  # pyright: ignore
            unflatten_func=ft.partial(_unflatten_module, cls),  # pyright: ignore
        )
        # Done!
        return cls

    @property
    def __signature__(cls):
        # Use signature of __init__ method for non-callable equinox modules
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.values())[1:]  # Remove self parameter
        return sig.replace(parameters=params)

    # This method is called whenever you initialise a module: `MyModule(...)`
    def __call__(cls, *args, **kwargs):
        __tracebackhide__ = True
        if _is_force_abstract[cls]:
            # Any other is-abstract checks will be handled in super().__call__.
            raise TypeError("Cannot instantiate abstract `equinox.Module`.")
        if _has_dataclass_init[cls]:
            for x in jtu.tree_leaves((args, kwargs)):
                _warn_jax_transformed_function(cls, x)
            # else it's handled in __setattr__, but that isn't called here.
        # [Step 1] Modules are immutable -- except during construction. So defreeze
        # before init.
        initable_cls = _make_initable_wrapper(cls)
        # [Step 2] Instantiate the class as normal.
        self = super(_ActualModuleMeta, initable_cls).__call__(*args, **kwargs)
        assert not _is_abstract(cls)
        # [Step 3] Check that all fields are occupied.
        missing_names = {
            field.name
            for field in dataclasses.fields(cls)  # pyright: ignore
            # Not `vars` or `__dict__`, to allow for `property`s overwriting a field.
            # Not recommended, but allowable for backward compatibility.
            if field.name not in dir(self)
        }
        if len(missing_names):
            raise ValueError(
                f"The following fields were not initialised during __init__: "
                f"{missing_names}"
            )
        # [Step 3.5] Prevent arrays from being marked as static
        for field in dataclasses.fields(self):
            if field.metadata.get("static", False):
                if any(
                    jtu.tree_map(
                        is_array, jtu.tree_flatten(getattr(self, field.name))[0]
                    )
                ):
                    warnings.warn(
                        "A JAX array is being set as static! This can result "
                        "in unexpected behavior and is usually a mistake to do.",
                        stacklevel=2,
                    )
        # Freeze.
        object.__setattr__(self, "__class__", cls)
        # [Step 4] Run any custom validators. (After freezing; as they run
        # unconditionally across the whole MRO, they aren't allowed to mutate.)
        for kls in cls.__mro__:
            try:
                check = kls.__dict__["__check_init__"]
            except KeyError:
                pass
            else:
                check(self)
        return self

    # This bit is kind of sneaky. This is the reason that `_has_dataclass_init` is done
    # after dataclass registration -- we sneakily treat it as a marker for whether
    # dataclass registration has happened yet. And if it hasn't, we hide any
    # `__wrapped__` attribute. We want such attributes for the sake of
    # `module_update_wrapper`, but if `dataclass` sees it then it tries to follow it.
    def __getattribute__(cls, item):
        value = super().__getattribute__(item)
        if (
            item == "__wrapped__"
            and isinstance(value, property)
            and cls not in _has_dataclass_init
        ):
            raise AttributeError
        else:
            return value

    def __setattr__(cls, item, value):
        if _not_magic(item) and inspect.isfunction(value):
            value = _wrap_method(value)
        super().__setattr__(item, value)


if TYPE_CHECKING:

    @dataclass_transform(field_specifiers=(dataclasses.field, field, static_field))
    class _ModuleMeta(abc.ABCMeta):
        __abstractvars__: frozenset[str]
        __abstractclassvars__: frozenset[str]
else:
    _ModuleMeta = _ActualModuleMeta


def _is_special_form(cls):
    # This function is basically a heuristic hack.
    # If you're getting spurious warnings from this, and think you have another kind of
    # class that should be excluded from Equinox's checks, then please open a GitHub
    # issue: https://github.com/patrick-kidger/equinox/issues
    if cls is _Initable:
        return True
    if cls.__module__ in ("typing", "typing_extensions", "collections.abc"):
        return True
    if Protocol in cls.__bases__:
        return True
    return False


def _not_magic(k: str) -> bool:
    return not (k.startswith("__") and k.endswith("__"))


class _wrap_method:
    def __init__(self, method):
        self.method = method
        if getattr(self.method, "__isabstractmethod__", False):
            self.__isabstractmethod__ = self.method.__isabstractmethod__

    def __get__(self, instance, owner):
        if instance is None:
            return self.method
        else:
            # Why `inplace=True`?
            # This is safe because the `BoundMethod` was only instantiated here.
            # This is necessary so that `_method.__self__ is instance`, which is used
            # as part of a no-cycle check in `_make_initable`.
            _method = _module_update_wrapper(
                BoundMethod(self.method, instance), None, inplace=True
            )
            return _method


_dummy_abstract = abc.abstractmethod(lambda self: 1)
_is_force_abstract = weakref.WeakKeyDictionary()
_is_strict = weakref.WeakKeyDictionary()
_has_dataclass_init = weakref.WeakKeyDictionary()


def _is_abstract(cls):
    return (
        _is_force_abstract[cls]
        or len(cls.__abstractmethods__) > 0
        or len(cls.__abstractvars__) > 0
        or len(cls.__abstractclassvars__) > 0
    )


_wrapper_field_names = {
    "__module__",
    "__name__",
    "__qualname__",
    "__doc__",
    "__annotations__",
}


class _Initable:
    # Prevent `__init_subclass__` from triggering when creating initable versions of
    # classes.
    def __init_subclass__(cls, **kwargs):
        del kwargs


_transform_types = {
    type(transform(lambda x: x))
    for transform in (
        jax.jit,
        jax.grad,
        jax.vmap,
        jax.value_and_grad,
        jax.jacfwd,
        jax.jacrev,
        jax.hessian,
        jax.custom_jvp,
        jax.custom_vjp,
        jax.checkpoint,  # pyright: ignore
        jax.pmap,
    )
}


def _warn_jax_transformed_function(cls, x):
    # not `isinstance`, just in case JAX every tries to override `__instancecheck__`.
    if type(x) in _transform_types:

        class _JaxTransformException(Exception):
            pass

        def _is_array_like(x):
            if is_array_like(x):
                raise _JaxTransformException

        while True:
            try:
                x = x.__wrapped__
            except AttributeError:
                break
            try:
                jtu.tree_map(_is_array_like, x)
            except _JaxTransformException:
                warnings.warn(
                    f"""
Possibly assigning a JAX-transformed callable as an attribute on
{cls.__module__}.{cls.__qualname__}. This will not have any of its parameters updated.

For example, the following code is buggy:
```python
class MyModule(eqx.Module):
vmap_linear: Callable

def __init__(self, ...):
    self.vmap_linear = jax.vmap(eqx.nn.Linear(...))

def __call__(self, ...):
    ... = self.vmap_linear(...)
```
This is because the callable returned from `jax.vmap` is *not* a PyTree. This means that
the parameters inside the `eqx.nn.Linear` layer will not receive gradient updates.

You can most easily fix this either by applying the wrapper at `__call__` time:
```python
class MyModule(eqx.Module):
linear: Callable

def __init__(self, ...):
    self.linear = eqx.nn.Linear(...)

def __call__(self, ...):
    ... = jax.vmap(self.linear)(...)
```
or by using `eqx.filter_vmap` instead (which *does* return a PyTree):
```python
class MyModule(eqx.Module):
vmap_linear: Callable

def __init__(self, ...):
    self.vmap_linear = eqx.filter_vmap(eqx.nn.Linear(...))

def __call__(self, ...):
    ... = self.vmap_linear(...)
```
""",
                    stacklevel=3,
                )
                break


def _make_initable_wrapper(cls: _ActualModuleMeta) -> _ActualModuleMeta:
    post_init = getattr(cls, "__post_init__", None)
    return _make_initable(cls, cls.__init__, post_init, wraps=False)


@ft.lru_cache(maxsize=128)
def _make_initable(
    cls: _ActualModuleMeta, init, post_init, wraps: bool
) -> _ActualModuleMeta:
    # Used as part of the key. Don't cache if these have changed.
    # In practice, monkey-patching these on the class -- after you've already
    # instantiated it somewhere! -- is an *ahem*, adventurous, thing to do. But never
    # let it be said that Equinox doesn't support you in your questionable life choices!
    del init, post_init

    if wraps:
        field_names = _wrapper_field_names
    else:
        field_names = {field.name for field in dataclasses.fields(cls)}

    class _InitableModule(_Initable, cls):
        pass

    def __setattr__(self, name, value):
        if name in field_names:
            if isinstance(value, BoundMethod) and value.__self__ is self:
                raise ValueError(
                    """Cannot assign methods in __init__.

That is, something like the following is not allowed:
```
class MyModule(eqx.Module):
    foo: Callable

    def __init__(self):
        self.foo = self.bar

    def bar(self):
        ...
```
this is because Equinox modules are PyTrees -- but the above does not have a tree
structure! `self.foo.__self__` is a cycle that brings us back to `self`.

In the above example, you probably want something like this instead:
```
class MyModule(eqx.Module):
    @property
    def foo(self):
        return self.bar

    def bar(self):
        ...
```
so that you can still use `self.foo`, but it is not stored in the PyTree structure.

This is a check that was introduced in Equinox v0.11.0. Before this, the above error
went uncaught, possibly leading to silently wrong behaviour.
"""
                )
            else:
                _warn_jax_transformed_function(type(self), value)
                object.__setattr__(self, name, value)
        else:
            raise AttributeError(f"Cannot set attribute {name}")

    # Done like this to avoid dataclasses complaining about overriding setattr on a
    # frozen class.
    _InitableModule.__setattr__ = __setattr__
    # Make beartype happy
    _InitableModule.__init__ = cls.__init__  # pyright: ignore
    # Appears in AbstractVar error messages
    _InitableModule.__name__ = cls.__name__
    _InitableModule.__qualname__ = cls.__qualname__
    # I don't have a specific use-case for this but it's probably good practice.
    _InitableModule.__module__ = cls.__module__

    return _InitableModule


cache_clears.append(_make_initable.cache_clear)


def _convert_fields(module, init: bool):
    for field in dataclasses.fields(module):
        if field.init is init:
            try:
                converter = field.metadata["converter"]
            except KeyError:
                pass
            else:
                try:
                    value = getattr(module, field.name)
                except AttributeError:
                    # Let the all-fields-are-filled check handle the error.
                    pass
                else:
                    setattr(module, field.name, converter(value))


# Used to provide a pretty repr when doing `jtu.tree_structure(SomeModule(...))`.
@dataclass()
class _FlattenedData:
    dynamic_field_names: tuple
    static_field_names: tuple
    static_field_values: tuple
    wrapper_field_names: tuple
    wrapper_field_values: tuple

    def __repr__(self):
        x = (
            self.dynamic_field_names,
            self.static_field_names,
            self.static_field_values,
        )
        return repr(x)[1:-1]


def _flatten_module(module: "Module", with_keys: bool):
    # Subnodes in the PyTree
    dynamic_field_names = []
    dynamic_field_values = []
    # Static metadata, placed in aux.
    static_field_names = []
    static_field_values = []
    # Python metadata like `__doc__` and `__module__`.
    wrapper_field_names = []
    wrapper_field_values = []

    for field_ in dataclasses.fields(module):
        name = field_.name
        try:
            # Not `getattr` so that we don't pick up `property`s.
            value = module.__dict__[name]
        except KeyError:
            # Uninitialised values during `__init__`, or when `property`s overwrite a
            # field.
            continue
        if field_.metadata.get("static", False):
            static_field_names.append(name)
            static_field_values.append(value)
        else:
            dynamic_field_names.append(name)
            if with_keys:
                dynamic_field_values.append((jtu.GetAttrKey(name), value))
            else:
                dynamic_field_values.append(value)
    sentinel = object()
    for name in _wrapper_field_names:
        value = getattr(module, name, sentinel)
        if value is not sentinel:
            wrapper_field_names.append(name)
            wrapper_field_values.append(value)
    aux = _FlattenedData(
        tuple(dynamic_field_names),
        tuple(static_field_names),
        tuple(static_field_values),
        tuple(wrapper_field_names),
        tuple(wrapper_field_values),
    )
    return tuple(dynamic_field_values), aux


def _unflatten_module(cls: type["Module"], aux: _FlattenedData, dynamic_field_values):
    # This doesn't go via `__init__`. A user may have done something nontrivial there,
    # and the field values may be dummy values as used in various places throughout JAX.
    # See also
    # https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization,
    # which was (I believe) inspired by Equinox's approach here.
    module = object.__new__(cls)
    for name, value in zip(aux.dynamic_field_names, dynamic_field_values):
        object.__setattr__(module, name, value)
    for name, value in zip(aux.static_field_names, aux.static_field_values):
        object.__setattr__(module, name, value)
    for name, value in zip(aux.wrapper_field_names, aux.wrapper_field_values):
        object.__setattr__(module, name, value)
    return module


class Module(metaclass=_ModuleMeta):
    """Base class. Create your model by inheriting from this.

    This will make your model a
    [dataclass](https://docs.python.org/3/library/dataclasses.html) and a
    [pytree](https://jax.readthedocs.io/en/latest/pytrees.html).

    **Fields**

    Specify all its fields at the class level (identical to
    [dataclasses](https://docs.python.org/3/library/dataclasses.html)). This defines
    its children as a PyTree.

    ```python
    class MyModule(equinox.Module):
        weight: jax.Array
        bias: jax.Array
        submodule: equinox.Module
    ```

    **Initialisation**

    A default `__init__` is automatically provided, which just fills in fields with
    the arguments passed. For example `MyModule(weight, bias, submodule)`.

    Alternatively (quite commonly) you can provide an `__init__` method yourself:

    ```python
    class MyModule(equinox.Module):
        weight: jax.Array
        bias: jax.Array
        submodule: equinox.Module

        def __init__(self, in_size, out_size, key):
            wkey, bkey, skey = jax.random.split(key, 3)
            self.weight = jax.random.normal(wkey, (out_size, in_size))
            self.bias = jax.random.normal(bkey, (out_size,))
            self.submodule = equinox.nn.Linear(in_size, out_size, key=skey)
    ```

    **Methods**

    It is common to create some methods on the class -- for example to define the
    forward pass of a model.

    ```python
    class MyModule(equinox.Module):
        ...  # as above

        def __call__(self, x):
            return self.submodule(x) + self.weight @ x + self.bias
    ```

    !!! tip

        You don't have to define `__call__`:

        - You can define other methods if you want.
        - You can define multiple methods if you want.
        - You can define no methods if you want. (And just use `equinox.Module` as a
            nice syntax for custom PyTrees.)

        No method is special-cased.

    **Usage**

    After you have defined your model, then you can use it just like any other
    PyTree -- that just happens to have some methods attached. In particular you can
    pass it around across `jax.jit`, `jax.grad` etc. in exactly the way that you're
    used to.

    !!! example

        If you wanted to, then it would be completely safe to do

        ```python
        class MyModule(equinox.Module):
            ...

            @jax.jit
            def __call__(self, x):
                ...
        ```

        because `self` is just a PyTree. Unlike most other neural network libraries,
        you can mix Equinox and native JAX without any difficulties at all.

    !!! tip "For fans of strong typing."

        Equinox modules are all [ABCs](https://docs.python.org/3/library/abc.html)
        by default. This means you can use
        [`abc.abstractmethod`](https://docs.python.org/3/library/abc.html#abc.abstractmethod).
        You can also create abstract instance attributes or abstract class
        attributes, see [`equinox.AbstractVar`][] and
        [`equinox.AbstractClassVar`][].
    """  # noqa: E501

    #     [`equinox.AbstractClassVar`][]. Finally, some optional strict type-checking
    #     may be enabled by passing `strict=True`, e.g.
    #     `class Foo(eqx.Module, strict=True)`; see
    #     [strict modules](../advanced_fields/#strict-modules) for details.
    # """  # noqa: E501

    def __hash__(self):
        return hash(tuple(jtu.tree_leaves(self)))

    def __eq__(  # pyright: ignore
        self, other
    ) -> Union[bool, np.bool_, Bool[Array, ""]]:
        return tree_equal(self, other)

    def __repr__(self):
        return tree_pformat(self)


# Not using `jax.tree_util.Partial` as it doesn't implement __eq__ very well. See #480.
class BoundMethod(Module):
    """Just like a normal Python bound method... except that this one is a PyTree!

    This stores `__self__` as a subnode.
    """

    __func__: types.FunctionType = field(static=True)
    __self__: Module

    def __call__(self, *args, **kwargs):
        __tracebackhide__ = True
        return self.__func__(self.__self__, *args, **kwargs)

    @property
    def __wrapped__(self):
        return self.__func__.__get__(  # pyright: ignore
            self.__self__, type(self.__self__)
        )

    # This should be unnecessary in principle. In practice something goes wrong on
    # Python 3.9 and it returns the wrong thing.
    @property
    def __signature__(self):
        return inspect.signature(self.__wrapped__)


#
# Part 3: some downstream pieces. These don't actually affect the core `Module`
# abstraction, but this file is a convenient place to put them.
#


def module_update_wrapper(
    wrapper: Module, wrapped: Optional[Callable[_P, _T]] = None
) -> Callable[_P, _T]:
    """Like `functools.update_wrapper` (or its better-known cousin, `functools.wraps`),
    but acts on [`equinox.Module`][]s, and does not modify its input (it returns the
    updated module instead).

    !!! Example

        ```python
        class Wrapper(eqx.Module):
            fn: Callable

            def __call__(self, *args, **kwargs):
                return self.fn(*args, **kwargs)

            @property
            def __wrapped__(self):
                return self.fn

        def make_wrapper(fn):
            return eqx.module_update_wrapper(Wrapper(fn))
        ```

    For example, [`equinox.filter_jit`][] returns a module representing the JIT'd
    computation. `module_update_wrapper` is used on this module to indicate that this
    JIT'd computation wraps the original one. (Just like how `functools.wraps` is used.)

    Note that as in the above example, the wrapper class must supply a `__wrapped__`
    property, which redirects to the wrapped object.

    **Arguments:**

    - `wrapper`: the instance of the wrapper.
    - `wrapped`: optional, the callable that is being wrapped. If omitted then
        `wrapper.__wrapped__` will be used.

    **Returns:**

    A copy of `wrapper`, with the attributes `__module__`, `__name__`, `__qualname__`,
    `__doc__`, and `__annotations__` copied over from the wrapped function.
    """
    return _module_update_wrapper(wrapper, wrapped, inplace=False)


def _module_update_wrapper(
    wrapper: Module, wrapped: Optional[Callable[_P, _T]], inplace: bool
) -> Callable[_P, _T]:
    cls = wrapper.__class__
    if not isinstance(getattr(cls, "__wrapped__", None), property):
        raise ValueError("Wrapper module must supply `__wrapped__` as a property.")

    if wrapped is None:
        wrapped = wrapper.__wrapped__  # pyright: ignore

    if not inplace:
        # Make a clone, to avoid mutating the original input.
        leaves, treedef = jtu.tree_flatten(wrapper)
        wrapper = jtu.tree_unflatten(treedef, leaves)

    initable_cls = _make_initable(cls, None, None, wraps=True)
    object.__setattr__(wrapper, "__class__", initable_cls)
    try:
        # Like `ft.update_wrapper(wrapper, wrapped, updated=())`.
        # We don't update __dict__ as it's common/possible for wrapper and wrapped to
        # both be classes implementing __call__, in which case copying __dict__ over
        # basically just breaks the wrapper class.
        # We don't set __wrapped__, and instead demand that the wrapper class tell us
        # how to redirect to the wrapped object. This is avoid duplicating part of the
        # PyTree.
        for attr in _wrapper_field_names:
            try:
                value = getattr(wrapped, attr)
            except AttributeError:
                pass
            else:
                setattr(wrapper, attr, value)
    finally:
        object.__setattr__(wrapper, "__class__", cls)
    return cast(Callable[_P, _T], wrapper)


class Partial(Module):
    """Like `functools.partial`, but treats the wrapped function, and partially-applied
    value: Any = field(static=True)
    args and kwargs, as a PyTree.

    This is very much like `jax.tree_util.Partial`. The difference is that the JAX
    version requires that `func` be specifically a *function* -- and will silently
    misbehave if given any non-function callable, e.g. [`equinox.nn.MLP`][]. In contrast
    the Equinox version allows for arbitrary callables.
    """

    func: Callable
    args: tuple[Any, ...]
    keywords: dict[str, Any]

    def __init__(self, func, /, *args, **kwargs):
        """**Arguments:**

        - `func`: the callable to partially apply.
        - `*args`: any positional arguments to apply.
        - `**kwargs`: any keyword arguments to apply.
        """
        self.func = func
        self.args = args
        self.keywords = kwargs

    def __call__(self, *args, **kwargs):
        """Call the wrapped `self.func`.

        **Arguments:**

        - `*args`: the arguments to apply. Passed after those arguments passed during
            `__init__`.
        - `**kwargs`: any keyword arguments to apply.

        **Returns:**

        The result of the wrapped function.
        """
        return self.func(*self.args, *args, **kwargs, **self.keywords)


class Static(Module):
    """Wraps a value into a `eqx.field(static=True)`.

    This is useful to treat something as just static metadata with respect to a JAX
    transformation; for example this is used to return non-arrays from a filtered
    transform.
    """

    _leaves: list[Any] = field(static=True)
    _treedef: PyTreeDef = field(static=True)  # pyright: ignore

    def __init__(self, value: Any):
        # By flattening, we handle pytrees without `__eq__` methods.
        # When comparing static metadata for equality, this means we never actually
        # call `value.__eq__`.
        self._leaves, self._treedef = jtu.tree_flatten(value)

    @property
    def value(self):
        return jtu.tree_unflatten(self._treedef, self._leaves)
