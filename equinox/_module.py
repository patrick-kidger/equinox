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
from typing import Any, cast, Optional, TYPE_CHECKING, TypeVar, Union
from typing_extensions import dataclass_transform, ParamSpec

import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, Bool, PyTreeDef

from ._better_abstract import ABCMeta, dataclass
from ._caches import internal_lru_caches
from ._doc_utils import doc_repr
from ._pretty_print import tree_pformat
from ._tree import tree_equal


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
        `bool`/`int`/`float`/`complex` values to JAX arrays.
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
        metadata = dict(kwargs["metadata"])
    except KeyError:
        metadata = kwargs["metadata"] = {}
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
    return dataclasses.field(**kwargs)


#
# Part 2: Modules!
# This is the core of Equinox.
#


# Inherits from ABCMeta to support `eqx.{AbstractVar, AbstractClassVar}` and
# `abc.abstractmethod`.
class _ModuleMeta(ABCMeta):  # pyright: ignore

    # This method is called whenever you definite a module: `class Foo(eqx.Module): ...`
    def __new__(mcs, name, bases, dict_, /, strict: bool = False, **kwargs):
        # [Step 1] We support an optional `strict` mode for Rust-like strictness in the
        # type checking.
        # In practice this is probably too much for your average user, but it's a great
        # way to build robust libraries.
        if strict:
            for base in bases:
                if not issubclass(base, Module):
                    raise TypeError(
                        "Strict `eqx.Module`s must only inherit from other subclasses "
                        "of `eqx.Module`."
                    )
        # [Step 2] Create the class as normal.
        cls = super().__new__(mcs, name, bases, dict_, **kwargs)
        # [Step 3] Arrange for bound methods to be treated as PyTrees as well. This
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
                if strict:
                    if not getattr(v, "__isabstractmethod__", False):
                        for base in bases:
                            old_v = getattr(base, k, _dummy_abstract)
                            if not inspect.isfunction(old_v):
                                raise TypeError(
                                    "Strict `eqx.Module`s cannot override non-methods "
                                    "with methods."
                                )
                            if not getattr(old_v, "__isabstractmethod__", False):
                                raise TypeError(
                                    "Strict `eqx.Module`s cannot override concrete "
                                    "methods."
                                )
        # [Step 4] Create a default `__init__` method if a user method isn't provided.
        #
        # If as a superclass has a custom `__init__`, then don't create a default
        # `__init__` here. (Otherwise e..g if `B` has a custom init then
        # `class A(B): pass` would set a dataclass init on `A`.)
        # If a superclass has a default `__init__`, then do create a new default one
        # here. (Dataclass default `__init__`s don't call `super()`, so they must be
        # overriden directly.)
        if "__init__" in cls.__dict__:
            _init = False
        else:
            for kls in cls.__mro__:
                try:
                    _init = _has_dataclass_init[kls]
                except KeyError:
                    pass
                else:
                    break
            else:
                assert name == "Module"
                _init = True  # eqx.Module itself
        if _init:
            # Dataclass-generated __init__
            init_doc = cls.__init__.__doc__
        if not _init:
            # User-provided __init__
            # _Initable check to avoid printing out another warning on initialisation.
            if getattr(cls, "__post_init__", None) is not None and not issubclass(
                cls, _Initable
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
                    "The above is purely how Python dataclasses work, and has nothing "
                    "to do with Equinox!\n"
                    "If you are using `__post_init__` to check that certain invariants "
                    "hold, then consider using `__check_init__` instead. This is an "
                    "Equinox-specific extension that is always ran. See here for more "
                    "details: "
                    "https://docs.kidger.site/equinox/api/module/advanced_fields/#checking-invariants"  # noqa: E501
                )
        # [Step 5] Register as a dataclass.
        cls = dataclass(eq=False, repr=False, frozen=True, init=_init)(
            cls  # pyright: ignore
        )
        # [Step 4b] -- finish off the business of default `__init__` methods.
        # (This part has to happen after dataclass registration.)
        _has_dataclass_init[cls] = _init
        if _init:
            # Assign `__doc__` in case its been manually overriden:
            # ```
            # class Foo(eqx.Module):
            #     x: int
            #
            # Foo.__init__.__doc__ = "Foo should be called with with an integer `x`."
            #
            # class Bar(Foo):
            #     pass
            # ```
            # With `Bar.__init__.__doc__` used during documentation generation.
            cls.__init__.__doc__ = init_doc  # pyright: ignore
            # TODO: is this next line still necessary?
            cls.__init__.__module__ = cls.__module__
        # [Step 6] Check strict abstract modules.
        if strict:
            if (
                len(cls.__abstractmethods__) > 0
                or len(cls.__abstractvars__) > 0
                or len(cls.__abstractclassvars__) > 0
            ):
                if not cls.__name__.startswith("Abstract"):
                    raise TypeError(
                        "Abstract strict `eqx.Module`s must be named starting with "
                        "`Abstract`."
                    )
                if not _init:
                    raise TypeError(
                        "Abstract strict `eqx.Module`s cannot have `__init__` methods."
                    )
                if len(dataclasses.fields(cls)) > 0:
                    raise TypeError(
                        "Abstract strict `eqx.Module`s cannot have fields. (You "
                        "probably meant to mark them as `eqx.AbstractVar[...]` "
                        "instead.)"
                    )
        # [Step 7] Register as a pytree.
        jtu.register_pytree_with_keys(
            cls,
            flatten_with_keys=ft.partial(
                _flatten_module, with_keys=True
            ),  # pyright: ignore
            flatten_func=ft.partial(
                _flatten_module, with_keys=False
            ),  # pyright: ignore
            unflatten_func=ft.partial(_unflatten_module, cls),  # pyright: ignore
        )
        # Done!
        return cls

    # This method is called whenever you initialise a module: `MyModule(...)`
    def __call__(cls, *args, **kwargs):
        # [Step 1] Modules are immutable -- except during construction. So defreeze
        # before init.
        initable_cls = _make_initable(cls, wraps=False)
        # [Step 2] Instantiate the class as normal. (`__init__` and `__post_init__`)
        self = super(_ModuleMeta, initable_cls).__call__(*args, **kwargs)
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
        # [Step 4] Run any custom converters.
        for field in dataclasses.fields(self):
            try:
                converter = field.metadata["converter"]
            except KeyError:
                pass
            else:
                setattr(self, field.name, converter(getattr(self, field.name)))
        object.__setattr__(self, "__class__", cls)
        # [Step 5] Run any custom validators.
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


if TYPE_CHECKING:

    @dataclass_transform(field_specifiers=(dataclasses.field, field, static_field))
    class _ModuleMeta(abc.ABCMeta):
        pass


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
_has_dataclass_init = weakref.WeakKeyDictionary()


_wrapper_field_names = {
    "__module__",
    "__name__",
    "__qualname__",
    "__doc__",
    "__annotations__",
}


class _Initable:
    pass


@ft.lru_cache(maxsize=128)
def _make_initable(cls: _ModuleMeta, wraps: bool) -> _ModuleMeta:
    if wraps:
        field_names = _wrapper_field_names
    else:
        field_names = {
            field.name for field in dataclasses.fields(cls)  # pyright: ignore
        }

    class _InitableModule(cls, _Initable):  # pyright: ignore
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

    return _InitableModule


internal_lru_caches.append(_make_initable)


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
        [`equinox.AbstractClassVar`][]. Finally, some optional strict type-checking
        may be enabled by passing `strict=True`, e.g.
        `class Foo(eqx.Module, strict=True)`; see [strict modules](../advanced_fields/#strict-modules) for details.
    """  # noqa: E501

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
        return self.__func__(self.__self__, *args, **kwargs)

    @property
    def __wrapped__(self):
        return self.__func__.__get__(  # pyright: ignore
            self.__self__, type(self.__self__)
        )


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

    initable_cls = _make_initable(cls, wraps=True)
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
