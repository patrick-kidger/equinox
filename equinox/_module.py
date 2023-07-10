import dataclasses
import functools as ft
import inspect
import weakref
from collections.abc import Callable
from typing import Any, cast, TypeVar, Union
from typing_extensions import dataclass_transform, ParamSpec

import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, Bool

from ._better_abstract import ABCMeta, dataclass
from ._caches import internal_lru_caches
from ._doc_utils import doc_repr
from ._pretty_print import tree_pformat
from ._tree import tree_equal


_P = ParamSpec("_P")
_T = TypeVar("_T")


def static_field(**kwargs):
    """Deprecated in favour of [`equinox.field`][], i.e. `eqx.field(static=True)`."""
    return field(**kwargs, static=True)


_identity = doc_repr(lambda x: x, "lambda x: x")


def field(
    *, converter: Callable[[Any], Any] = _identity, static: bool = False, **kwargs
):
    """Equinox supports extra functionality on top of the default dataclasses.

    **Arguments:**

    - `converter`: a function to call on this field when the model is initialised. For
        example, `field(converter=jax.numpy.asarray)` to convert
        `bool`/`int`/`float`/`complex` values to JAX arrays.
    - `static`: whether the field should not interact with any JAX transform at all (by
        making it part of the PyTree structure rather than a leaf).
    - `**kwargs`: All other keyword arguments are passed on to `dataclass.field`.

    **Converter**

    !!! example

        ```python
        class MyModule(eqx.Module):
            foo: Array = eqx.field(converter=jax.numpy.asarray)

        mymodule = MyModule(1.0)
        assert isinstance(mymodule.foo, jax.Array)
        ```

    **Static**

    !!! example

        ```python
        class MyModule(eqx.Module):
            normal_field: int
            static_field: int = eqx.field(static=True)

        mymodule = MyModule("normal", "static")
        leaves, treedef = jax.tree_util.tree_flatten(mymodule)
        assert leaves == ["normal"]
        assert "static" in str(treedef)
        ```

    This means that it does not interact with any JAX transforms, like JIT or grad.
    This means that it's usually a bug to make JAX arrays be static fields.

    This is an advanced feature that should very rarely be used. It is preferred to
    just filter out each field with `eqx.partition` whenever you need to select only
    some fields.
    """
    try:
        metadata = dict(kwargs["metadata"])
    except KeyError:
        metadata = kwargs["metadata"] = {}
    if "converter" in metadata:
        raise ValueError("Cannot use metadata with `static` already set.")
    if "static" in metadata:
        raise ValueError("Cannot use metadata with `static` already set.")
    metadata["converter"] = converter
    if static:
        metadata["static"] = True
    return dataclasses.field(**kwargs)


class _wrap_method:
    def __init__(self, method):
        self.method = method
        if getattr(self.method, "__isabstractmethod__", False):
            self.__isabstractmethod__ = self.method.__isabstractmethod__

    def __get__(self, instance, owner):
        if instance is None:
            return self.method
        _method = ft.wraps(self.method)(jtu.Partial(self.method, instance))
        delattr(_method, "__wrapped__")
        return _method


def _not_magic(k: str) -> bool:
    return not (k.startswith("__") and k.endswith("__"))


_has_dataclass_init = weakref.WeakKeyDictionary()


# Inherits from ABCMeta as a convenience for a common use-case.
# It's not a feature we use ourselves.
@dataclass_transform(field_specifiers=(dataclasses.field, field, static_field))
class _ModuleMeta(ABCMeta):
    def __new__(mcs, name, bases, dict_):  # pyright: ignore
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
            init_doc = cls.__init__.__doc__
        cls = dataclass(eq=False, repr=False, frozen=True, init=_init)(
            cls  # pyright: ignore
        )
        # must happen after `dataclass(...)` we use this in `__getattribute__` to avoid
        # making any property(def __wrapped__) visible until then. We want to be able to
        # support property(def __wrapped__) for the sake of classes whose instances are
        # wrappers (i.e. via `module_update_wrapper`), but this means that a
        # `__wrapped__` object is also visible on the class object itself, despite not
        # pointing to a callable. `dataclass(...)` spots this and tries to use it form
        # a signature.
        _has_dataclass_init[cls] = _init
        if _init:
            cls.__init__.__doc__ = init_doc  # pyright: ignore
            cls.__init__.__module__ = cls.__module__
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
        return cls

    # See note above for `_has_dataclass_init`.
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

    def __call__(cls, *args, **kwargs):
        # Defreeze it during __init__
        initable_cls = _make_initable(cls, wraps=False)
        self = super(_ModuleMeta, initable_cls).__call__(*args, **kwargs)
        missing_names = {
            field.name
            for field in dataclasses.fields(cls)  # pyright: ignore
            if field.init and field.name not in dir(self)
        }
        if len(missing_names):
            raise ValueError(
                f"The following fields were not initialised during __init__: "
                f"{missing_names}"
            )
        for field in dataclasses.fields(self):
            try:
                converter = field.metadata["converter"]
            except KeyError:
                pass
            else:
                setattr(self, field.name, converter(getattr(self, field.name)))
        object.__setattr__(self, "__class__", cls)
        return self


_wrapper_field_names = {
    "__module__",
    "__name__",
    "__qualname__",
    "__doc__",
    "__annotations__",
}


@ft.lru_cache(maxsize=128)
def _make_initable(cls: _ModuleMeta, wraps: bool) -> _ModuleMeta:
    if wraps:
        field_names = _wrapper_field_names
    else:
        field_names = {
            field.name for field in dataclasses.fields(cls)  # pyright: ignore
        }

    class _InitableModule(cls):  # pyright: ignore
        pass

    # Done like this to avoid dataclasses complaining about overriding setattr on a
    # frozen class.
    def __setattr__(self, name, value):
        if name in field_names:
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(f"Cannot set attribute {name}")

    _InitableModule.__setattr__ = __setattr__
    # Make beartype happy
    _InitableModule.__init__ = cls.__init__  # pyright: ignore
    # Appears in AbstractVar error messages
    _InitableModule.__name__ = cls.__name__
    _InitableModule.__qualname__ = cls.__qualname__

    return _InitableModule


internal_lru_caches.append(_make_initable)


# This class exists primarily just to hide the wrapper fields from the PyTreeDef repr.
# Stuff like the docstring can be pretty long and entirely unhelpful to print out.
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
    dynamic_field_names = []
    dynamic_field_values = []
    static_field_names = []
    static_field_values = []
    wrapper_field_names = []
    wrapper_field_values = []
    for field_ in dataclasses.fields(module):
        name = field_.name
        try:
            value = module.__dict__[name]
        except KeyError:
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

    A default `__init__` is automatically provided, which just fills in fields with the
    arguments passed. For example `MyModule(weight, bias, submodule)`.

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

    After you have defined your model, then you can use it just like any other PyTree
    -- that just happens to have some methods attached. In particular you can pass it
    around across `jax.jit`, `jax.grad` etc. in exactly the way that you're used to.

    !!! example

        If you wanted to, then it would be completely safe to do

        ```python
        class MyModule(equinox.Module):
            ...

            @jax.jit
            def __call__(self, x):
                ...
        ```

        because `self` is just a PyTree. Unlike most other neural network libaries, you
        can mix Equinox and native JAX without any difficulties at all.

    !!! tip

        Equinox modules are all [ABCs](https://docs.python.org/3/library/abc.html) by
        default. This means you can use [`abc.abstractmethod`](https://docs.python.org/3/library/abc.html#abc.abstractmethod).
        You can also create abstract instance attributes or abstract class attributes,
        see [`equinox.AbstractVar`][] and [`equinox.AbstractClassVar`][].
    """  # noqa: E501

    def __hash__(self):
        return hash(tuple(jtu.tree_leaves(self)))

    def __eq__(  # pyright: ignore
        self, other
    ) -> Union[bool, np.bool_, Bool[Array, ""]]:
        return tree_equal(self, other)

    def __repr__(self):
        return tree_pformat(self)


# Modifies in-place, just like functools.update_wrapper
def module_update_wrapper(
    wrapper: Module, wrapped: Callable[_P, _T]
) -> Callable[_P, _T]:
    """Like `functools.update_wrapper` (or its better-known cousin, `functools.wraps`),
    but can be used on [`equinox.Module`][]s. (Which are normally immutable.)

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
            return eqx.module_update_wrapper(Wrapper(fn), fn)
        ```

    For example, [`equinox.filter_jit`][] returns a module representing the JIT'd
    computation. `module_update_wrapper` is used on this module to indicate that this
    JIT'd computation wraps the original one. (Just like how `functools.wraps` is used.)

    Note that as in the above example, the wrapper class must supply a `__wrapped__`
    property, which redirects to the wrapped object.
    """
    cls = wrapper.__class__
    if not isinstance(getattr(cls, "__wrapped__", None), property):
        raise ValueError("Wrapper module must supply `__wrapped__` as a property.")
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


class Static(Module):
    value: Any = field(static=True)


class Partial(Module):
    """Like `functools.partial`, but treats the wrapped function, and partially-applied
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
