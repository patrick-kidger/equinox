import functools as ft
import inspect
import weakref
from dataclasses import field, fields
from typing import Any, Callable, cast, Dict, Tuple, Type, TypeVar, Union
from typing_extensions import dataclass_transform, ParamSpec

import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, Bool

from ._better_abstract import ABCMeta, dataclass
from ._pretty_print import tree_pformat
from ._tree import tree_equal


_P = ParamSpec("_P")
_T = TypeVar("_T")


def static_field(**kwargs):
    """Used for marking that a field should _not_ be treated as a leaf of the PyTree
    of a [`equinox.Module`][]. (And is instead treated as part of the structure, i.e.
    as extra metadata.)

    !!! example

        ```python
        class MyModule(equinox.Module):
            normal_field: int
            static_field: int = equinox.static_field()

        mymodule = MyModule("normal", "static")
        leaves, treedef = jtu.tree_flatten(mymodule)
        assert leaves == ["normal"]
        assert "static" in str(treedef)
        ```

    In practice this should rarely be used; it is usually preferred to just filter
    out each field with `eqx.partition` whenever you need to select only some fields.

    **Arguments:**

    - `**kwargs`: If any are passed then they are passed on to `dataclass.field`.
        (Recall that Equinox uses dataclasses for its modules.)
    """
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
        return jtu.Partial(self.method, instance)


def _not_magic(k: str) -> bool:
    return not (k.startswith("__") and k.endswith("__"))


_has_dataclass_init = weakref.WeakKeyDictionary()


# Inherits from ABCMeta as a convenience for a common use-case.
# It's not a feature we use ourselves.
@dataclass_transform(field_specifiers=(field, static_field))
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
        _has_dataclass_init[cls] = _init
        if _init:
            init_doc = cls.__init__.__doc__
        cls = dataclass(eq=False, repr=False, frozen=True, init=_init)(
            cls  # pyright: ignore
        )
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

    def __call__(cls, *args, **kwargs):
        # Defreeze it during __init__
        initable_cls = _make_initable(cls, wraps=False)
        self = super(_ModuleMeta, initable_cls).__call__(*args, **kwargs)
        object.__setattr__(self, "__class__", cls)
        missing_names = {
            field.name
            for field in fields(cls)  # pyright: ignore
            if field.init and field.name not in dir(self)
        }
        if len(missing_names):
            raise ValueError(
                f"The following fields were not initialised during __init__: "
                f"{missing_names}"
            )
        return self


_wrapper_field_names = {
    "__module__",
    "__name__",
    "__qualname__",
    "__doc__",
    "__annotations__",
    "__wrapped__",
}


@ft.lru_cache(maxsize=128)
def _make_initable(cls: _ModuleMeta, wraps: bool) -> _ModuleMeta:
    if wraps:
        field_names = _wrapper_field_names
    else:
        field_names = {field.name for field in fields(cls)}  # pyright: ignore

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
    for field_ in fields(module):
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


def _unflatten_module(cls: Type["Module"], aux: _FlattenedData, dynamic_field_values):
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
    """

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

        def make_wrapper(fn):
            return eqx.module_update_wrapper(Wrapper(fn), fn)
        ```

    For example, [`equinox.filter_jit`][] returns a module representing the JIT'd
    computation. `module_update_wrapper` is used on this module to indicate that this
    JIT'd computation wraps the original one. (Just like how `functools.wraps` is used.)
    """
    cls = wrapper.__class__
    initable_cls = _make_initable(cls, wraps=True)
    object.__setattr__(wrapper, "__class__", initable_cls)
    try:
        # updated = ("__dict__",) is the default, but that's a bit much.
        # It's common/possible for wrapper and wrapped to both be classes
        # implementing __call__, in which case copying __dict__ over basically
        # just breaks the wrapper class.
        ft.update_wrapper(wrapper, wrapped, updated=())  # pyright: ignore
    finally:
        object.__setattr__(wrapper, "__class__", cls)
    return cast(Callable[_P, _T], wrapper)


class Static(Module):
    value: Any = static_field()


class Partial(Module):
    """Like `functools.partial`, but treats the wrapped function, and partially-applied
    args and kwargs, as a PyTree.

    This is very much like `jax.tree_util.Partial`. The difference is that the JAX
    version requires that `func` be specifically a *function* -- and will silently
    misbehave if given any non-function callable, e.g. [`equinox.nn.MLP`][]. In contrast
    the Equinox version allows for arbitrary callables.
    """

    func: Callable
    args: Tuple[Any, ...]
    keywords: Dict[str, Any]

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
