import dataclasses
import functools as ft
import inspect
import itertools
import types
import warnings
import weakref
from collections.abc import Callable, Hashable
from typing import Any, cast, Final, ParamSpec, TypeVar
from typing_extensions import dataclass_transform

import jax
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, Bool, PyTree

from .._filters import is_array, is_array_like, is_inexact_array_like
from .._pretty_print import tree_pformat
from .._tree import tree_equal
from ._better_abstract import better_dataclass, BetterABCMeta
from ._field import field


# Legacy compatibibility API, passed to `strict` below.
def StrictConfig(force_abstact: bool = False, **kwargs):
    del kwargs
    if force_abstact:
        return None
    else:
        return False


wrapper_field_names: Final = {
    "__module__",
    "__name__",
    "__qualname__",
    "__doc__",
    "__annotations__",
}


_abstract_module_registry = weakref.WeakSet()
_has_dataclass_init = weakref.WeakKeyDictionary()
_flatten_sentinel = object()


# Used to provide a pretty repr when doing `jtu.tree_structure(SomeModule(...))`.
@dataclasses.dataclass(slots=True)
class _FlattenedData:
    dynamic_field_names: tuple
    static_fields: tuple[tuple[str, Any], ...]
    wrapper_fields: tuple[tuple[str, Any], ...]

    def __repr__(self):
        return repr((self.dynamic_field_names, self.static_fields))[1:-1]


class _ModuleFlattener:
    __slots__: tuple[str, str] = ("dynamic_fs", "static_fs")
    dynamic_fs: tuple[str, ...]
    static_fs: tuple[str, ...]

    def __init__(self, fields: tuple[dataclasses.Field[Any], ...]):
        dynamic_fs = []
        static_fs = []
        for f in fields:
            if f.metadata.get("static", False):
                static_fs.append(f.name)
            else:
                dynamic_fs.append(f.name)
        self.dynamic_fs = tuple(dynamic_fs)
        self.static_fs = tuple(static_fs)

    def flatten(self, obj: "Module") -> tuple[tuple[PyTree, ...], _FlattenedData]:
        get = obj.__dict__.get
        dynamic_fs = []
        dynamic_vs = []
        for k in self.dynamic_fs:
            v = get(k, _flatten_sentinel)
            if v is _flatten_sentinel:
                continue
            dynamic_fs.append(k)
            dynamic_vs.append(v)
        aux = _FlattenedData(
            tuple(dynamic_fs),
            tuple([(k, get(k, _flatten_sentinel)) for k in self.static_fs]),
            tuple([(k, get(k, _flatten_sentinel)) for k in wrapper_field_names]),
        )
        return tuple(dynamic_vs), aux

    def flatten_with_keys(
        self, obj: "Module"
    ) -> tuple[tuple[tuple[Any, PyTree], ...], _FlattenedData]:
        get = obj.__dict__.get
        dynamic_fs = []
        dynamic_vs = []
        for k in self.dynamic_fs:
            v = get(k, _flatten_sentinel)
            if v is _flatten_sentinel:
                continue
            dynamic_fs.append(k)
            dynamic_vs.append((jtu.GetAttrKey(k), v))
        aux = _FlattenedData(
            tuple(dynamic_fs),
            tuple([(k, get(k, _flatten_sentinel)) for k in self.static_fs]),
            tuple([(k, get(k, _flatten_sentinel)) for k in wrapper_field_names]),
        )
        return tuple(dynamic_vs), aux

    @staticmethod
    def unflatten_with_cls(
        module_cls: type["Module"],
        aux: _FlattenedData,
        dynamic_field_values: tuple[PyTree, ...],
    ) -> "Module":
        # This doesn't go via `__init__`. A user may have done something
        # nontrivial there, and the field values may be dummy values as used in
        # various places throughout JAX. See also
        # https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization,
        # which was (I believe) inspired by Equinox's approach here.
        module = object.__new__(module_cls)
        for name, value in zip(aux.dynamic_field_names, dynamic_field_values):
            object.__setattr__(module, name, value)
        for name, value in itertools.chain(aux.static_fields, aux.wrapper_fields):
            if value is not _flatten_sentinel:
                object.__setattr__(module, name, value)
        return module


def _error_method_assignment(self, value):
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
        jax.checkpoint,  # pyright: ignore[reportPrivateImportUsage]
        jax.pmap,
    )
}


class _JaxTransformException(Exception):
    pass


def _is_array_like(x):
    if is_array_like(x):
        raise _JaxTransformException


def _warn_jax_transformed_function(cls, x):
    # not `isinstance`, just in case JAX every tries to override `__instancecheck__`.
    if type(x) in _transform_types:
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


class _IdSet:
    __slots__ = ("_dict",)

    def __init__(self):
        self._dict = {}

    def __contains__(self, key):
        return id(key) in self._dict.keys()

    def add(self, key):
        if key not in self:
            id_key = id(key)
            # Hold on to `key` to be sure that `id(key)` does not get reallocated.
            self._dict[id_key] = key

    def remove(self, key):
        del self._dict[id(key)]


_currently_initialising = _IdSet()


# This deliberately does not pass `frozen_default=True`, as that clashes with custom
# `__init__` methods.
@dataclass_transform(field_specifiers=(dataclasses.field, field))
class _ModuleMeta(BetterABCMeta):
    def __new__(
        mcs,
        name,
        bases,
        namespace,
        *,
        is_abstract: bool = False,
        strict: None | bool = False,
        **kwargs,
    ):
        if strict is None:
            # Legacy compatibility API. Checking that this has the desired behaviour:
            #
            # - No argument => not abstract. Still true now, the default `strict=False`
            #    does not take this branch.
            # - `strict=False` => not abstract. Still true now as above.
            # - `strict=True` => not abstract. Still true now as above.
            # - `strict=StrictConfig(force_abstract=False)` => not abstract as
            #    `StrictConfig` returns `False` in this case.
            # - `strict=StrictConfig(force_abstract=True)` => abstract. And indeed then
            #    we take this branch as `StrictConfig` returns `None` in this case.
            is_abstract = True
        del strict

        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        if is_abstract:
            _abstract_module_registry.add(cls)

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
        _has_dataclass_init[cls] = has_dataclass_init
        # Check before `dataclass` adds an `__init__` method.
        if not has_dataclass_init and hasattr(cls, "__post_init__"):
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
        if has_dataclass_init:
            init_doc = cls.__init__.__doc__

        cls = better_dataclass(eq=False, repr=False, init=has_dataclass_init)(cls)
        for f in dataclasses.fields(cls):  # pyright: ignore[reportArgumentType]
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
        if has_dataclass_init:
            cls.__init__.__doc__ = init_doc  # pyright: ignore[reportPossiblyUnboundVariable]

        flattener = _ModuleFlattener(dataclasses.fields(cls))  # pyright: ignore[reportArgumentType]
        jtu.register_pytree_with_keys(
            cls,
            flatten_with_keys=flattener.flatten_with_keys,  # pyright: ignore
            flatten_func=flattener.flatten,  # pyright: ignore
            unflatten_func=ft.partial(flattener.unflatten_with_cls, cls),  # pyright: ignore
        )

        return cls

    def __call__(cls, *args, **kwargs):  # noqa: N805
        __tracebackhide__ = True
        if cls in _abstract_module_registry:
            # Any other is-abstract checks will be handled in super().__call__.
            raise TypeError("Cannot instantiate abstract `equinox.Module`.")
        if _has_dataclass_init[cls]:
            for x in jtu.tree_leaves((args, kwargs)):
                _warn_jax_transformed_function(cls, x)

        self = None
        try:
            self = super().__call__(*args, **kwargs)  # pyright: ignore[reportAttributeAccessIssue]
        finally:
            if self is not None:
                _currently_initialising.remove(self)
        assert not is_abstract_module(cls)  # pyright: ignore[reportArgumentType]

        fields = dataclasses.fields(cls)  # pyright: ignore[reportArgumentType]
        # Not `vars` or `__dict__`, to allow for `property`s overwriting a field.
        # Not recommended, but allowable for backward compatibility.
        dir_self = dir(self)
        missing_names = {f.name for f in fields if f.name not in dir_self}
        if len(missing_names) > 0:
            raise TypeError(
                f"The following fields were not initialised during __init__: "
                f"{missing_names}"
            )

        for f in fields:
            if (converter := f.metadata.get("converter")) is not None:
                object.__setattr__(self, f.name, converter(getattr(self, f.name)))
            if f.metadata.get("static", False):
                if any(jtu.tree_map(is_array, jtu.tree_leaves(getattr(self, f.name)))):
                    warnings.warn(
                        "A JAX array is being set as static! This can result "
                        "in unexpected behavior and is usually a mistake to do.",
                        stacklevel=2,
                    )
            if not f.init:
                if any(
                    jtu.tree_map(
                        is_inexact_array_like, jtu.tree_leaves(getattr(self, f.name))
                    )
                ):
                    warnings.warn(
                        "Using `field(init=False)` on `equinox.Module` can lead to "
                        "surprising behaviour when used around `jax.grad`. In the "
                        "following example, observe how JAX computes gradients with "
                        "respect to the `.len` attribute (which is a PyTree leaf "
                        "passed across the `jax.grad` boundary) and that there are no "
                        "gradients with respect to `.a` or `.b`:\n"
                        "\n"
                        "```\n"
                        "import equinox as eqx\n"
                        "import jax\n"
                        "import jax.numpy as jnp\n"
                        "\n"
                        "class Foo(eqx.Module):\n"
                        "    a: jax.Array\n"
                        "    b: jax.Array\n"
                        "    len: jax.Array = eqx.field(init=False)\n"
                        "\n"
                        "    def __post_init__(self):\n"
                        "        self.len = jnp.sqrt(self.a**2 + self.b**2)\n"
                        "\n"
                        "    def __call__(self, x):\n"
                        "        return self.len * x\n"
                        "\n"
                        "@jax.jit\n"
                        "@jax.grad\n"
                        "def call(module, x):\n"
                        "    return module(x)\n"
                        "\n"
                        "grads = call(Foo(jnp.array(3.0), jnp.array(4.0)), 5)\n"
                        "# Foo(\n"
                        "#   a=Array(0., dtype=float32, weak_type=True),\n"
                        "#   b=Array(0., dtype=float32, weak_type=True),\n"
                        "#   len=Array(5., dtype=float32, weak_type=True)\n"
                        "# )\n"
                        "```",
                        stacklevel=2,
                    )

        for parent_cls in cls.__mro__:
            try:
                check_init = parent_cls.__dict__["__check_init__"]
            except KeyError:  # noqa: PERF203
                pass
            else:
                check_init(self)

        return self

    # Ensure that `help(FooModule)` still works, even though we've overriden `__call__`.
    @property
    def __signature__(cls):  # noqa: N805
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.values())[1:]  # Remove self parameter
        return sig.replace(parameters=params)


class Module(Hashable, metaclass=_ModuleMeta):
    def __new__(cls, *args, **kwargs):
        del args, kwargs
        self = super().__new__(cls)
        # We record currently-initialising modules
        _currently_initialising.add(self)
        return self

    def __repr__(self):
        return tree_pformat(self)

    def __hash__(self) -> int:
        return hash(
            tuple(
                (f.name, getattr(self, f.name)) for f in dataclasses.fields(type(self))
            )
        )

    def __eq__(self, other) -> bool | np.bool_ | Bool[Array, ""]:  # pyright: ignore
        return tree_equal(self, other)

    def __setattr__(self, name: str, value: Any):
        if self in _currently_initialising:
            allowed_names = frozenset({f.name for f in dataclasses.fields(self)}).union(
                wrapper_field_names
            )
            if name in allowed_names:
                _error_method_assignment(self, value)
                _warn_jax_transformed_function(type(self), value)
                object.__setattr__(self, name, value)
                return
        # Allow:
        # ```
        # class SomeModule(eqx.Module, Generic[T]): ...
        # x = SomeModule[int]()
        # x.__orig_class__ # SomeModule[int]
        # ```
        # This attribute is set after instantiation here:
        # https://github.com/python/cpython/blob/7b3ab5921fa25ed8b97b6296f97c5c78aacf5447/Lib/typing.py#L728
        # So without special-casing it's incompatible with frozen dataclasses.
        if name == "__orig_class__":
            object.__setattr__(self, name, value)
        raise dataclasses.FrozenInstanceError(f"cannot assign to field '{name}'")

    def __getattribute__(self, name: str, /) -> Any:
        out = super().__getattribute__(name)
        # Arrange for bound methods to be treated as PyTrees as well. This
        # ensures that
        # ```
        # @jax.jit
        # def run(fn):
        #     ...
        # run(SomeModule().some_method)
        # ```
        # works.
        if (
            not _is_magic(name)
            and isinstance(out, types.MethodType)
            and out.__self__ is self
        ):
            out = BoundMethod(object.__getattribute__(out, "__func__"), self)
        return out


def _is_magic(k: str) -> bool:
    return (k.startswith("__") and k.endswith("__")) or (k == "_abc_impl")


def is_abstract_module(cls: type[Module]) -> bool:
    if not issubclass(cls, Module):
        raise TypeError(f"{cls} is not a subclass of `Module`.")
    return (
        (len(cls.__abstractmethods__) > 0)
        or (len(cls.__abstractvars__) > 0)
        or (len(cls.__abstractclassvars__) > 0)
        or (cls in _abstract_module_registry)
    )


_P = ParamSpec("_P")
_T = TypeVar("_T")


def module_update_wrapper(
    wrapper: Module, wrapped: Callable[_P, _T] | None = None
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
    cls = wrapper.__class__
    if not isinstance(getattr(cls, "__wrapped__", None), property):
        raise ValueError("Wrapper module must supply `__wrapped__` as a property.")

    if wrapped is None:
        wrapped = wrapper.__wrapped__  # pyright: ignore

    # Make a clone, to avoid mutating the original input.
    leaves, treedef = jtu.tree_flatten(wrapper)
    wrapper = jtu.tree_unflatten(treedef, leaves)

    # Like `ft.update_wrapper(wrapper, wrapped, updated=())`.
    # We don't update __dict__ as it's common/possible for wrapper and wrapped to
    # both be classes implementing __call__, in which case copying __dict__ over
    # basically just breaks the wrapper class.
    # We don't set __wrapped__, and instead demand that the wrapper class tell us
    # how to redirect to the wrapped object. This is avoid duplicating part of the
    # PyTree.
    _currently_initialising.add(wrapper)
    try:
        for field_name in wrapper_field_names:
            try:
                value = getattr(wrapped, field_name)
            except AttributeError:
                pass
            else:
                setattr(wrapper, field_name, value)
    finally:
        _currently_initialising.remove(wrapper)
    return cast(Callable[_P, _T], wrapper)


from ._prebuilt import BoundMethod  # After Module is defined.
