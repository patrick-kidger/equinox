import dataclasses
import functools as ft
import inspect
import textwrap
import types
import warnings
import weakref
from collections.abc import Callable, Hashable
from typing import Any, cast, Final, Literal, ParamSpec, TYPE_CHECKING, TypeVar
from typing_extensions import dataclass_transform

import jax
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, Bool

from .._filters import is_array, is_array_like, is_inexact_array_like
from .._pretty_print import tree_pformat
from .._tree import tree_equal
from ._better_abstract import better_dataclass, BetterABCMeta
from ._field import field


# Legacy compatibility API, passed to `strict` below.
def StrictConfig(
    force_abstact: bool = False, **kwargs: object
) -> Literal[False] | None:
    del kwargs
    return None if force_abstact else False


WRAPPER_FIELD_NAMES: Final = (
    "__module__",
    "__name__",
    "__qualname__",
    "__doc__",
    "__annotations__",
)


_abstract_module_registry = weakref.WeakSet()
_has_dataclass_init = weakref.WeakKeyDictionary()
_module_info = weakref.WeakKeyDictionary()
_MISSING = object()


def _make_tuple_type(count: int, element_type: str = "Any") -> str:
    """Generate a tuple type annotation string for a given count of elements."""
    if count == 0:
        return "tuple[()]"
    elif count == 1:
        return f"tuple[{element_type}]"
    else:
        return f"tuple[{', '.join([element_type] * count)}]"


INDENT: Final = 4

FIELDS_INFO = f"""
    Dynamic fields: {{dynamic}}
    Static fields: {{static}}
    Wrapper fields: {WRAPPER_FIELD_NAMES}
"""[1:-1]  # (trim leading and trailing newlines)


FLATTEN_CODE_BASE = '''
def flatten(obj: module_cls) -> {return_annotation}:
    """Generated flatten function for {qualname}.
    
    {fields_info}
    """
    return {dynamic_vals}, {aux}
'''


FLATTEN_WITH_KEYS_CODE_BASE = '''
def flatten_with_keys(obj: module_cls) -> {return_annotation}:
    """Generated flatten_with_keys function for {qualname}.
    
    {fields_info}
    """
    return {key_tuple}, {aux}
'''

UNFLATTEN_FUNC_BASE = '''
def unflatten(
    module_cls: type[T],
    aux_data: {aux_type},
    children: {dynamic_type},
) -> T:
    """Generated unflatten function for {qualname}.

    {fields_info}
    """
    self = object.__new__(module_cls)
    {setters}
    return self
'''

SET_DYNAMIC_BASE = """
object.__setattr__(self, {name!r}, children[{i}])
"""[1:-1]  # (trim leading and trailing newlines)

SET_AUX_BASE = """
object.__setattr__(self, {name!r}, aux_data[{i}])
"""[1:-1]  # (trim leading and trailing newlines)

SET_WRAPPER_BASE = """
if aux_data[{i}] is not _MISSING:
    object.__setattr__(self, {name!r}, aux_data[{i}])
"""[1:-1]

SET_WRAPPER_LINES = "\n".join(
    SET_WRAPPER_BASE.format(i=i, name=k) for i, k in enumerate(WRAPPER_FIELD_NAMES)
)


def _generate_flatten_functions(cls: type, fields: tuple[dataclasses.Field[Any], ...]):
    """Generate optimized flatten/unflatten functions for a specific field config."""
    # Separate dynamic and static fields
    dynamic_fs_, static_fs_ = [], []
    for f in fields:
        if f.metadata.get("static", False):
            static_fs_.append(f.name)
        else:
            dynamic_fs_.append(f.name)
    dynamic_fs, static_fs = tuple(dynamic_fs_), tuple(static_fs_)
    # aux_fs = WRAPPER_FIELD_NAMES + static_fs

    # Build field info for docs
    fields_info = FIELDS_INFO.format(dynamic=dynamic_fs, static=static_fs)[INDENT:]

    # -------------------------------------------
    # Generate flatten function

    # Directly access dynamic fields by name
    if dynamic_fs:
        dynamic_exprs = [f"obj.{name}" for name in dynamic_fs]
        dynamic_vals = f"({', '.join(dynamic_exprs)},)"
    else:
        dynamic_vals = "()"

    # For static fields, we need to store their values in aux data
    if static_fs:
        static_exprs = [f"obj.{name}" for name in static_fs]
        static_aux = f"({', '.join(static_exprs)},)"
    else:
        static_aux = "()"

    # Build return type annotation
    dynamic_type = _make_tuple_type(len(dynamic_fs))
    static_type = _make_tuple_type(len(static_fs))
    qualname = cls.__qualname__

    flatten_code = FLATTEN_CODE_BASE.format(
        return_annotation=f"tuple[{dynamic_type}, {static_type}]",
        qualname=qualname,
        fields_info=fields_info,
        dynamic_vals=dynamic_vals,
        aux=static_aux,
    )

    # -------------------------------------------
    # Generate flatten_with_keys function

    key_exprs = [f"(jtu.GetAttrKey({name!r}), obj.{name})" for name in dynamic_fs]
    key_tuple = f"({', '.join(key_exprs)},)" if dynamic_fs else "()"

    keys_dynamic_type = _make_tuple_type(len(dynamic_fs), "tuple[Any, Any]")

    flatten_with_keys_code = FLATTEN_WITH_KEYS_CODE_BASE.format(
        return_annotation=f"tuple[{keys_dynamic_type}, {static_type}]",
        qualname=qualname,
        fields_info=fields_info,
        key_tuple=key_tuple,
        aux=static_aux,
    )

    # -------------------------------------------
    # Generate unflatten function - directly set fields by index
    # Extract types from flatten return type: tuple[dynamic_type, static_type]

    unflatten_lines: list[str] = []
    if dynamic_fs or static_fs:
        unflatten_lines.append("# Set dynamic fields directly by index")
    # Set dynamic fields directly by index
    unflatten_lines.extend(
        SET_DYNAMIC_BASE.format(i=i, name=k) for i, k in enumerate(dynamic_fs)
    )
    # Set wrapper fields from aux_data
    # unflatten_lines.append(SET_WRAPPER_LINES)
    # Set static fields from aux_data
    unflatten_lines.extend(
        SET_AUX_BASE.format(i=i, name=k)
        # for i, k in enumerate(static_fs, start=len(WRAPPER_FIELD_NAMES))
        for i, k in enumerate(static_fs, start=0)
    )

    unflatten_code = UNFLATTEN_FUNC_BASE.format(
        aux_type=static_type,
        dynamic_type=dynamic_type,
        qualname=qualname,
        fields_info=fields_info,
        setters=textwrap.indent("\n".join(unflatten_lines), " " * INDENT)[INDENT:],
    )

    # -------------------------------------------
    # Compile all functions

    # Namespace for flatten functions (they need module_cls)
    flatten_ns = {
        "jtu": jtu,
        "object": object,
        "module_cls": cls,
        "Any": Any,
        "tuple": tuple,
        "_MISSING": _MISSING,
    }

    # Namespace for unflatten function (takes module_cls as parameter)
    unflatten_ns = {
        "object": object,
        "Any": Any,
        "tuple": tuple,
        "type": type,
        "T": TypeVar("T"),
        "_MISSING": _MISSING,
    }

    # Use class-specific filenames for better __code__ introspection
    exec(compile(flatten_code, f"<generated_flatten_{qualname}>", "exec"), flatten_ns)
    exec(
        compile(
            flatten_with_keys_code, f"<generated_flatten_with_keys_{qualname}>", "exec"
        ),
        flatten_ns,
    )
    exec(
        compile(unflatten_code, f"<generated_unflatten_{qualname}>", "exec"),
        unflatten_ns,
    )

    # Extract the generated functions from respective namespaces to
    # set the proper module reference.
    module_name = getattr(cls, "__module__", "equinox._module._module")
    flatten_func = flatten_ns["flatten"]
    flatten_func.__module__ = module_name
    flatten_with_keys_func = flatten_ns["flatten_with_keys"]
    flatten_with_keys_func.__module__ = module_name
    unflatten_func = unflatten_ns["unflatten"]
    unflatten_func.__module__ = module_name

    return flatten_func, flatten_with_keys_func, unflatten_func


MSG_METHOD_IN_INIT: Final = """Cannot assign methods in __init__.

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


def _error_method_assignment(self, value: object, /) -> None:
    if isinstance(value, BoundMethod) and value.__self__ is self:
        raise ValueError(MSG_METHOD_IN_INIT)


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


def _is_array_like(x: object, /) -> None:
    if is_array_like(x):
        raise _JaxTransformException


MSG_JAX_XFM_FUNC: Final = """
Possibly assigning a JAX-transformed callable as an attribute on
{0}.{1}. This will not have any of its parameters updated.

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
"""


def _warn_jax_transformed_function(cls: "_ModuleMeta", x: object) -> None:
    # not `isinstance`, just in case JAX every tries to override `__instancecheck__`.
    if type(x) in _transform_types:
        while True:
            try:
                x = getattr(x, "__wrapped__")
            except AttributeError:
                break
            try:
                jtu.tree_map(_is_array_like, x)
            except _JaxTransformException:
                warnings.warn(
                    MSG_JAX_XFM_FUNC.format(cls.__module__, cls.__qualname__),
                    stacklevel=3,
                )
                break


class _IdSet:
    __slots__ = ("_dict",)

    def __init__(self):
        self._dict: dict[int, Module] = {}

    def __contains__(self, key: "Module") -> bool:
        return id(key) in self._dict.keys()

    def add(self, key: "Module") -> None:
        if key not in self:
            id_key = id(key)
            # Hold on to `key` to be sure that `id(key)` does not get reallocated.
            self._dict[id_key] = key

    def remove(self, key: "Module") -> None:
        del self._dict[id(key)]


_currently_initialising = _IdSet()


_MSG_CUSTOM_INIT_AND_POST_INIT = """
Class `{cls.__module__}.{cls.__qualname__}` has both an `__init__` method and a
`__post_init__` method. This means that the `__post_init__` method will not be
run!

The reason for this is that `__post_init__` is intended to be used with the
automatically-generated `__init__` method provided by Python dataclasses, which
are generated of the form:

```
def __init__(self, field1, field2)
    self.field1 = field1
    self.field2 = field2
    self.__post_init__()
```

and as such a user-provided `__init__` overrides both the setting of fields, and
the calling of `__post_init__`.

The above is how Python dataclasses work, and has nothing to do with Equinox!

If you are using `__post_init__` to check that certain invariants hold, then
consider using `__check_init__` instead. This is an Equinox-specific extension
that is always ran. See here for more details:
https://docs.kidger.site/equinox/api/module/advanced_fields/#checking-invariants
"""[1:]


# This deliberately does not pass `frozen_default=True`, as that clashes with custom
# `__init__` methods.
@dataclass_transform(field_specifiers=(dataclasses.field, field))
class _ModuleMeta(BetterABCMeta):
    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, object],
        *,
        is_abstract: bool = False,
        strict: None | bool = False,
        **kwargs: object,
    ) -> type["_ModuleMeta"]:
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
            warnings.warn(_MSG_CUSTOM_INIT_AND_POST_INIT.format(cls=cls), stacklevel=2)
        if has_dataclass_init:
            init_doc = cls.__init__.__doc__

        cls = better_dataclass(eq=False, repr=False, init=has_dataclass_init)(cls)
        fields = dataclasses.fields(cls)  # pyright: ignore[reportArgumentType]
        for f in fields:  # pyright: ignore[reportArgumentType]
            if f.name not in cls.__init__.__annotations__:
                continue  # Odd behaviour, so skip.
            try:
                converter = f.metadata["converter"]
            except KeyError:
                pass
            else:
                try:
                    sig = inspect.signature(converter)
                except ValueError:
                    # e.g. `inspect.signature(str)` fails
                    converter_annotation = Any
                else:
                    parameters = list(sig.parameters.values())
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

        # Cache the field names for later use.
        _module_info[cls] = frozenset(f.name for f in fields)

        # Generate optimized flatten/unflatten functions
        flatten_func, flatten_with_keys_func, unflatten_func = (
            _generate_flatten_functions(cls, fields)  # pyright: ignore[reportArgumentType]
        )

        jtu.register_pytree_with_keys(
            cls,
            flatten_with_keys=flatten_with_keys_func,  # pyright: ignore
            flatten_func=flatten_func,  # pyright: ignore
            unflatten_func=ft.partial(unflatten_func, cls),  # pyright: ignore
        )

        return cls

    def __call__(cls, *args: object, **kwargs: object):  # noqa: N805
        __tracebackhide__ = True
        if cls in _abstract_module_registry:
            # Any other is-abstract checks will be handled in super().__call__.
            raise TypeError("Cannot instantiate abstract `equinox.Module`.")
        if _has_dataclass_init[cls]:
            for x in jtu.tree_leaves((args, kwargs)):
                _warn_jax_transformed_function(cls, x)

        tryself = None
        try:
            self = tryself = super().__call__(*args, **kwargs)  # pyright: ignore[reportAttributeAccessIssue]
        finally:
            if tryself is not None:
                _currently_initialising.remove(tryself)
            del tryself
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

    def __new__(cls, *args: object, **kwargs: object) -> "Module":
        del args, kwargs
        self = super().__new__(cls)
        # We record currently-initialising modules
        _currently_initialising.add(self)
        return self

    def __repr__(self) -> str:
        return tree_pformat(self)

    def __hash__(self) -> int:
        return hash(
            tuple(
                (f.name, getattr(self, f.name)) for f in dataclasses.fields(type(self))
            )
        )

    def __eq__(self, other: object, /) -> bool | np.bool_ | Bool[Array, ""]:  # pyright: ignore
        return tree_equal(self, other)

    if not TYPE_CHECKING:

        def __setattr__(self, name: str, value: Any) -> None:
            if self in _currently_initialising and (
                name in _module_info[type(self)] or name in WRAPPER_FIELD_NAMES
            ):
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


def _is_magic(k: str, /) -> bool:
    return (k.startswith("__") and k.endswith("__")) or (k == "_abc_impl")


def is_abstract_module(cls: type[Module], /) -> bool:
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
        for field_name in WRAPPER_FIELD_NAMES:
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
