import dataclasses
from collections.abc import Callable
from typing import Any, Generic, Protocol, runtime_checkable, TypeVar
from typing_extensions import dataclass_transform

import jax.tree_util as jtu
from jaxtyping import PyTreeDef

from ._field import field
from ._flatten import WRAPPER_FIELD_NAMES
from ._module import (
    _abstract_module_registry,
    _currently_initialising,
    _module_info,
    _ModuleMeta,
    Module,
)


_Return = TypeVar("_Return")
_Return_co = TypeVar("_Return_co", covariant=True)


@dataclass_transform(field_specifiers=(dataclasses.field, field))
class _FastModuleMeta(_ModuleMeta):
    """Metaclass for internal Modules created often enough that the
    instantiation-time checks in `_ModuleMeta.__call__` are worth skipping.

    Opting in is deliberately not inherited: a subclass of a class using this
    metaclass goes through the usual `_ModuleMeta.__call__`, checks and all.
    """

    # Set on every class using this metaclass; True only for those opting in.
    __fast_init__: bool

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, object],
        **kwargs: Any,
    ):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        # Only the class that explicitly asks for this metaclass gets the
        # fastpath; anything inheriting from it does not.
        cls.__fast_init__ = fast = not any(  # pyright: ignore[reportAttributeAccessIssue]
            isinstance(b, _FastModuleMeta) for b in bases
        )
        if fast:
            # These are the things the fastpath skips that would otherwise fail
            # silently. Checked once, at class-creation time, so that a future
            # edit to an opted-in class cannot quietly lose them.
            info = _module_info[cls]
            if (
                info.converter_fields
                or info.check_init_methods
                or info.non_init_field_names
            ):
                raise TypeError(
                    "`_FastModuleMeta` does not support converters, "
                    "`__check_init__`, or `init=False` fields."
                )
            # The fastpath does not consult `_abstract_module_registry`, so an
            # abstract class would silently become instantiable.
            if cls in _abstract_module_registry:
                raise TypeError("`_FastModuleMeta` classes cannot be abstract.")
        return cls

    def __call__(cls, *args: object, **kwargs: object):  # noqa: N805
        __tracebackhide__ = True
        if not cls.__fast_init__:
            return super().__call__(*args, **kwargs)
        tryself = None
        try:
            # Deliberately skipping `_ModuleMeta.__call__`.
            self = tryself = super(_ModuleMeta, cls).__call__(*args, **kwargs)
        finally:
            if tryself is not None:
                _currently_initialising.remove(tryself)
            del tryself
        return self


@runtime_checkable
class _FuncDescriptor(Protocol[_Return_co]):
    """A callable that also supports the descriptor protocol (has ``__get__``)."""

    def __call__(self, __self: Any, /, *args: Any, **kwargs: Any) -> _Return_co: ...

    def __get__(self, obj: Any, objtype: Any = None) -> Callable[..., _Return_co]: ...


# Not using `jax.tree_util.Partial` as it doesn't implement __eq__ very well. See #480.
class BoundMethod(Module, Generic[_Return], metaclass=_FastModuleMeta):
    """Just like a normal Python bound method... except that this one is a PyTree!

    This stores `__self__` as a subnode.
    """

    __func__: _FuncDescriptor[_Return] = field(static=True)
    __self__: Module

    def __call__(self, *args: Any, **kwargs: Any) -> _Return:
        __tracebackhide__ = True
        return self.__func__(self.__self__, *args, **kwargs)

    @property
    def __wrapped__(self) -> Callable[..., _Return]:
        return self.__func__.__get__(self.__self__, type(self.__self__))

    def __getattr__(self, name: str) -> Any:
        if name in WRAPPER_FIELD_NAMES:
            return getattr(self.__func__, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


class Partial(Module, Generic[_Return]):
    """Like `functools.partial`, but treats the wrapped function, and partially-applied
    args and kwargs, as a PyTree.

    This is very much like `jax.tree_util.Partial`. The difference is that the JAX
    version requires that `func` be specifically a *function* -- and will silently
    misbehave if given any non-function callable, e.g. [`equinox.nn.MLP`][]. In contrast
    the Equinox version allows for arbitrary callables.
    """

    func: Callable[..., _Return]
    args: tuple[Any, ...]
    keywords: dict[str, Any]

    def __init__(self, func: Callable[..., _Return], /, *args: Any, **kwargs: Any):
        """**Arguments:**

        - `func`: the callable to partially apply.
        - `*args`: any positional arguments to apply.
        - `**kwargs`: any keyword arguments to apply.
        """
        self.func = func
        self.args = args
        self.keywords = kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> _Return:
        """Call the wrapped `self.func`.

        **Arguments:**

        - `*args`: the arguments to apply. Passed after those arguments passed during
            `__init__`.
        - `**kwargs`: any keyword arguments to apply.

        **Returns:**

        The result of the wrapped function.
        """
        return self.func(*self.args, *args, **kwargs, **self.keywords)


_Value = TypeVar("_Value")


class Static(Module, Generic[_Value]):
    """Wraps a value into a `eqx.field(static=True)`.

    This is useful to treat something as just static metadata with respect to a JAX
    transformation; for example this is used to return non-arrays from a filtered
    transform.
    """

    _leaves: list[Any] = field(static=True)
    _treedef: PyTreeDef = field(static=True)  # pyright: ignore

    def __init__(self, value: _Value):
        # By flattening, we handle pytrees without `__eq__` methods.
        # When comparing static metadata for equality, this means we never actually
        # call `value.__eq__`.
        self._leaves, self._treedef = jtu.tree_flatten(value)

    @property
    def value(self) -> _Value:
        return jtu.tree_unflatten(self._treedef, self._leaves)
