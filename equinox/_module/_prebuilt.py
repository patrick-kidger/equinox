import types
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import jax.tree_util as jtu
from jaxtyping import PyTreeDef

from ._field import field
from ._module import Module, wrapper_field_names


# Not using `jax.tree_util.Partial` as it doesn't implement __eq__ very well. See #480.
class BoundMethod(Module):
    """Just like a normal Python bound method... except that this one is a PyTree!

    This stores `__self__` as a subnode.
    """

    __func__: types.FunctionType = field(static=True)
    __self__: Module

    def __post_init__(self):
        for field_name in wrapper_field_names:
            try:
                value = getattr(self.__func__, field_name)
            except AttributeError:
                pass
            else:
                setattr(self, field_name, value)

    def __call__(self, *args, **kwargs):
        __tracebackhide__ = True
        return self.__func__(self.__self__, *args, **kwargs)

    @property
    def __wrapped__(self):
        return self.__func__.__get__(self.__self__, type(self.__self__))  # pyright: ignore[reportAttributeAccessIssue]


_Return = TypeVar("_Return")


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
