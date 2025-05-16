from collections.abc import Callable
from typing import cast, ParamSpec, TypeVar

import jax.tree_util as jtu

from ._module import Module, wrapper_field_names


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
    for field_name in wrapper_field_names:
        try:
            value = getattr(wrapped, field_name)
        except AttributeError:
            pass
        else:
            setattr(wrapper, field_name, value)
    return cast(Callable[_P, _T], wrapper)
