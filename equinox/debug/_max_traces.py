import functools as ft
import inspect
import weakref
from collections.abc import Callable
from typing import Optional, overload, TypeVar
from typing_extensions import ParamSpec

from .._custom_types import sentinel
from .._eval_shape import filter_eval_shape
from .._module import field, Module, module_update_wrapper
from .._pretty_print import tree_pformat
from .._tree import tree_equal


_T = TypeVar("_T")
_P = ParamSpec("_P")


# If we wanted we could actually store this directly on the `_AssertMaxTraces` object.
# Replace `tag` with a list of a single integer, `[0]`, and propagate the list as static
# metadata so that even after flattening and unflattening they share a single counter.
#
# In practice we do it this way to make it abundantly clear that this counter is shared
# across cloned (flattened+unflattened) instances.
_traces = weakref.WeakKeyDictionary()
_seen_args = weakref.WeakKeyDictionary()


class _Weakrefable:
    __slots__ = ("__weakref__",)


class _AssertMaxTraces(Module):
    fn: Callable
    max_traces: Optional[int] = field(static=True)
    tag: _Weakrefable = field(static=True)

    def __init__(self, fn, max_traces):
        self.fn = fn
        self.max_traces = max_traces
        self.tag = _Weakrefable()
        _traces[self.tag] = 0
        _seen_args[self.tag] = {}

    @property
    def __wrapped__(self):
        return self.fn

    def __call__(self, *args, **kwargs):
        __tracebackhide__ = True
        num_traces = _traces[self.tag] = _traces[self.tag] + 1
        arguments = inspect.signature(self.fn).bind(*args, **kwargs).arguments
        if self.max_traces is not None and num_traces > self.max_traces:
            for name, value in arguments.items():
                struct = filter_eval_shape(lambda: value)
                seen_structs = _seen_args[self.tag].get(name, [])
                for seen_struct in seen_structs:
                    if tree_equal(struct, seen_struct) is True:
                        break
                else:
                    struct_str = tree_pformat(struct, struct_as_array=True)
                    seen_struct_str = "\n".join(
                        tree_pformat(seen_struct, struct_as_array=True)
                        for seen_struct in seen_structs
                    )
                    raise RuntimeError(
                        f"{self.fn} can only be traced {self.max_traces} times. "
                        f"However, it is has now been traced {num_traces} times. It "
                        f"appears that the '{name}' argument is responsible: it "
                        f"currently has the value:\n{struct_str}\nPrevious values are:"
                        f"\n{seen_struct_str}"
                    )
            raise RuntimeError(
                f"{self.fn} can only be traced {self.max_traces} times. However, "
                f"it is has now been traced {num_traces} times. Could not determine "
                "argument was responsible for re-tracing."
            )
        for name, value in arguments.items():
            struct = filter_eval_shape(lambda: value)
            if name not in _seen_args[self.tag]:
                _seen_args[self.tag][name] = []
            _seen_args[self.tag][name].append(struct)
        return self.fn(*args, **kwargs)


@overload
def assert_max_traces(
    fn: Callable[_P, _T], *, max_traces: Optional[int]
) -> Callable[_P, _T]: ...


@overload
def assert_max_traces(
    *,
    max_traces: Optional[int],
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...


def assert_max_traces(fn: Callable = sentinel, *, max_traces: Optional[int]):
    """Asserts that the wrapped callable is not called more than `max_traces` times.

    The typical use-case for this is to check that a JIT-compiled function is not
    compiled more than `max_traces` times. (I.e. this function can be used to guard
    against bugs.) In this case it should be placed within the JIT wrapper.

    !!! Example

        ```python
        @eqx.filter_jit
        @eqx.debug.assert_max_traces(max_traces=1)
        def f(x, y):
            return x + y
        ```

    **Arguments:**

    - `fn`: The callable to wrap.
    - `max_traces`: keyword only argument. The maximum number of calls that are
        allowed. Can be `None` to allow arbitrarily many calls; in this case the number
        of calls can still can be found via [`equinox.debug.get_num_traces`][].

    **Returns:**

    A wrapped version of `fn` that tracks the number of times it is called. This will
    raise a `RuntimeError` if is called more than `max_traces` many times.

    !!! info

        See also
        [`chex.assert_max_traces`](https://github.com/google-deepmind/chex#jax-tracing-assertions)
        which provides similar functionality.

        The differences are that (a) Chex's implementation is a bit stricter, as the
        following will raise:
        ```python
        import chex
        import jax

        def f(x):
            pass

        f2 = jax.jit(chex.assert_max_traces(f, 1))
        f3 = jax.jit(chex.assert_max_traces(f, 1))

        f2(1)
        f3(1)  # will raise, despite the fact that f2 and f3 are different.
        ```
        and (b) Equinox's implementation supports non-function callables.

        You may prefer the Chex version if you prefer the stricter raising behaviour of
        the above code.
    """
    if fn is sentinel:
        return ft.partial(assert_max_traces, max_traces=max_traces)
    return module_update_wrapper(_AssertMaxTraces(fn, max_traces))


def get_num_traces(fn) -> int:
    """Given a function wrapped in [`equinox.debug.assert_max_traces`][], return the
    number of times which it has been traced so far.
    """
    num_traces = _get_num_traces(fn)
    if num_traces is None:
        raise ValueError(
            f"{fn} does not appear to wrapped with `eqx.debug.assert_max_traces`"
        )
    return num_traces


def _get_num_traces(fn) -> Optional[int]:
    if isinstance(fn, _AssertMaxTraces):
        return _traces[fn.tag]
    elif hasattr(fn, "__wrapped__"):
        return _get_num_traces(fn.__wrapped__)
    else:
        return None
