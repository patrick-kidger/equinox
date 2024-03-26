import operator
from collections.abc import Callable
from typing import Any, Optional, TYPE_CHECKING

import jax
import jax.tree_util as jtu


class _Metaω(type):
    def __rpow__(cls, value):
        return cls(value)


class _ω(metaclass=_Metaω):
    """Provides friendlier syntax for mapping with `jax.tree_util.tree_map`.

    !!! example

        ```python
        (ω(a) + ω(b)).ω == jax.tree_util.tree_map(operator.add, a, b)
        ```

    !!! tip

        To minimise the number of brackets, the following `__rpow__` syntax can be
        used:

        ```python
        (a**ω + b**ω).ω == jax.tree_util.tree_map(operator.add, a, b)
        ```

        This is entirely equivalent to the above.
    """

    def __init__(self, value, is_leaf=None):
        """
        **Arguments:**

        - `value`: The PyTree to wrap.
        - `is_leaf`: An optional value for the `is_leaf` argument to
          `jax.tree_util.tree_map`.

        !!! note

            The `is_leaf` argument cannot be set when using the `__rpow__` syntax for
            initialisation.
        """
        self.ω = value
        self.is_leaf = is_leaf

    def __repr__(self):
        return f"ω({self.ω})"

    def __getitem__(self, item):
        return ω(
            jtu.tree_map(lambda x: x[item], self.ω, is_leaf=self.is_leaf),
            is_leaf=self.is_leaf,
        )

    def call(self, fn):
        return ω(
            jtu.tree_map(fn, self.ω, is_leaf=self.is_leaf),
            is_leaf=self.is_leaf,
        )

    @property
    def at(self):
        return _ωUpdateHelper(self.ω, self.is_leaf)


if TYPE_CHECKING:
    ω: Any = ...
else:
    ω = _ω


def _equal_code(fn1: Optional[Callable], fn2: Optional[Callable]):
    """Checks whether fn1 and fn2 both have the same code.

    It's essentially impossible to see if two functions are equivalent, so this won't,
    and isn't intended, to catch every possible difference between fn1 and fn2. But it
    should at least catch the common case that `is_leaf` is specified for one input and
    not specified for the other.
    """
    sentinel1 = object()
    sentinel2 = object()
    code1 = getattr(getattr(fn1, "__code__", sentinel1), "co_code", sentinel2)
    code2 = getattr(getattr(fn2, "__code__", sentinel1), "co_code", sentinel2)
    return type(code1) == type(code2) and code1 == code2


def _set_binary(base, name: str, op: Callable[[Any, Any], Any]) -> None:
    def fn(self, other):
        if isinstance(other, ω):
            if jtu.tree_structure(self.ω) != jtu.tree_structure(other.ω):
                raise ValueError("PyTree structures must match.")
            if not _equal_code(self.is_leaf, other.is_leaf):
                raise ValueError("`is_leaf` must match.")
            return ω(
                jtu.tree_map(op, self.ω, other.ω, is_leaf=self.is_leaf),
                is_leaf=self.is_leaf,
            )
        elif isinstance(other, (bool, complex, float, int, jax.Array)):
            return ω(
                jtu.tree_map(lambda x: op(x, other), self.ω, is_leaf=self.is_leaf),
                is_leaf=self.is_leaf,
            )
        else:
            raise RuntimeError("Type of `other` not understood.")

    fn.__name__ = name
    fn.__qualname__ = base.__qualname__ + "." + name
    setattr(base, name, fn)


def _set_unary(base, name: str, op: Callable[[Any], Any]) -> None:
    def fn(self):
        return ω(
            jtu.tree_map(op, self.ω, is_leaf=self.is_leaf),
            is_leaf=self.is_leaf,
        )

    fn.__name__ = name
    fn.__qualname__ = base.__qualname__ + "." + name
    setattr(base, name, fn)


def _rev(op):
    def __rev(x, y):
        return op(y, x)

    return __rev


for name, op in [
    ("__add__", operator.add),
    ("__sub__", operator.sub),
    ("__mul__", operator.mul),
    ("__matmul__", operator.matmul),
    ("__truediv__", operator.truediv),
    ("__floordiv__", operator.floordiv),
    ("__mod__", operator.mod),
    ("__pow__", operator.pow),
    ("__lshift__", operator.lshift),
    ("__rshift__", operator.rshift),
    ("__and__", operator.and_),
    ("__xor__", operator.xor),
    ("__or__", operator.or_),
    ("__radd__", _rev(operator.add)),
    ("__rsub__", _rev(operator.sub)),
    ("__rmul__", _rev(operator.mul)),
    ("__rmatmul__", _rev(operator.matmul)),
    ("__rtruediv__", _rev(operator.truediv)),
    ("__rfloordiv__", _rev(operator.floordiv)),
    ("__rmod__", _rev(operator.mod)),
    ("__rpow__", _rev(operator.pow)),
    ("__rlshift__", _rev(operator.lshift)),
    ("__rrshift__", _rev(operator.rshift)),
    ("__rand__", _rev(operator.and_)),
    ("__rxor__", _rev(operator.xor)),
    ("__ror__", _rev(operator.or_)),
    ("__lt__", operator.lt),
    ("__le__", operator.le),
    ("__eq__", operator.eq),
    ("__ne__", operator.ne),
    ("__gt__", operator.gt),
    ("__ge__", operator.ge),
]:
    _set_binary(ω, name, op)


for name, op in [
    ("__neg__", operator.neg),
    ("__pos__", operator.pos),
    ("__abs__", operator.abs),
    ("__invert__", operator.invert),
]:
    _set_unary(ω, name, op)


class _ωUpdateHelper:
    def __init__(self, value, is_leaf):
        self.value = value
        self.is_leaf = is_leaf

    def __getitem__(self, item):
        return _ωUpdateRef(self.value, item, self.is_leaf)


class _ωUpdateRef:
    def __init__(self, value, item, is_leaf):
        self.value = value
        self.item = item
        self.is_leaf = is_leaf


def _set_binary_at(base, name: str, op: Callable[[Any, Any, Any], Any]) -> None:
    def fn(self, other):
        if isinstance(other, ω):
            if jtu.tree_structure(self.value) != jtu.tree_structure(other.ω):
                raise ValueError("PyTree structures must match.")
            if not _equal_code(self.is_leaf, other.is_leaf):
                raise ValueError("is_leaf specifications must match.")
            return ω(
                jtu.tree_map(
                    lambda x, y: op(x, self.item, y),
                    self.value,
                    other.ω,
                    is_leaf=self.is_leaf,
                ),
                is_leaf=self.is_leaf,
            )
        elif isinstance(other, (bool, complex, float, int, jax.Array)):
            return ω(
                jtu.tree_map(
                    lambda x: op(x, self.item, other), self.value, is_leaf=self.is_leaf
                ),
                is_leaf=self.is_leaf,
            )
        else:
            raise RuntimeError("Type of `other` not understood.")

    fn.__name__ = name
    fn.__qualname__ = base.__qualname__ + "." + name
    setattr(base, name, fn)


for name, op in [
    ("set", lambda x, y, z, **kwargs: x.at[y].set(z, **kwargs)),
    ("add", lambda x, y, z, **kwargs: x.at[y].add(z, **kwargs)),
    ("multiply", lambda x, y, z, **kwargs: x.at[y].multiply(z, **kwargs)),
    ("divide", lambda x, y, z, **kwargs: x.at[y].divide(z, **kwargs)),
    ("power", lambda x, y, z, **kwargs: x.at[y].power(z, **kwargs)),
    ("min", lambda x, y, z, **kwargs: x.at[y].min(z, **kwargs)),
    ("max", lambda x, y, z, **kwargs: x.at[y].max(z, **kwargs)),
]:
    _set_binary_at(_ωUpdateRef, name, op)
