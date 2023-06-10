"""JAX-compatible enums.

```python
class RESULTS(eqxi.Enumeration):
    success = "success"
    linear_solve_failed = "Linear solve failed to converge"
    diffeq_solve_failed = "Differential equation solve exploded horribly"

# Enums can be passed through JIT:
jax.jit(lambda x: x)(RESULTS.success)

# Enums support `.where` (on the enum class) analogous to `jnp.where`:
result = RESULTS.where(diff < tol, RESULTS.success, RESULTS.linear_solve_failed)
result = RESULTS.where(step < max_steps, result, RESULTS.diffeq_solve_failed)

# Enums support equality checking:
x = jnp.where(result == RESULTS.success, a, b)

# Enums cannot be compared against anything except an Enumeration of the same type:
result == 0  # ValueError at trace time

# Enums support runtime error messages via `.error_if` (on the enum item):
solution = ...
solution = result.error_if(solution, result != RESULTS.successful)

# Enums support (multiple) inheritance, and `.promote` (on the enum class):
# (Note that you will need add a `# pyright: ignore` wherever you inherit.)
class MORE_RESULTS(RESULTS):
    flux_capciter_overloaded = "Run for your life!"

result == RESULTS.success
result == MORE_RESULTS.success  # ValueError at trace time
result = MORE_RESULTS.promote(result)  # requires issubclass(MORE_RESULTS, RESULTS)
result == MORE_RESULTS.success  # now this works
```
"""

from typing import cast, TYPE_CHECKING, Union
from typing_extensions import Self

import jax.core
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Int

from .._module import field, Module
from ._errors import branched_error_if


def _magic(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _store(
    cls,
    name: str,
    message: str,
    name_to_item: dict[str, "EnumerationItem"],
    index_to_message: list[str],
) -> int:
    try:
        our_item = name_to_item[name]
    except KeyError:
        our_index = len(index_to_message)
        name_to_item[name] = EnumerationItem(np.array(our_index, dtype=np.int32), cls)
        index_to_message.append(message)
        return our_index
    else:
        our_index = our_item._value.item()
        existing_message = index_to_message[our_index]
        if message != existing_message:
            raise ValueError(
                f"Enumeration has duplicate incompatible values for {name}"
            )
        return our_index


class _EnumerationMeta(type):
    def __new__(mcs, cls_name, bases, namespace):
        assert "_name_to_item" not in namespace
        assert "_index_to_message" not in namespace
        assert "_base_offsets" not in namespace
        assert "promote" not in namespace
        assert "where" not in namespace
        name_to_item = {}
        index_to_message = []
        base_offsets = {}
        new_namespace = dict(
            _name_to_item=name_to_item,
            _index_to_message=index_to_message,
            _base_offsets=base_offsets,
        )
        for name, value in namespace.items():
            if _magic(name):
                new_namespace[name] = value
        cls = super().__new__(mcs, cls_name, bases, new_namespace)
        for original_base in bases:
            # skip Enumeration and object
            for base in original_base.__mro__[:-2]:
                if not issubclass(base, Enumeration):
                    raise ValueError(
                        "Cannot subclass enumerations with non-enumerations."
                    )
                if base not in base_offsets.keys():
                    offsets = np.zeros(len(base), dtype=np.int32)
                    for name, base_item in base._name_to_item.items():
                        base_index = base_item._value.item()
                        message = base._index_to_message[base_index]
                        our_index = _store(
                            cls, name, message, name_to_item, index_to_message
                        )
                        offsets[base_index] = our_index
                    base_offsets[base] = jnp.asarray(offsets)
        for name, message in namespace.items():
            if not _magic(name):
                _store(cls, name, message, name_to_item, index_to_message)
        return cls

    def __getattr__(cls, item: str):
        try:
            return cls._name_to_item[item]
        except KeyError as e:
            raise AttributeError from e

    def __getitem__(cls, item) -> str:
        if not isinstance(item, EnumerationItem):
            raise ValueError("Index must be an enumeration item.")
        if item._enumeration is not cls:
            raise ValueError("Enumeration item must be from this enumeration.")
        if isinstance(item._value, jax.core.Tracer):
            return "traced"
        elif isinstance(item._value, (np.ndarray, Array)):
            return cls._index_to_message[item._value.item()]
        else:
            # PyTrees have to be generic wrt leaf type.
            return "unknown"

    def __len__(cls):
        return len(cls._name_to_item)

    def __instancecheck__(cls, value):
        return isinstance(value, EnumerationItem) and value._enumeration is cls

    def promote(cls, item):
        if not isinstance(item, EnumerationItem):
            raise ValueError("Can only promote enumerations.")
        if (not issubclass(cls, item._enumeration)) or item._enumeration is cls:
            raise ValueError("Can only promote from inherited enumerations.")
        value = cls._base_offsets[item._enumeration][item._value]
        return EnumerationItem(value, cls)

    def where(cls, pred, a, b):
        if isinstance(a, EnumerationItem) and isinstance(b, EnumerationItem):
            if a._enumeration is cls and b._enumeration is cls:
                value = jnp.where(pred, a._value, b._value)
                cls = cast(type[Enumeration], cls)
                return EnumerationItem(value, cls)
        name = f"{cls.__module__}.{cls.__qualname__}"
        raise ValueError(f"Arguments to {name}.where(...) must be members of {name}.")


class EnumerationItem(Module):
    _value: Int[Union[Array, np.ndarray], ""]
    _enumeration: type["Enumeration"] = field(static=True)

    def __eq__(self, other) -> Bool[ArrayLike, ""]:  # pyright: ignore
        if isinstance(other, EnumerationItem):
            if self._enumeration is other._enumeration:
                return self._value == other._value
        raise ValueError(
            "Can only compare equality between enumerations of the same type."
        )

    def __ne__(self, other) -> Bool[ArrayLike, ""]:  # pyright: ignore
        if isinstance(other, EnumerationItem):
            if self._enumeration is other._enumeration:
                return self._value != other._value
        raise ValueError(
            "Can only compare equality between enumerations of the same type."
        )

    def __repr__(self):
        prefix = f"{self._enumeration.__module__}.{self._enumeration.__qualname__}"
        message = self._enumeration[self]
        return f"{prefix}<{message}>"

    def error_if(self, token, pred):
        return branched_error_if(
            token, pred, self._value, self._enumeration._index_to_message
        )


if TYPE_CHECKING:
    import enum
    from typing import ClassVar

    class _Sequence(type):
        def __getitem__(cls, item) -> str:
            ...

        def __len__(cls) -> int:
            ...

    class Enumeration(  # pyright: ignore
        enum.Enum, EnumerationItem, metaclass=_Sequence
    ):
        _name_to_item: ClassVar[dict[str, EnumerationItem]]
        _index_to_message: ClassVar[list[str]]
        _base_offsets: ClassVar[dict["Enumeration", int]]

        @classmethod
        def promote(cls, item: "Enumeration") -> Self:
            ...

        @classmethod
        def where(cls, pred: Bool[ArrayLike, "..."], a: Self, b: Self) -> Self:
            ...

else:

    class Enumeration(metaclass=_EnumerationMeta):
        pass
