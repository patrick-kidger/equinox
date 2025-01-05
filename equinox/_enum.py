import warnings
from typing import Any, cast, TYPE_CHECKING, Union

import jax._src.traceback_util as traceback_util
import jax.core
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Int

from ._doc_utils import doc_repr
from ._errors import branched_error_if
from ._module import field, Module


traceback_util.register_exclusion(__file__)


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
        name_to_item[name] = EnumerationItem(np.array(our_index, dtype=np.int32), cls)  # pyright: ignore
        index_to_message.append(message)
        return our_index
    else:
        our_index = our_item._value.item()
        existing_message = index_to_message[our_index]
        if message != existing_message:
            warnings.warn(
                f"Enumeration has duplicate incompatible values for {name}. Using the "
                f"latest, which is '{message}'."
            )
        index_to_message[our_index] = message
        return our_index


class _EnumerationMeta(type):
    def __new__(mcs, cls_name, bases, namespace):
        assert "_name_to_item" not in namespace
        assert "_index_to_message" not in namespace
        assert "_base_offsets" not in namespace
        name_to_item = {}
        index_to_message = []
        base_offsets = {}
        new_namespace = dict(
            _name_to_item=name_to_item,
            _index_to_message=index_to_message,
            _base_offsets=base_offsets,
        )
        if cls_name == "Enumeration" and "Enumeration" not in globals().keys():
            new_namespace["promote"] = namespace.pop("promote")
            new_namespace["where"] = namespace.pop("where")
        else:
            if "promote" in namespace:
                raise ValueError(
                    "Cannot have enumeration item with name `promote`, as this "
                    "conflicts with the classmethod of this name"
                )
            if "where" in namespace:
                raise ValueError(
                    "Cannot have enumeration item with name `where`, as this conflicts "
                    "with the classmethod of this name"
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
                    base_offsets[base] = np.asarray(offsets)
        for name, message in namespace.items():
            if not _magic(name):
                _store(cls, name, message, name_to_item, index_to_message)
        if "__doc__" not in cls.__dict__ or cls.__doc__ is None:
            doc_pieces = [
                """An
[enumeration](https://docs.kidger.site/equinox/api/enumerations/), with the
following entries:
"""
            ]
            for name, item in name_to_item.items():
                message = index_to_message[item._value.item()]
                if message == "":
                    doc_pieces.append(f"- `{name}`")
                else:
                    message = message.replace("\n", "\n    ")
                    doc_pieces.append(f"- `{name}`: {message}")
            cls.__doc__ = "\n\n".join(doc_pieces)
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


class EnumerationItem(Module):
    _value: Int[Union[Array, np.ndarray[Any, np.dtype[np.signedinteger]]], ""]
    # Should have annotation `"type[Enumeration]"`, but this fails due to beartype bug
    # #289.
    _enumeration: Any = field(static=True)

    if TYPE_CHECKING:

        def __init__(self, x):
            pass

    def __eq__(self, other) -> Bool[Array, ""]:
        if isinstance(other, EnumerationItem):
            if self._enumeration is other._enumeration:
                with jax.ensure_compile_time_eval():
                    return jnp.asarray(self._value == other._value)
        raise ValueError(
            "Can only compare equality between enumerations of the same type."
        )

    def __ne__(self, other) -> Bool[Array, ""]:  # pyright: ignore
        if isinstance(other, EnumerationItem):
            if self._enumeration is other._enumeration:
                with jax.ensure_compile_time_eval():
                    return jnp.asarray(self._value != other._value)
        raise ValueError(
            "Can only compare equality between enumerations of the same type."
        )

    def __repr__(self):
        prefix = f"{self._enumeration.__module__}.{self._enumeration.__qualname__}"
        message = self._enumeration[self]
        return f"{prefix}<{message}>"

    def error_if(self, token, pred):
        """Conditionally raise a runtime error, with message given by this enumeration
        item.

        See [`equinox.error_if`][] for more information on the behaviour of errors.

        **Arguments:**

        - `token`: the token to thread the error on to. This should be a PyTree
            containing at least one array. The error will be checked after this array
            has been computed, and before the return value from this function is used.
        - `pred`: the condition to raise the error on. Will raise if this evaluates to
            True at runtime.

        **Returns:**

        `token` is returned unchanged.
        """
        return branched_error_if(
            token, pred, self._value, self._enumeration._index_to_message
        )

    def is_traced(self) -> bool:
        return isinstance(self._value, jax.core.Tracer)


if TYPE_CHECKING:
    import enum
    from typing import ClassVar
    from typing_extensions import Self

    class _Sequence(type):
        def __getitem__(cls, item) -> str: ...

        def __len__(cls) -> int: ...

    class Enumeration(enum.Enum, EnumerationItem, metaclass=_Sequence):  # pyright: ignore
        _name_to_item: ClassVar[dict[str, EnumerationItem]]  # pyright: ignore
        _index_to_message: ClassVar[list[str]]  # pyright: ignore
        _base_offsets: ClassVar[dict["Enumeration", int]]

        @classmethod
        def promote(cls, item: "Enumeration") -> Self: ...

        @classmethod
        def where(cls, pred: Bool[ArrayLike, "..."], a: Self, b: Self) -> Self: ...

else:
    _Enumeration = doc_repr(Any, "Enumeration")

    class Enumeration(metaclass=_EnumerationMeta):
        """JAX-compatible enums.

        Enumerations are instantiated using class syntax, and values looked up on the
        class:
        ```python
        class RESULTS(eqx.Enumeration):
            success = "Hurrah!"
            linear_solve_failed = "Linear solve failed to converge"
            diffeq_solve_failed = "Differential equation solve exploded horribly"

        result = RESULTS.success
        ```

        Enumerations support equality checking:
        ```python
        x = jnp.where(result == RESULTS.success, a, b)
        ```

        Enumerations cannot be compared against anything except an Enumeration of the
        same type:
        ```python
        result == 0  # ValueError at trace time: `0` is an integer, not an Enumeration.
        result == SOME_OTHER_ENUMERATION.foo  # Likewise, also a ValueError.
        ```

        Enumerations can be passed through JIT:
        ```python
        jax.jit(lambda x: x)(RESULTS.success)
        ```

        Enumerations use their assigned value in their repr:
        ```python
        print(RESULTS.success)  # RESULTS<Hurrah!>
        ```

        Given a Enumeration element, just the string can be looked up by indexing it:
        ```python
        result = RESULTS.success
        print(RESULTS[result])  # Hurrah!
        ```

        Enumerations support inheritance, to include all of the superclasses' fields as
        well as any new ones. Note that you will need add a `# pyright: ignore` wherever
        you inherit.
        ```python
        class RESULTS(eqx.Enumeration):
            success = "success"
            linear_solve_failed = "Linear solve failed to converge"
            diffeq_solve_failed = "Differential equation solve exploded horribly"

        class MORE_RESULTS(RESULTS):  # pyright: ignore
            flux_capacitor_overloaded = "Run for your life!"

        result = MORE_RESULTS.linear_solve_failed
        ```

        Enumerations are often used to represent error conditions. As such they have
        built-in support for raising runtime errors, via [`equinox.error_if`][]:
        ```python
        x = result.error_if(x, pred)
        ```
        this is equivalent to `x = eqx.error_if(x, pred, msg)`, where `msg` is the
        string corresponding to the enumeration item.
        """

        @classmethod
        def promote(cls, item: _Enumeration) -> _Enumeration:
            """Enums support `.promote` (on the class) to promote from an inherited
            class.

            !!! Example

                ```python
                class RESULTS(eqx.Enumeration):
                    success = "success"
                    linear_solve_failed = "Linear solve failed to converge"
                    diffeq_solve_failed = "Differential equation solve exploded horribly"

                class MORE_RESULTS(RESULTS):  # pyright: ignore
                    flux_capacitor_overloaded = "Run for your life!"

                result == RESULTS.success

                # This is a ValueError at trace time
                result == MORE_RESULTS.success

                # This works. You can only promote from superclasses to subclasses.
                result = MORE_RESULTS.promote(result)
                result == MORE_RESULTS.success
                ```

            **Arguments:**

            - `item`: an item from a parent Enumeration.

            **Returns:**

            `item`, but as a member of this Enumeration.
            """  # noqa: E501
            if not isinstance(item, EnumerationItem):
                raise ValueError("Can only promote enumerations.")
            if (not issubclass(cls, item._enumeration)) or item._enumeration is cls:
                raise ValueError("Can only promote from inherited enumerations.")
            with jax.ensure_compile_time_eval():
                value = jnp.asarray(cls._base_offsets[item._enumeration])[item._value]
            return EnumerationItem(value, cls)

        @classmethod
        def where(
            cls, pred: Bool[ArrayLike, "..."], a: _Enumeration, b: _Enumeration
        ) -> _Enumeration:
            """Enumerations support `.where` (on the class), analogous to `jnp.where`.

            !!! Example

                ```python
                result = RESULTS.where(diff < tol, RESULTS.success, RESULTS.linear_solve_failed)
                result = RESULTS.where(step < max_steps, result, RESULTS.diffeq_solve_failed)
                ```

            **Arguments:**

            - `pred`: a scalar boolean array.
            - `a`: an item of the enumeration. Must be of the same Enumeration as
                `.where` is accessed from.
            - `b`: an item of the enumeration. Must be of the same Enumeration as
                `.where` is accessed from.

            **Returns:**

            `a` if `pred` is true, and `b` is `pred` is false.
            """  # noqa: E501
            if jnp.shape(pred) != () or jnp.result_type(pred) != jnp.bool_:
                raise ValueError("`where` requires a scalar boolean predicate")
            if isinstance(a, EnumerationItem) and isinstance(b, EnumerationItem):
                if a._enumeration is cls and b._enumeration is cls:
                    if a._value is b._value:
                        return a
                    else:
                        with jax.ensure_compile_time_eval():
                            value = jnp.where(pred, a._value, b._value)
                        cls = cast(type[Enumeration], cls)
                        return EnumerationItem(value, cls)
            name = f"{cls.__module__}.{cls.__qualname__}"
            raise ValueError(
                f"Arguments to {name}.where(...) must be members of {name}."
            )
