import typing_extensions as te
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING, TypeVar, Union


_T = TypeVar("_T", bound=Sequence)

if TYPE_CHECKING:
    # StrictTypeGuard is a pyright-specific extension that performs type narrowing in
    # the `else` branch as well:
    # https://github.com/microsoft/pyright/issues/3450
    def all_sequences(
        x: Union[Sequence[Any], Sequence[_T]]
    ) -> "te.StrictTypeGuard[Sequence[_T]]":
        ...

else:
    # beartype doesn't like StrictTypeGuard
    def all_sequences(x: Union[Sequence[Any], Sequence[_T]]) -> bool:
        return all(isinstance(xi, Sequence) for xi in x)
