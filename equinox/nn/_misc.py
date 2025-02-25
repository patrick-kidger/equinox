import typing_extensions as te
from collections.abc import Callable, Sequence
from typing import Any, TYPE_CHECKING, TypeVar, Union

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray


_T = TypeVar("_T", bound=Sequence)

if TYPE_CHECKING:
    # StrictTypeGuard is a pyright-specific extension that performs type narrowing in
    # the `else` branch as well:
    # https://github.com/microsoft/pyright/issues/3450
    def all_sequences(
        x: Union[Sequence[Any], Sequence[_T]],
    ) -> "te.TypeIs[Sequence[_T]]": ...

    _S = TypeVar("_S")

    def named_scope(name: str) -> Callable[[_S], _S]: ...

else:
    # beartype doesn't like StrictTypeGuard
    def all_sequences(x: Union[Sequence[Any], Sequence[_T]]) -> bool:
        return all(isinstance(xi, Sequence) for xi in x)

    named_scope = jax.named_scope


def default_init(
    key: PRNGKeyArray, shape: tuple[int, ...], dtype: Any, lim: float
) -> jax.Array:
    if jnp.issubdtype(dtype, jnp.complexfloating):
        real_dtype = jnp.finfo(dtype).dtype
        rkey, ikey = jrandom.split(key, 2)
        real = jrandom.uniform(rkey, shape, real_dtype, minval=-lim, maxval=lim)
        imag = jrandom.uniform(ikey, shape, real_dtype, minval=-lim, maxval=lim)
        return real.astype(dtype) + 1j * imag.astype(dtype)
    else:
        return jrandom.uniform(key, shape, dtype, minval=-lim, maxval=lim)
