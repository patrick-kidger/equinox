import math
from collections.abc import Callable, Sequence
from typing import Any, Optional, TypeVar, Union

import jax
import jax.lax as lax
import jax.tree_util as jtu
from jaxtyping import Array, Bool

from .common import common_rewrite, fixed_asarray


_T = TypeVar("_T")
_Bool = Union[bool, Bool[Array, ""]]
_Node = Any


def bounded_while_loop(
    cond_fun: Callable[[_T], _Bool],
    body_fun: Callable[[_T], _T],
    init_val: _T,
    *,
    max_steps: int,
    buffers: Optional[Callable[[_T], Union[_Node, Sequence[_Node]]]] = None,
    base: int = 16,
):
    """Reverse-mode autodifferentiable while loop.

    **Arguments:**

    - `cond_fun`: As `lax.while_loop`.
    - `body_fun`: As `lax.while_loop`.
    - `init_val`: As `lax.while_loop`.
    - `max_steps`: A bound on the maximum number of steps, after which the loop
        terminates unconditionally.
    - `buffers`: If passed, then every leaf of `tree_leaves(buffers(init_val))` must
        be an array; all such arrays become buffers supporting only `[]` and
        `.at[].set()`. However they will act efficiently, without spurious copies.
    - `base`: Run time will increase slightly as `base` increases. Compilation time will
        decrease substantially as `math.ceil(math.log(max_steps, base))` decreases.
        (Which happens as `base` increases.)

    **Returns:**

    The final value; as `lax.while_loop`.
    """

    if not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a non-negative integer")
    init_val = jtu.tree_map(fixed_asarray, init_val)
    if max_steps == 0:
        return init_val

    cond_fun_, body_fun_, init_val_, _ = common_rewrite(
        cond_fun, body_fun, init_val, max_steps, buffers, makes_false_steps=True
    )
    del cond_fun, body_fun, init_val
    rounded_max_steps = base ** int(math.ceil(math.log(max_steps, base)))
    _, _, _, val = _while_loop(cond_fun_, body_fun_, init_val_, rounded_max_steps, base)
    return val


def _while_loop(cond_fun, body_fun, val, max_steps, base):
    if max_steps == 1:
        return body_fun(val)
    else:

        def call(val):
            return _while_loop(cond_fun, body_fun, val, max_steps // base, base)

        def scan_fn(val, _):
            return lax.cond(cond_fun(val), call, lambda x: x, val), None

        # Don't put checkpointing on the lowest level
        if max_steps != base:
            scan_fn = jax.checkpoint(scan_fn, prevent_cse=False)  # pyright: ignore

        return lax.scan(scan_fn, val, xs=None, length=base)[0]
