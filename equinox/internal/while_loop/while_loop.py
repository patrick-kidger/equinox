from typing import Any, Callable, Literal, Optional, Sequence, TypeVar, Union

import jax.lax as lax
from jaxtyping import Array, Bool

from .bounded import bounded_while_loop
from .checkpointed import checkpointed_while_loop
from .common import common_rewrite


_T = TypeVar("_T")
_Bool = Union[bool, Bool[Array, ""]]
_Node = Any


# In time we may be able to deprecate this:
# - bloops will support reverse-mode autodiff;
# - stateful ops might make efficient scatters happen automatically.
# So the only bit we have to do ourselves will be online checkpointing.
def while_loop(
    cond_fun: Callable[[_T], _Bool],
    body_fun: Callable[[_T], _T],
    init_val: _T,
    *,
    max_steps: Optional[int] = None,
    buffers: Optional[Callable[[_T], Union[_Node, Sequence[_Node]]]] = None,
    kind: Literal["lax", "checkpointed", "bounded"],
    checkpoints: Optional[int] = None,
    base: int = 16,
) -> _T:
    """A better while loop, supporting (1) reverse-mode autodifferentiation; (2) online
    checkpointing schemes; (3) efficient in-place scatters (normally XLA tends to make
    unnecessary copies).

    **Arguments:**

    - `cond_fun`: As `lax.while_loop`.
    - `body_fun`: As `lax.while_loop`.
    - `init_val`: As `lax.while_loop`.

    - `max_steps`: A bound on the maximum number of steps, after which the loop
        terminates unconditionally. Can be set to `None` for arbitrarily many steps.

    - `buffers`: If passed, then every leaf of `tree_leaves(buffers(init_val))` must
        be an array; all such arrays become buffers supporting only `[]` and
        `.at[].set()`. However they will act efficiently, without spurious copies.

    - `kind`: The type of while loop that is lowered to underneath. This may either be
        `None`, an `int`, `"lax"`, or `"bounded"`.

        If `kind` is `"lax"` then the loop is efficiently forward-mode
        autodifferentiable, but does not support reverse-mode autodifferentiation.

        If `kind` is `"checkpointed"` then the loop is efficiently reverse-mode
        autodifferentiable, but does not support forward-mode autodifferentiation, and
        all `buffers` will be write-only whilst inside the loop.

        If `kind` is `"bounded"` then the loop may be both forward- and reverse-mode
        autodifferentiated, but it requires an `int` value for `max_steps`, and as
        `max_steps` grows then time and memory usage will increase.

    - `checkpoints`: Only used if `kind="checkpointed"`. Specifies the number of
        checkpoints to use; if `None` then this is automatically derived from
        `max_steps`.

    - `base`: Only used if `kind="bounded"`. Run time will increase slightly as `base`
        increases. Compilation time will decrease substantially as
        `math.ceil(math.log(max_steps, base))` decreases. (Which happens as `base`
        increases.)

    **Returns:**

    The final value; as `lax.while_loop`.
    """

    if kind == "lax":
        del kind, checkpoints, base
        cond_fun_, body_fun_, init_val_, _ = common_rewrite(
            cond_fun, body_fun, init_val, max_steps, buffers, readable=True
        )
        del cond_fun, body_fun, init_val
        _, _, final_val = lax.while_loop(cond_fun_, body_fun_, init_val_)
        return final_val
    elif kind == "checkpointed":
        del kind, base
        return checkpointed_while_loop(
            cond_fun,
            body_fun,
            init_val,
            max_steps=max_steps,
            buffers=buffers,
            checkpoints=checkpoints,
        )
    elif kind == "bounded":
        del kind, checkpoints
        if max_steps is None:
            raise ValueError("kind='bounded' requires `max_steps` to be specified")
        return bounded_while_loop(
            cond_fun,
            body_fun,
            init_val,
            max_steps=max_steps,
            buffers=buffers,
            base=base,
        )
    else:
        raise ValueError(f"Unrecognised kind of while loop '{kind}'")
