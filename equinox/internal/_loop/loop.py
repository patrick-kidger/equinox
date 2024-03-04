from collections.abc import Callable, Sequence
from typing import Any, Literal, Optional, TypeVar, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Bool

from .bounded import bounded_while_loop
from .checkpointed import checkpointed_while_loop
from .common import common_rewrite


_Carry = TypeVar("_Carry")
_X = TypeVar("_X")
_Y = TypeVar("_Y")
_Bool = Union[bool, Bool[Array, ""]]
_Node = Any


def while_loop(
    cond_fun: Callable[[_Carry], _Bool],
    body_fun: Callable[[_Carry], _Carry],
    init_val: _Carry,
    *,
    max_steps: Optional[int] = None,
    buffers: Optional[Callable[[_Carry], Union[_Node, Sequence[_Node]]]] = None,
    kind: Literal["lax", "checkpointed", "bounded"],
    checkpoints: Optional[int] = None,
    base: int = 16,
) -> _Carry:
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
        You should avoid performing in-place updates to any quantity that is not a
        buffer.

    - `kind`: The type of while loop that is lowered to underneath. This may either be
        `"lax"`, `"checkpointed"`, or `"bounded"`.

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

    !!! Danger

        Note that `buffers` is subject to the following restrictions:

        - You should never write to the same location twice. (Even before it is passed
            into the loop: e.g.
            ```python
            xs = xs.at[0].set(x).at[0].set(y)
            while_loop(cond, body, xs, buffers=lambda xs: xs)
            ```
            is not allowed.)
        - You should only read from it (`buf[i]`) at locations (`i`) that you have
          written to previously (`buf.at[i].set(...)`).

        These assumptions are *completely unchecked* and you will get incorrect
        gradients if you violate these assumptions.

    **Returns:**

    The final value; as `lax.while_loop`.
    """

    if kind == "lax":
        del kind, checkpoints, base
        cond_fun_, body_fun_, init_val_, _ = common_rewrite(
            cond_fun, body_fun, init_val, max_steps, buffers, makes_false_steps=False
        )
        del cond_fun, body_fun, init_val
        _, _, _, final_val = lax.while_loop(cond_fun_, body_fun_, init_val_)
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


def scan(
    f: Callable[[_Carry, _X], tuple[_Carry, _Y]],
    init: _Carry,
    xs: _X,
    length: Optional[int] = None,
    *,
    buffers: Optional[Callable[[_Carry], Union[_Node, Sequence[_Node]]]] = None,
    kind: Literal["lax", "checkpointed"],
    checkpoints: Union[None, int, Literal["all"]] = None,
) -> tuple[_Carry, _Y]:
    """As `jax.lax.scan`, but with optional checkpointing to reduce memory usage.

    **Arguments:**

    - `f`: As `jax.lax.scan`.
    - `init`: As `jax.lax.scan`.
    - `xs`: As `jax.lax.scan`.
    - `length`: As `jax.lax.scan`.

    - `buffers`: If passed, then every leaf of `tree_leaves(buffers(init))` must
        be an array; all such arrays become buffers supporting only `[]` and
        `.at[].set()`. However they will act efficiently, without spurious copies.
        You should avoid performing in-place updates to any quantity that is not a
        buffer.

    - `kind`: The type of scan that is lowered to underneath. This may either be
        `"lax"` or `"checkpointed"`.

        If `kind` is `"lax"` then the usual `lax.scan` is used.

        If `kind` is `"checkpointed"` then the scan uses checkpointing to reduce memory
        usage. It will not be forward-mode autodifferentiable.

    - `checkpoints`: Only used if `kind="checkpointed"`. Specifies the number of
        checkpoints to use; if `None` then this is set proportional to `sqrt(length)`.
        Can also be a string `"all"`, representing checkpointing every step.

    !!! Danger

        Note that `buffers` is subject to the same restrictions as
        `equinox.internal.while_loop`.

    Returns:

    As `jax.lax.scan`.
    """

    init, xs = jtu.tree_map(jnp.asarray, (init, xs))

    if kind == "lax":
        return lax.scan(f, init, xs, length)

    lengths = {jnp.shape(x)[0] for x in jtu.tree_leaves(xs)}
    if length is not None:
        lengths.add(length)
    if len(lengths) == 1:
        length = lengths.pop()
    else:
        raise ValueError(f"Got inconsistent lengths in scan: {lengths}.")
    del lengths

    if checkpoints == "all":
        checkpoints = length

    def cond_fun(val):
        return True

    def body_fun(val):
        i, carry, ys = val
        x = jtu.tree_map(lambda z: z[i], xs)
        carry, y = f(carry, x)
        ys = jtu.tree_map(lambda z, zs: zs.at[i].set(z), y, ys)
        return i + 1, carry, ys

    x0 = jtu.tree_map(lambda z: z[0], xs)
    _, y0_shape = jax.eval_shape(f, init, x0)
    ys = jtu.tree_map(lambda z: jnp.empty((length,) + z.shape, z.dtype), y0_shape)
    init_val = (0, init, ys)

    if buffers is None:
        _buffers = lambda val: val[2]
    else:
        node_or_nodes = buffers(init)
        is_tuple = True
        is_leaf = lambda node: node is node_or_nodes

        def _find(node):
            nonlocal is_tuple
            if is_leaf(node):
                is_tuple = False

        jtu.tree_map(_find, init, is_leaf=is_leaf)
        if is_tuple:
            _buffers = lambda val: tuple(buffers(val[1])) + (val[2],)
        else:
            _buffers = lambda val: (buffers(val[1]), val[2])

    _, carry, ys = checkpointed_while_loop(
        cond_fun,
        body_fun,
        init_val,
        buffers=_buffers,
        max_steps=length,
        checkpoints=checkpoints,
    )
    return carry, ys
