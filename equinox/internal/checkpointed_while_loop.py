import math
import operator
from typing import Callable, Optional, TypeVar, Union

import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Bool

from ..filters import is_array, is_inexact_array
from ..grad import filter_closure_convert, filter_custom_vjp, filter_vjp
from ..tree import tree_at


_T = TypeVar("T")
_Bool = Union[bool, Bool[Array, ""]]


def checkpointed_while_loop(
    cond_fun: Callable[[_T], _Bool],
    body_fun: Callable[[_T], _T],
    init_val: _T,
    max_steps: Optional[int],
    checkpoints: Union[int, str],
):
    # Used on the _checkpointed_while_loop branch: simplifies a lot of the reasoning
    # and bookkeeping necessary.
    init_val = jtu.tree_map(jnp.asarray, init_val)

    if max_steps is None or checkpoints == "all":
        return lax.while_loop(cond_fun, body_fun, init_val, max_steps)
    else:
        assert isinstance(max_steps, int)
        if checkpoints == "binomial":
            checkpoints = math.ceil(math.log2(max_steps))
        else:
            raise ValueError(f"Unrecognised checkpoints={checkpoints}")
        assert isinstance(checkpoints, int)
        body_fun = filter_closure_convert(body_fun, init_val)
        vjp_arg = (init_val, body_fun)
        return _checkpointed_while_loop(vjp_arg, cond_fun, max_steps, checkpoints)


@filter_custom_vjp
def _checkpointed_while_loop(vjp_arg, cond_fun, max_steps, checkpoints):
    del checkpoints
    init_val, body_fun = vjp_arg
    return lax.while_loop(cond_fun, body_fun, init_val, max_steps)


def _get_residual(vjp_fn):
    (residuals,) = vjp_fn.args
    (residuals,) = residuals.args
    return residuals


def _call(val, body_fun):
    return body_fun(val)


_sentinel = object()


# TODO: introduce checkpointing!
def _checkpointed_while_loop_fwd(vjp_arg, cond_fun, max_steps, checkpoints):
    assert jtu.tree_map(is_array, vjp_arg)
    init_val, body_fun = vjp_arg

    vjp_fns = _sentinel

    def f(carry, _):
        step, val = carry
        # Use `filter_vjp` to neatly handle floating-point arrays.
        # We pass in `body_fun` as an argument as it contains its closed-over values in
        # its PyTree structure, and we do want to compute cotangents wrt these.
        val2, vjp_fn = filter_vjp(_call, val, body_fun)
        residual = _get_residual(vjp_fn)
        nonlocal vjp_fns
        vjp_fns = tree_at(_get_residual, vjp_fn, object())
        return (step + 1, val2), residual

    def early_exit(carry):
        _, val = carry
        return jnp.invert(cond_fun(val))

    init_carry = 0, init_val
    (num_steps, final_val), residuals = lax.scan(
        f, init_carry, xs=None, length=max_steps, early_exit=early_exit
    )
    assert vjp_fns is not _sentinel
    vjp_fns = tree_at(_get_residual, vjp_fns, residuals)
    return final_val, (num_steps, vjp_fns)


def _checkpointed_while_loop_bwd(
    residuals, grad_final_val, vjp_arg, cond_fun, max_steps, checkpoints
):
    _, body_fun = vjp_arg
    grad_body_fun = jtu.tree_map(
        lambda x: jnp.zeros_like(x) if is_inexact_array(x) else None, body_fun
    )
    del cond_fun, body_fun
    num_steps, vjp_fns = residuals
    init_carry = num_steps, grad_final_val, grad_body_fun

    def _cond_fun(carry):
        step, _, _ = carry
        return step > 0

    def _body_fun(carry):
        step, grad_val, grad_body_fun = carry
        step = step - 1
        vjp_fn = jtu.tree_map(lambda x: x[step], vjp_fns)
        (grad_val2, grad_body_fun2) = vjp_fn(grad_val)
        grad_body_fun_update = jtu.tree_map(operator.add, grad_body_fun2, grad_body_fun)
        return step, grad_val2, grad_body_fun_update

    _, grad_init_val, grad_body_fun = lax.while_loop(_cond_fun, _body_fun, init_carry)
    return grad_init_val, grad_body_fun


_checkpointed_while_loop.defvjp(
    _checkpointed_while_loop_fwd, _checkpointed_while_loop_bwd
)


# TODO: given number of checkpoints and number of forward steps, calculate total
# number of steps. (+forward as method to RecursiveCheckpointAdjoint)
