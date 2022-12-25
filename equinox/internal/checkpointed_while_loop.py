"""Implements backpropagation through a while loop by using checkpointing.

(Variously known as "treeverse", "optimal checkpointing", "binomial checkpointing",
"recursive checkpointing", "revolve", etc.)

The algorithm used here is the online version (when the number of steps isn't known in
advance), as proposed in:

    Stumm and Walther 2010
    New Algorithms for Optimal Online Checkpointing
    https://tu-dresden.de/mn/math/wir/ressourcen/dateien/forschung/publikationen/pdf2010/new_algorithms_for_optimal_online_checkpointing.pdf

and also depends on the results of:

    Wang and Moin 2008
    Minimal repetition dynamic checkpointing algorithm for unsteady adjoint calculation
    https://web.stanford.edu/group/ctr/ResBriefs08/4_checkpointing.pdf

This matches the performance of the offline version (classical treeverse, when the
number of steps is known in advance) provided that the number of steps is less than or
equal to `(num_checkpoints + 1) * (num_checkpoints + 2) / 2`; see the Stumm--Walther
paper. After that is may make extra steps (as compared to the offline version), but does
still have similar asymptotic complexity.

For context, the two classical references for (offline) treeverse are:

    Griewank 1992
    Achieiving logarithmic growth of temporal and spatial complexity in reverse
    automatic differentiation
    https://ftp.mcs.anl.gov/pub/tech_reports/reports/P228.pdf

and

    Griewank and Walther 2000
    Algorithm 799: revolve: an implementation of checkpointing for the reverse or
    adjoint mode of computational differentiation
    https://dl.acm.org/doi/pdf/10.1145/347837.347846
"""
# I think this code is not quite maximally efficient. A few things that could be
# improved:
# - The initial value is available on the backward pass twice: once as an argument,
#   once as a saved checkpoint. We should be able to get away without this repetition.
# - We only implement Algorithm I of Stumm--Wather. Additionally implementing
#   Algorithm II would be worthwhile. (But finickity, as their description of it in the
#   paper leaves something to be desired. And may also have an off-by-one-error, like
#   their Figure 2.2 does?)

import functools as ft
import operator
from typing import Callable, Optional, TypeVar, Union

import jax
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Bool

from ..filters import is_inexact_array
from ..grad import filter_closure_convert, filter_custom_vjp, filter_vjp
from .ad import nondifferentiable
from .errors import error_if
from .unvmap import unvmap_any, unvmap_max


_T = TypeVar("T")
_Bool = Union[bool, Bool[Array, ""]]


def checkpointed_while_loop(
    cond_fun: Callable[[_T], _Bool],
    body_fun: Callable[[_T], _T],
    init_val: _T,
    max_steps: Optional[int] = None,
    *,
    checkpoints: int,
):
    """Reverse-mode autodifferentiable while loop, using optimal online checkpointing.

    The usual `jax.lax.while_loop` is not reverse-mode autodifferentiable, since it
    would need to save a potentially unbounded amount of residuals between the forward
    and backward pass. However, JAX/XLA requires that all memory buffers be of known
    (bounded) size.

    This works around this limitation by saving values to a prespecified number of
    checkpoints, and then recomputing other intermediate value on-the-fly.

    Checkpointing in this way is a classical autodifferentiation technique, usually used
    to reduce memory consumption. (And it's still useful for this purpose for us too.)

    **Arguments:**

    - `cond_fun`: As `lax.while_loop`.
    - `body_fun`: As `lax.while_loop`.
    - `init_val`: As `lax.while_loop`.
    - `checkpoints`: The number of steps at which to checkpoint. The memory consumed
        will be that of `checkpoints`-many copies of `init_val`. (As the state is
        updated throughout the loop.)

    **Returns:**

    The final value; as `lax.while_loop`.

    !!! Info

        This function is not forward-mode autodifferentiable.

    !!! cite "References"

        Selecting which steps at which to save checkpoints (and when this is done, which
        old checkpoint to evict) is important for minimising the amount of recomputation
        performed.

        This is a difficult, but solved, problem! So if you are using this function in
        academic work, then you should cite the following references.

        The implementation here performs "online checkpointing", as the number of steps
        is not known in advance. This was developed in:

        ```bibtex
        @article{stumm2010new,
            author = {Stumm, Philipp and Walther, Andrea},
            title = {New Algorithms for Optimal Online Checkpointing},
            journal = {SIAM Journal on Scientific Computing},
            volume = {32},
            number = {2},
            pages = {836--854},
            year = {2010},
            doi = {10.1137/080742439},
        }

        @article{wang2009minimal,
            author = {Wang, Qiqi and Moin, Parviz and Iaccarino, Gianluca},
            title = {Minimal Repetition Dynamic Checkpointing Algorithm for Unsteady
                     Adjoint Calculation},
            journal = {SIAM Journal on Scientific Computing},
            volume = {31},
            number = {4},
            pages = {2549--2567},
            year = {2009},
            doi = {10.1137/080727890},
        }
        ```

        For reference, the classical "offline checkpointing" (also known as "treeverse",
        "recursive binary checkpointing", "revolved" etc.) was developed in:

        ```bibtex
        @article{griewank1992achieving,
            author = {Griewank, Andreas},
            title = {Achieving logarithmic growth of temporal and spatial complexity in
                     reverse automatic differentiation},
            journal = {Optimization Methods and Software},
            volume = {1},
            number = {1},
            pages = {35--54},
            year  = {1992},
            publisher = {Taylor & Francis},
            doi = {10.1080/10556789208805505},
        }

        @article{griewank2000revolve,
            author = {Griewank, Andreas and Walther, Andrea},
            title = {Algorithm 799: Revolve: An Implementation of Checkpointing for the
                     Reverse or Adjoint Mode of Computational Differentiation},
            year = {2000},
            publisher = {Association for Computing Machinery},
            volume = {26},
            number = {1},
            doi = {10.1145/347837.347846},
            journal = {ACM Trans. Math. Softw.},
            pages = {19--45},
        }
        ```
    """
    if checkpoints < 1:
        raise ValueError("Must have at least one checkpoint")
    if max_steps == 0:
        return init_val
    init_val = jtu.tree_map(jnp.asarray, init_val)
    body_fun = filter_closure_convert(body_fun, init_val)
    vjp_arg = (init_val, body_fun)
    return _checkpointed_while_loop(vjp_arg, cond_fun, max_steps, checkpoints)


@filter_custom_vjp
def _checkpointed_while_loop(vjp_arg, cond_fun, max_steps, checkpoints):
    """Uncheckpointed forward used when not differentiating."""
    del checkpoints
    init_val, body_fun = vjp_arg
    if max_steps is None:
        # Hashable wrapper; needed to avoid JAX issue #13554
        def _body_fun(val):
            return body_fun(val)

        return lax.while_loop(cond_fun, _body_fun, init_val)

    def _cond_fun(carry):
        step, val = carry
        return (step < max_steps) & cond_fun(val)

    def _body_fun(carry):
        step, val = carry
        return step + 1, body_fun(val)

    _, final_val = lax.while_loop(_cond_fun, _body_fun, (0, init_val))
    return final_val


def _assert_unbatched(*x):
    """Identity function. Raises a trace-time assert if it is batched.

    Careful control over which quantities get batched is needed to make
    `checkpointed_while_loop` work under `jax.vmap`.
    """
    return jtu.tree_map(_assert_unbatched_p.bind, x)


def _error(x, b):
    msg = (
        "Internal trace-time error in `equinox.internal.checkpointed_while_loop`. "
        "Please raise an issue at https://http://github.com/patrick-kidger/equinox"
    )
    assert False, msg


_assert_unbatched_p = jax.core.Primitive("assert_unbatched")
_assert_unbatched_p.def_impl(lambda x: x)
_assert_unbatched_p.def_abstract_eval(lambda x: x)
batching.primitive_batchers[_assert_unbatched_p] = _error
mlir.register_lowering(
    _assert_unbatched_p, mlir.lower_fun(lambda x: x, multiple_results=False)
)


def _scalar_index(i, x):
    """As `x[i]`, but slightly more efficient for a nonnegative scalar `i`.

    (As it avoids support for negative indexing, and lowers to `dynamic_slice` rather
     than `gather`.)
    """
    assert jnp.shape(i) == ()
    return lax.dynamic_index_in_dim(x, i, keepdims=False)


def _unique_index(i, x):
    """As `x[i]`, but states that `i` has unique indices."""
    # lax.gather's API is impenetrable. This is way easier...
    jaxpr = jax.make_jaxpr(lambda _x, _i: _x[_i])(x, i)
    *rest_eqns, eqn = jaxpr.jaxpr.eqns
    assert eqn.primitive == jax.lax.gather_p
    new_params = dict(eqn.params)
    new_params["unique_indices"] = True
    new_eqn = eqn.replace(params=new_params)
    new_eqns = (*rest_eqns, new_eqn)
    new_jaxpr = jaxpr.replace(jaxpr=jaxpr.jaxpr.replace(eqns=new_eqns))
    (out,) = jax.core.jaxpr_as_fun(new_jaxpr)(x, i)
    return out


def _stumm_walther_i(step, save_state):
    """Algorithm 1 from:

    Stumm and Walther 2010
    New Algorithms for Optimal Online Checkpointing
    https://tu-dresden.de/mn/math/wir/ressourcen/dateien/forschung/publikationen/pdf2010/new_algorithms_for_optimal_online_checkpointing.pdf
    """
    step, save_state = _assert_unbatched(step, save_state)
    i, o, p, s = save_state
    index = i
    save_residual = s
    i = jnp.where(s, i + 1, i)
    i = jnp.where(s & (i > o), 1, i)
    s = jnp.where(step + 1 == p, False, s)
    pred = step == p
    p = jnp.where(pred, p + o, p)
    o = jnp.where(pred, o - 1, o)
    i = jnp.where(pred, o, i)
    s = jnp.where(pred, o > 0, s)
    out = save_residual, index, (i, o, p, s)
    msg = (
        "Internal run-time error when checkpointing "
        "`equinox.internal.checkpointed_while_loop`. "
        "Please raise an issue at https://http://github.com/patrick-kidger/equinox"
    )
    out = error_if(out, pred & (o == -1), msg)
    (out,) = _assert_unbatched(out)
    return out


def _any_dispensable(dispensable, residual_steps, levels):
    del levels
    dispensable_steps = jnp.where(dispensable, residual_steps, 0)
    index = dispensable_steps.argmax()
    level = 0
    dispensable2 = dispensable.at[index].set(False)
    return index, level, dispensable2


def _none_dispensable(dispensable, residual_steps, levels):
    index = residual_steps.argmax()
    level = levels[index] + 1
    dispensable2 = jnp.where(levels < levels[index], True, dispensable)
    return index, level, dispensable2


def _wang_moin(step, save_state, residual_steps):
    """Algorithm 1 or 3 from:

    Wang and Moin 2008
    Minimal repetition dynamic checkpointing algorithm for unsteady adjoint calculation
    https://web.stanford.edu/group/ctr/ResBriefs08/4_checkpointing.pdf
    """
    step, save_state, residual_steps = _assert_unbatched(
        step, save_state, residual_steps
    )
    levels, dispensable = save_state
    if len(residual_steps) == 1:
        # Don't save if we only have space to save the initial value, which is already
        # stored.
        save_residual = False
        index = 0
        levels2 = levels
        dispensable2 = dispensable
    else:
        save_residual = len(residual_steps) > 1
        index, level, dispensable2 = lax.cond(
            dispensable.any(),
            _any_dispensable,
            _none_dispensable,
            dispensable,
            residual_steps,
            levels,
        )
        levels2 = levels.at[index].set(level)
    out = save_residual, index, (levels2, dispensable2)
    (out,) = _assert_unbatched(out)
    return out


def _stumm_walther_i_wrapper(step, save_state, residual_steps):
    del residual_steps
    save_state_sw_i, save_state_wm = save_state
    save_residual, index, save_state_sw_i_2 = _stumm_walther_i(step, save_state_sw_i)
    return save_residual, index, (save_state_sw_i_2, save_state_wm)


def _wang_moin_wrapper(step, save_state, residual_steps):
    save_state_sw_i, save_state_wm = save_state
    save_residual, index, save_state_wm_2 = _wang_moin(
        step, save_state_wm, residual_steps
    )
    return save_residual, index, (save_state_sw_i, save_state_wm_2)


def _should_save_residual(step, save_state, residual_steps, u2_minus_2):
    """This is the controller for whether we should save the current value at each step,
    and if so which memory location to save it in.
    """
    # TODO: also implement Algorithm 2 of Stumm and Walther, which gives improved
    # results for u2 < step < u3.
    return lax.cond(
        step > u2_minus_2,
        _wang_moin_wrapper,
        _stumm_walther_i_wrapper,
        step,
        save_state,
        residual_steps,
    )


def _unreachable_checkpoint_step(x):
    """Dummy value used to represent a checkpoint we never reach."""
    dtype = jnp.result_type(x)  # x can be a dtype or an arraylike
    return jnp.iinfo(dtype).max


def _checkpointed_while_loop_fwd(vjp_arg, cond_fun, max_steps, checkpoints):
    """Run the while loop, saving checkpoints whenever the controller
    (`_should_save_residual`) requires.
    """
    init_val, body_fun = vjp_arg
    # Equation (2.2) of Stumm and Walther
    u2_minus_2 = ((checkpoints + 1) * (checkpoints + 2)) // 2 - 2

    def _cond_fun(carry):
        pred, *_ = carry
        # We set things up with an unbatched cond_fun so that the body doesn't all get
        # batched, so that `_should_save_residual` doesn't get batched inputs, so that
        # our saved residuals are in lockstep across the batch.
        return unvmap_any(pred)

    def _body_fun(carry):
        pred, step, save_state, val, residual_steps, residuals = carry
        save_state, residual_steps = _assert_unbatched(save_state, residual_steps)

        step2 = step + 1
        save_residual, index, save_state2 = _should_save_residual(
            unvmap_max(step), save_state, residual_steps, u2_minus_2
        )
        val2 = body_fun(val)
        # Manually handle only keeping batch elements that should have ran, since have
        # an unvmap'd cond_fun.
        pred2 = pred & cond_fun(val2)
        if max_steps is not None:
            pred2 = pred2 & (step2 < max_steps)
        step2, val2 = jtu.tree_map(
            lambda v, v2: lax.select(pred, v2, v), (step, val), (step2, val2)
        )

        def _maybe_update(xs, x):
            where_x = jnp.where(save_residual, x, _scalar_index(index, xs))
            return lax.dynamic_update_index_in_dim(xs, where_x, index, axis=0)

        residual_steps2, residuals2 = jtu.tree_map(
            _maybe_update, (residual_steps, residuals), (unvmap_max(step), val)
        )
        save_state2, residual_steps2 = _assert_unbatched(save_state2, residual_steps2)
        return pred2, step2, save_state2, val2, residual_steps2, residuals2

    int_dtype = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32
    init_pred = cond_fun(init_val)
    init_step = jnp.array(0, dtype=int_dtype)  # dtype matches init_residual_steps
    init_save_state_sw_i = 0, checkpoints, checkpoints, True
    init_save_state_wm = (
        jnp.zeros(checkpoints, dtype=int_dtype).at[0].set(jnp.iinfo(int_dtype).max),
        jnp.full((checkpoints,), False),
    )
    init_save_state = (init_save_state_sw_i, init_save_state_wm)
    # Uses the fact that `_unreachable_checkpoint_step` returns intmax, so that in our
    # sorting later, in the steps < checkpoints case, all unused memory gets sorted
    # to the end.
    init_residual_steps = jnp.full(
        checkpoints, _unreachable_checkpoint_step(int_dtype), dtype=int_dtype
    )
    # Fill value for the memory isn't important.
    init_residuals = jtu.tree_map(
        lambda x: jnp.zeros((checkpoints,) + x.shape, x.dtype), init_val
    )
    init_carry = (
        init_pred,
        init_step,
        init_save_state,
        init_val,
        init_residual_steps,
        init_residuals,
    )
    # `pred` is whether to make a step
    # `step` is an increasing counter 0, 1, 2, 3, ...
    # `save_state` is state used by the logic for whether to save a checkpoint each step
    # `val` is the evolving state of the loop
    # `residual_steps` is the buffer of the `step` for each checkpoint
    # `residuals` is the buffer of the `val` for each checkpoint

    final_carry = lax.while_loop(_cond_fun, _body_fun, init_carry)
    _, num_steps, _, final_val, final_residual_steps, final_residuals = final_carry

    # The above procedure may produce residuals saved in jumbled-up order. Meanwhile
    # treeverse (used on the backward pass) treats the residuals like a stack,
    # reading and writing the most recent residual to and from the end. So sort the
    # residuals we've produced here to obtain the desired invariant, i.e. that the
    # residuals are in order.
    # TODO: does this introduce a 2x memory overhead?  It may be that we can do better
    # here.
    sort_indices = jnp.argsort(final_residual_steps)
    final_residual_steps, final_residuals = jtu.tree_map(
        ft.partial(_unique_index, sort_indices), (final_residual_steps, final_residuals)
    )
    (final_residual_steps,) = _assert_unbatched(final_residual_steps)
    return final_val, (num_steps, final_residual_steps, final_residuals)


def _load_from_checkpoint(step_grad_val, index, residual_steps, residuals):
    """Loads a residual from the store of checkpoints."""
    # step_grad_val is the current location of grad_val.
    # index is the next currently empty slot for saving a residual.
    step_grad_val, index, residual_steps = _assert_unbatched(
        step_grad_val, index, residual_steps
    )

    # Subtract one to get the most recent residual, and then load it.
    # (Clip to zero just to not error from index == 0 on the very last step; the result
    # is unused in this case.)
    read_index = jnp.maximum(index - 1, 0)
    step_val2, val2 = jtu.tree_map(
        ft.partial(_scalar_index, read_index), (residual_steps, residuals)
    )

    # We may need to keep this residual around, and jump back to it multiple times.
    # (In which case index2 == index.) Or this may be the last time and we don't need to
    # save it any more. (In which case index2 == index - 1.)
    # If `step_val2 + 1 == step_grad_val2` then we're about to make a U-turn on the next
    # step, so we won't need to load from this checkpoint. (And as
    # `_load_from_checkpoint` is itself used within a U-turn, then in practice this
    # triggers whenever we get >1 U-turns back-to-back.)
    index2 = jnp.where(step_val2 + 1 == step_grad_val, read_index, index)
    step_val2, index2 = _assert_unbatched(step_val2, index2)
    return step_val2, val2, index2


def _maybe_save_to_checkpoint(
    step_val,
    step_grad_val,
    step_next_checkpoint,
    index,
    val,
    residual_steps,
    residuals,
    checkpoints,
):
    """Might save a residual to the store of checkpoints."""
    (
        step_val,
        step_grad_val,
        step_next_checkpoint,
        index,
        residual_steps,
    ) = _assert_unbatched(
        step_val, step_grad_val, step_next_checkpoint, index, residual_steps
    )
    save_checkpoint = step_val == step_next_checkpoint

    def _maybe_update(xs, x):
        where_x = jnp.where(save_checkpoint, x, _scalar_index(index, xs))
        return lax.dynamic_update_index_in_dim(xs, where_x, index, axis=0)

    residual_steps2, residuals2 = jtu.tree_map(
        _maybe_update, (residual_steps, residuals), (step_val, val)
    )
    index2 = jnp.where(save_checkpoint, index + 1, index)
    step_next_checkpoint2 = jnp.where(
        save_checkpoint,
        _calc_next_checkpoint(step_val, step_grad_val, index2, checkpoints),
        step_next_checkpoint,
    )
    index2, step_next_checkpoint2, residual_steps2 = _assert_unbatched(
        index2, step_next_checkpoint2, residual_steps2
    )
    return index2, step_next_checkpoint2, residual_steps2, residuals2


def _calc_next_checkpoint(step_val, step_grad_val, index, checkpoints):
    """Determines the step at which we next want to save a checkpoint."""
    # Note that when this function is called, `step_val` is always at the most recent
    # checkpoint.
    step_val, step_grad_val, index = _assert_unbatched(step_val, step_grad_val, index)

    # Using treeverse...
    # ...Checkpoints are either placed binomially (most of the time)...
    out_binomial = step_val + (step_grad_val - step_val) // 2
    # ...or linearly (when the space to cross fits within the checkpoint budget).
    out_linear = step_val + 1
    within_budget = (step_grad_val - step_val - 2) <= (checkpoints - index)
    out = jnp.where(within_budget, out_linear, out_binomial)
    # Why -2?
    # If `step_val + 1 == step_grad_val` then we're just going to make a single U-turn,
    # and don't need to store any checkpoints.
    # If `step_val + 2 == step_grad_val` then we're just going make a fwd, then a
    # U-turn, and can then load the existing checkpoint at `step_val` in order to make
    # the next U-turn. Once again we don't need to store any checkpoints.
    # If `step_val + 3 == step_grad_val` then ideally we would store a checkpoint at
    # `step_val + 1`.
    # Meanwhile, `checkpoints - index` is the number of spaces we have left in which to
    # store checkpoints. (e.g. `checkpoints == index` indicates that our buffer is
    # full.)
    # Thus if `step_val + 3 == step_grad_val` then
    # `step_grad_val - step_val - 2 <= checkpoints - index`
    # is the desired condition. (For "save a checkpoint if you can".)
    # Now proceed by induction: if `step_val + i == step_grad_val` for some `i > 3` and
    # we trigger this condition, we will reduce the LHS by one (as `step_val `increments
    # in a fwd step) and increase the RHS side by one (as `index` increments as we save
    # a checkpoint) and we eventually reduce back to the `step_val + 3 == step_grad_val`
    # case.

    # Logic as above: in these cases we don't need to store a checkpoint at all.
    no_checkpoint = index == checkpoints
    no_checkpoint = no_checkpoint | (step_val + 1 == step_grad_val)
    no_checkpoint = no_checkpoint | (step_val + 2 == step_grad_val)

    step_next_checkpoint = jnp.where(
        no_checkpoint, _unreachable_checkpoint_step(out), out
    )
    # Invariant: `step_val < step_next_checkpoint`. (Due to the invariant
    # `step_val + 1 <= step_grad_val`.)
    (step_next_checkpoint,) = _assert_unbatched(step_next_checkpoint)
    return step_next_checkpoint


def _fwd(
    body_fun,
    step_val,
    step_grad_val,
    step_next_checkpoint,
    index,
    val,
    grad_val,
    grad_body_fun,
):
    """Propagates the primal forward one step."""
    (step_val,) = _assert_unbatched(step_val)
    step_val2 = step_val + 1
    val2 = body_fun(val)
    (step_val2,) = _assert_unbatched(step_val2)
    return (
        step_val2,
        step_grad_val,
        step_next_checkpoint,
        index,
        val2,
        grad_val,
        grad_body_fun,
    )


def _make_u_turn(residual_steps, residuals, checkpoints):
    """Propagates the cotangent backward one step."""
    (residual_steps,) = _assert_unbatched(residual_steps)

    def _u_turn(
        body_fun,
        step_val,
        step_grad_val,
        step_next_checkpoint,
        index,
        val,
        grad_val,
        grad_body_fun,
    ):
        del step_val, step_next_checkpoint
        step_grad_val, index = _assert_unbatched(step_grad_val, index)

        # Use `filter_vjp` to neatly handle floating-point arrays.
        #
        # We pass in `body_fun` as an argument as it contains its closed-over values
        # in its PyTree structure, and we do want to compute cotangents wrt these.
        _, vjp_fn = filter_vjp(lambda b, v: b(v), body_fun, val)

        grad_body_fun_update, grad_val2 = vjp_fn(grad_val)
        grad_body_fun2 = jtu.tree_map(operator.add, grad_body_fun, grad_body_fun_update)
        step_grad_val2 = step_grad_val - 1
        step_val2, val2, index2 = _load_from_checkpoint(
            step_grad_val2, index, residual_steps, residuals
        )
        step_next_checkpoint2 = _calc_next_checkpoint(
            step_val2, step_grad_val2, index2, checkpoints
        )
        step_val2, step_grad_val2, step_next_checkpoint2, index2 = _assert_unbatched(
            step_val2, step_grad_val2, step_next_checkpoint2, index2
        )
        return (
            step_val2,
            step_grad_val2,
            step_next_checkpoint2,
            index2,
            val2,
            grad_val2,
            grad_body_fun2,
        )

    return _u_turn


def _checkpointed_while_loop_bwd(
    remainders, grad_final_val, vjp_arg, cond_fun, max_steps, checkpoints
):
    """Time for the complicated bit: iterate backward through a checkpointed while loop,
    loading values from checkpoints and using treeverse to toggle between forward and
    backward steps.
    """
    _, body_fun = vjp_arg
    grad_final_body_fun = jtu.tree_map(
        lambda x: jnp.zeros_like(x) if is_inexact_array(x) else None, body_fun
    )
    del cond_fun, max_steps
    num_steps, init_residual_steps, init_residuals = remainders
    (init_residual_steps,) = _assert_unbatched(init_residual_steps)

    def _cond_fun(carry):
        _, step_grad_val, *_ = carry
        # step_grad_val is the location of our cotangent. We want to keep going until
        # this has got all the way to the start.
        return unvmap_any(step_grad_val > 0)

    def _body_fun(carry):
        (
            step_val,
            step_grad_val,
            step_next_checkpoint,
            index,
            val,
            grad_val,
            grad_body_fun,
            residual_steps,
            residuals,
        ) = carry
        step_val, step_next_checkpoint, index, residual_steps = _assert_unbatched(
            step_val, step_next_checkpoint, index, residual_steps
        )

        msg = (
            "Internal run-time error when backpropagating through "
            "`equinox.internal.checkpointed_while_loop`. "
            "Please raise an issue at https://http://github.com/patrick-kidger/equinox"
        )
        step_val = error_if(step_val, step_val >= unvmap_max(step_grad_val), msg)

        #
        # First either propagate our primal state forward, or make a U-turn if the
        # primal state has caught up to the cotangent state.
        #

        perform_u_turn = step_val + 1 == unvmap_max(step_grad_val)
        (perform_u_turn,) = _assert_unbatched(perform_u_turn)
        (
            step_val2,
            step_grad_val2,
            step_next_checkpoint2,
            index2,
            val2,
            grad_val2,
            grad_body_fun2,
        ) = lax.cond(
            perform_u_turn,
            _make_u_turn(residual_steps, residuals, checkpoints),
            _fwd,
            body_fun,
            step_val,
            unvmap_max(step_grad_val),
            step_next_checkpoint,
            index,
            val,
            grad_val,
            grad_body_fun,
        )

        #
        # Second, decide whether to store our current primal state in a checkpoint.
        # Note that this can only actually trigger on `_fwd` and not on `_u_turn`, as
        # if `_u_turn` happens then `step_val2 < step_next_checkpoint2`, but
        # `_maybe_save_to_checkpoint` has a `step_val2 == step_next_checkpoint2` check.
        #
        # Nonetheless these operations have to happen outside of the `lax.cond`, so that
        # the in-place update is the final operation happening to `residuals` and
        # `residual_steps` in the body function. This is needed to ensure that the copy
        # get elided when under vmap. See JAX issue #13522.
        #

        (
            index2,
            step_next_checkpoint2,
            residual_steps2,
            residuals2,
        ) = _maybe_save_to_checkpoint(
            step_val2,
            unvmap_max(step_grad_val2),
            step_next_checkpoint2,
            index2,
            val2,
            residual_steps,
            residuals,
            checkpoints,
        )

        #
        # Third, handle batching appropriately. This means resetting the cotangent
        # computation until the overall computation has got to this batch element.
        #
        # In particular it does *not* mean resetting the primal computation. We need to
        # perform primal computations even before `pred == True`, in order to recover
        # some intermediate primal states in those `_fwd` steps that occur immediately
        # prior to the first U-turn which affects this batch element.
        #

        pred = step_grad_val == unvmap_max(step_grad_val)
        step_grad_val2, grad_val2, grad_body_fun2 = jtu.tree_map(
            lambda v, v2: lax.select(pred, v2, v),
            (step_grad_val, grad_val, grad_body_fun),
            (step_grad_val2, grad_val2, grad_body_fun2),
        )
        step_val2, step_next_checkpoint2, index2, residual_steps2 = _assert_unbatched(
            step_val2, step_next_checkpoint2, index2, residual_steps2
        )
        return (
            step_val2,
            step_grad_val2,
            step_next_checkpoint2,
            index2,
            val2,
            grad_val2,
            grad_body_fun2,
            residual_steps2,
            residuals2,
        )

    # We can index into our residuals using 0, 1, ..., checkpoints - 1.
    # `index` is used to refer to the next empty spot, so it takes values in
    # 0, 1, ..., checkpoints - 1, checkpoints, where `index == checkpoints` indicates
    # that there are no empty spots and the whole buffer is full. (And this is used in
    # `_calc_step_next_checkpoint`.)
    init_index = jnp.minimum(unvmap_max(num_steps), checkpoints)
    init_step_grad_val = num_steps
    init_step_val, init_val, init_index = _load_from_checkpoint(
        unvmap_max(init_step_grad_val), init_index, init_residual_steps, init_residuals
    )
    init_step_next_checkpoint = _calc_next_checkpoint(
        init_step_val, unvmap_max(init_step_grad_val), init_index, checkpoints
    )
    init_carry = (
        init_step_val,
        init_step_grad_val,
        init_step_next_checkpoint,
        init_index,
        init_val,
        grad_final_val,
        grad_final_body_fun,
        init_residual_steps,
        init_residuals,
    )
    # Note that the checkpoint buffers hold both values computed on the forward pass,
    # and checkpoints recomputed on the backward pass.
    #
    # Controller State
    # ----------------
    # `step_val`: see `val`.
    # `step_grad_val`: see `grad_val`. Note that there is an invariant
    #   `step_val + 1 <= step_grad_val`.
    # `step_next_checkpoint`: the step at which we next need to save a checkpoint, for
    #   the recomputed forward computation.
    # `index` is the index of the memory buffer to save the next checkpoint, for the
    #   recomputed forward computation.
    #
    # Numerical computations
    # ----------------------
    # `val` is the evolving state of the forward loop (reloaded from checkpoints). The
    #   forward loop step that this comes from is `step_val`. As such `val` and
    #   `step_val` will jump back-and-forth as values are loaded from checkpoints and
    #   then recomputed forward using `_fwd`.
    # `grad_val` is the cotangent that we're propagating backward. The step of the
    #   forward loop (that it holds the cotangent for) is given in `step_grad_val`. As
    #   such `step_grad_val` decrements weakly monotonically, and `grad_val` is updated
    #   on every U-turn.
    # `grad_body_fun` is the cotangent being accumulated for `body_fun`. It updates on
    #   every U-turn.
    #
    # Buffers
    # -------
    # `residual_steps` is the buffer of the `step` for each checkpoint
    # `residuals` is the buffer of the `val` for each checkpoint

    final_carry = lax.while_loop(_cond_fun, _body_fun, init_carry)
    *_, grad_init_val, grad_body_fun, _, _ = final_carry
    out = grad_init_val, grad_body_fun
    # I think combining higher-order autodifferentiation with treeverse is an open
    # problem? Probably JAX can differentiate through this but it'll be really
    # inefficient, so to be safe we disable it for now.
    msg = "`checkpointed_while_loop` is only first-order autodifferentiable"
    out = nondifferentiable(out, msg=msg)
    return out


_checkpointed_while_loop.defvjp(
    _checkpointed_while_loop_fwd, _checkpointed_while_loop_bwd
)
