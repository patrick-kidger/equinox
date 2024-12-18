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
# I think this code is not *quite* maximally efficient. A few things that could be
# improved:
# - The initial value is available on the backward pass twice: once as an argument,
#   once as a saved checkpoint. We should be able to get away without this repetition.
# - We only implement Algorithm I of Stumm--Wather. Additionally implementing
#   Algorithm II would be worthwhile. (But finickity, as their description of it in the
#   paper leaves something to be desired. And may also have an off-by-one-error, like
#   their Figure 2.2 does?)

import functools as ft
import math
import operator
import typing
from collections.abc import Callable, Sequence
from typing import Any, cast, Optional, TypeVar, Union

import jax
import jax.core
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, Bool

from ..._ad import filter_closure_convert, filter_custom_jvp, filter_custom_vjp
from ..._errors import error_if
from ..._filters import combine, filter, is_array, is_inexact_array, partition
from ..._module import Static
from ..._tree import tree_at, tree_equal
from .._nontraceable import nonbatchable
from .common import common_rewrite, fixed_asarray


_T = TypeVar("_T")
_Bool = Union[bool, Bool[Array, ""]]
_Node = Any


def checkpointed_while_loop(
    cond_fun: Callable[[_T], _Bool],
    body_fun: Callable[[_T], _T],
    init_val: _T,
    *,
    max_steps: Optional[int] = None,
    buffers: Optional[Callable[[_T], Union[_Node, Sequence[_Node]]]] = None,
    checkpoints: Optional[int] = None,
) -> _T:
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
    - `max_steps`: A bound on the maximum number of steps. Set to `None` to allow an
        arbitrary number of steps. (`checkpointed_while_loop` is reverse-mode
        autodifferentiable regardless of whether `max_steps` is finite.)
    - `checkpoints`: The number of steps at which to checkpoint. The memory consumed
        will be that of `checkpoints`-many copies of `init_val`. (As the state is
        updated throughout the loop.)
    - `buffers`: If passed, then every array in `tree_leaves(buffers(init_val))` will
        become a write-only buffer. (Supporting only `.at[].set()`.)

    **Returns:**

    The final value; as `lax.while_loop`.

    !!! Info

        This function is not forward-mode autodifferentiable.

    !!! Info

        `buffers` is useful in the following way.

        Recall that a checkpointed while loop works by storing a copy of the evolving
        value (of the same structure as `init_val`). However, if part of that value is
        a large buffer than is being progressively written in to as the loop progresses,
        then we will end up with multiple copies of this large buffer.

        As this slows things down, we offer a special API specifically to handle this
        case.

    !!! Danger

        Note that `buffers` is subject to the following restrictions:

        - You should never write to the same location twice.
        - You should only read from it (`buf[i]`) at locations (`i`) that you have
          written to previously.

        These assumptions are *completely unchecked* and you will get incorrect
        gradients if you violate these assumptions.

    ??? cite "References"

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
        "recursive binary checkpointing", "revolve" etc.) was developed in:

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
    if checkpoints is None:
        if max_steps is None:
            raise ValueError(
                "Must specify either `max_steps` or `checkpoints` in "
                "`equinox.internal.checkpointed_while_loop`."
            )
        # Binomial logarithmic growth is what is needed in classical treeverse.
        #
        # If
        # `max_steps < (checkpoints + 1)(checkpoints + 2)/2`
        # then the time spend recomputing will be optimised; see equation (2.2) of
        # "New Algorithms for Optimal Online Checkpointing", Stumm and Walther 2010.
        # https://tu-dresden.de/mn/math/wir/ressourcen/dateien/forschung/publikationen/pdf2010/new_algorithms_for_optimal_online_checkpointing.pdf
        # and also the `_should_save_residuals` function below.
        #
        # So by default we use `checkpoints = O(sqrt(max_steps))`.
        # (Classical treeverse uses only `O(log(max_steps))`, of course, but doesn't
        # handle the online case.)
        if max_steps == 1:
            checkpoints = 1
        else:
            checkpoints = math.floor(-1.5 + math.sqrt(2 * max_steps + 0.25)) + 1
            assert 2 * max_steps < ((checkpoints + 1) * (checkpoints + 2))
    if checkpoints < 1:
        raise ValueError("Must have at least one checkpoint")
    init_val = jtu.tree_map(fixed_asarray, init_val)
    if max_steps == 0:
        return init_val
    if max_steps is not None:
        checkpoints = min(checkpoints, max_steps)
    cond_fun_, body_fun_, init_val_, buffers_ = common_rewrite(
        cond_fun, body_fun, init_val, max_steps, buffers, makes_false_steps=False
    )
    del cond_fun, body_fun, init_val, buffers
    cond_fun_ = filter_closure_convert(cond_fun_, init_val_)
    cond_fun_ = jtu.tree_map(_stop_gradient, cond_fun_)
    body_fun_ = filter_closure_convert(body_fun_, init_val_)
    vjp_arg = (init_val_, body_fun_)
    final_val_ = _checkpointed_while_loop(
        vjp_arg, cond_fun_, checkpoints, buffers_, max_steps
    )
    _, _, _, final_val = _stop_gradient_on_unperturbed(init_val_, final_val_, body_fun_)
    return final_val


def _stop_gradient(x):
    if is_array(x):
        return lax.stop_gradient(x)
    else:
        return x


@filter_custom_vjp
def _checkpointed_while_loop(vjp_arg, cond_fun, checkpoints, buffers, max_steps):
    """Uncheckpointed forward used when not differentiating."""
    del checkpoints, buffers, max_steps
    init_val, body_fun = vjp_arg
    while_loop = jax.named_call(lax.while_loop, name="checkpointed-no-vjp")
    # Hashable wrapper; JAX issue #13554 and
    # https://github.com/patrick-kidger/equinox/issues/768
    return while_loop(lambda x: cond_fun(x), lambda x: body_fun(x), init_val)


def _scalar_index(i, x):
    """As `x[i]`, but slightly more efficient for a nonnegative scalar `i`.

    (As it avoids support for negative indexing, and lowers to `dynamic_slice` rather
     than `gather`.)
    """
    assert jnp.shape(i) == ()
    return lax.dynamic_index_in_dim(x, i, keepdims=False)


def _unique_index(i, x):
    """As `x[i]`, but states that `i` has unique indices."""
    return x.at[i].get(unique_indices=True)


def _stumm_walther_i(step, save_state):
    """Algorithm 1 from:

    Stumm and Walther 2010
    New Algorithms for Optimal Online Checkpointing
    https://tu-dresden.de/mn/math/wir/ressourcen/dateien/forschung/publikationen/pdf2010/new_algorithms_for_optimal_online_checkpointing.pdf
    """
    step, save_state = nonbatchable((step, save_state))
    i, o, p, s = save_state
    i = cast(ArrayLike, i)
    s = cast(ArrayLike, s)
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
        "Please raise an issue at https://github.com/patrick-kidger/equinox"
    )
    if getattr(typing, "TESTING", False):
        out = error_if(out, pred & (o == -1), msg)
    out = nonbatchable(out)
    return out


def _any_dispensable(dispensable, residual_steps: Array, levels):
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
    step, save_state, residual_steps = nonbatchable((step, save_state, residual_steps))
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
    out = nonbatchable(out)
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


def _should_save_residual(
    step, save_state, residual_steps, u2_minus_1, checkpoints, max_steps
):
    """This is the controller for whether we should save the current value at each step,
    and if so which memory location to save it in.
    """
    # TODO: also implement Algorithm 2 of Stumm and Walther, which gives improved
    # results for u2 < step < u3.

    if checkpoints == max_steps:
        # Important special case! We don't need to do any computations in this case.
        # This is a measurable performance optimisation.
        save_residual = True
        index = step
        save_state2 = save_state
    else:
        if max_steps is not None and max_steps <= u2_minus_1:
            # `step < max_steps` implies `step + 1 <= max_steps`. If it is also the case
            # that `max_steps <= u2 - 1` then overall we know that `step <= u2 - 2`,
            # which is the condition for using Stumm--Walther.
            #
            # Skipping this cond offers a measurable performance optimisation.
            pred = False
        else:
            step, u2_minus_1 = nonbatchable((step, u2_minus_1))
            pred = step >= u2_minus_1
        save_residual, index, save_state2 = lax.cond(
            pred,
            _wang_moin_wrapper,
            _stumm_walther_i_wrapper,
            step,
            save_state,
            residual_steps,
        )
    return save_residual, index, save_state2


def _unreachable_checkpoint_step(x):
    """Dummy value used to represent a checkpoint we never reach."""
    dtype = jnp.result_type(x)  # x can be a dtype or an arraylike
    return jnp.iinfo(dtype).max


def _array_to_none(x):
    assert is_array(x)
    return None


@_checkpointed_while_loop.def_fwd
def _checkpointed_while_loop_fwd(
    perturbed, vjp_arg, cond_fun, checkpoints, buffers, max_steps
):
    """Run the while loop, saving checkpoints whenever the controller
    (`_should_save_residual`) requires.
    """
    del perturbed
    init_val, body_fun = vjp_arg
    # Equation (2.2) of Stumm and Walther
    u2_minus_1 = ((checkpoints + 1) * (checkpoints + 2)) // 2 - 1

    def _cond_fun(carry):
        _, _, val, _, _ = carry
        return cond_fun(val)

    def _body_fun(carry):
        step, save_state, val, residual_steps, residuals = carry
        save_state, residual_steps = nonbatchable((save_state, residual_steps))

        step2 = step + 1
        save_residual, index, save_state2 = _should_save_residual(
            step, save_state, residual_steps, u2_minus_1, checkpoints, max_steps
        )
        val2 = body_fun(val)

        def _maybe_update(xs, x):
            where_x = jnp.where(save_residual, x, _scalar_index(index, xs))
            where_x = cast(Array, where_x)
            return lax.dynamic_update_index_in_dim(xs, where_x, index, axis=0)

        val_no_buffers = tree_at(buffers(None), val, replace_fn=_array_to_none)
        residual_steps2, residuals2 = jtu.tree_map(
            _maybe_update,
            (residual_steps, residuals),
            (step, val_no_buffers),
        )
        save_state2, residual_steps2 = nonbatchable((save_state2, residual_steps2))
        return step2, save_state2, val2, residual_steps2, residuals2

    int_dtype = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32  # pyright: ignore
    init_step = jnp.array(0, dtype=int_dtype)  # dtype matches init_residual_steps
    init_save_state_sw_i = 0, checkpoints, checkpoints, True
    dtype_max = jnp.iinfo(int_dtype).max  # pyright: ignore
    init_save_state_wm = (
        jnp.zeros(checkpoints, dtype=int_dtype).at[0].set(dtype_max),
        jnp.full((checkpoints,), False),
    )
    init_save_state = (init_save_state_sw_i, init_save_state_wm)
    # Uses the fact that `_unreachable_checkpoint_step` returns intmax, so that in our
    # sorting later, in the steps < checkpoints case, all unused memory gets sorted
    # to the end.
    init_residual_steps = jnp.full(
        checkpoints, _unreachable_checkpoint_step(int_dtype), dtype=int_dtype
    )

    init_val_no_buffers = tree_at(buffers(None), init_val, replace_fn=_array_to_none)
    # Fill value for the memory isn't important.
    init_residuals = jtu.tree_map(
        lambda x: jnp.zeros((checkpoints,) + x.shape, x.dtype), init_val_no_buffers
    )
    init_carry = (
        init_step,
        init_save_state,
        init_val,
        init_residual_steps,
        init_residuals,
    )
    # `step` is an increasing counter 0, 1, 2, 3, ...
    # `save_state` is state used by the logic for whether to save a checkpoint each step
    # `val` is the evolving state of the loop
    # `residual_steps` is the buffer of the `step` for each checkpoint
    # `residuals` is the buffer of the `val` for each checkpoint

    while_loop = jax.named_call(lax.while_loop, name="checkpointed-fwd")
    final_carry = while_loop(_cond_fun, _body_fun, init_carry)
    num_steps, _, final_val, final_residual_steps, final_residuals = final_carry
    filled_buffers = buffers(_is_none)(final_val)

    # The above procedure may produce residuals saved in jumbled-up order. Meanwhile
    # treeverse (used on the backward pass) treats the residuals like a stack,
    # reading and writing the most recent residual to and from the end. So sort the
    # residuals we've produced here to obtain the desired invariant, i.e. that the
    # residuals are in order.
    if checkpoints != max_steps:
        # If `checkpoints == max_steps` then residuals are already sorted.
        sort_indices = jnp.argsort(final_residual_steps)
        final_residual_steps, final_residuals = jtu.tree_map(
            ft.partial(_unique_index, sort_indices),
            (final_residual_steps, final_residuals),
        )
    num_steps, final_residual_steps = nonbatchable((num_steps, final_residual_steps))
    return final_val, (num_steps, final_residual_steps, final_residuals, filled_buffers)


def _load_from_checkpoint(
    step_grad_val, index, residual_steps, residuals, checkpoints, max_steps
):
    """Loads a residual from the store of checkpoints."""
    # step_grad_val is the current location of grad_val.
    # index is the next currently empty slot for saving a residual.
    step_grad_val, index, residual_steps = nonbatchable(
        (step_grad_val, index, residual_steps)
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
    if checkpoints == max_steps:
        # Known to be symbolically true in this case; this is a slight performance
        # optimisation.
        pred = True
    else:
        pred = step_val2 + 1 == step_grad_val
    index2 = jnp.where(pred, read_index, index)
    step_val2, index2 = nonbatchable((step_val2, index2))
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
    ) = nonbatchable(
        (step_val, step_grad_val, step_next_checkpoint, index, residual_steps)
    )
    save_checkpoint = step_val == step_next_checkpoint

    def _maybe_update(xs, x):
        where_x = jnp.where(save_checkpoint, x, _scalar_index(index, xs))
        where_x = cast(Array, where_x)
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
    index2, step_next_checkpoint2, residual_steps2 = nonbatchable(
        (index2, step_next_checkpoint2, residual_steps2)
    )
    return index2, step_next_checkpoint2, residual_steps2, residuals2


def _calc_next_checkpoint(step_val, step_grad_val, index, checkpoints):
    """Determines the step at which we next want to save a checkpoint."""
    # Note that when this function is called, `step_val` is always at the most recent
    # checkpoint.
    step_val, step_grad_val, index = nonbatchable((step_val, step_grad_val, index))

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
    step_next_checkpoint = nonbatchable(step_next_checkpoint)
    return step_next_checkpoint


def _is_none(x):
    return x is None


def _get_value(x):
    assert type(x) is jax.custom_derivatives.CustomVJPPrimal
    return x.value


def _is_symbolic_zero_or_none(x):
    return isinstance(x, jax.custom_derivatives.SymbolicZero) or x is None


def _not_symbolic_zero(x):
    return not isinstance(x, jax.custom_derivatives.SymbolicZero)


def _materialise_symbolic_zero(is_zero, ct_x, x):
    if is_zero:
        assert ct_x is None
        return None
    else:
        if ct_x is None:
            assert is_inexact_array(x)
            return jnp.zeros_like(x)
        else:
            return ct_x


def _assert_bool(x):
    assert type(x) is bool


def _fwd(
    step_val,
    step_grad_val,
    step_next_checkpoint,
    index,
    val_candidate,
    grad_val,
    grad_body_fun,
):
    """Propagates the primal forward one step."""
    step_val2 = nonbatchable(step_val + 1)
    return (
        step_val2,
        step_grad_val,
        step_next_checkpoint,
        index,
        val_candidate,
        grad_val,
        grad_body_fun,
    )


def _make_u_turn(vjp_fn, residual_steps, residuals, checkpoints, max_steps):
    """Propagates the cotangent backward one step."""
    residual_steps = nonbatchable(residual_steps)

    def _u_turn(
        step_val,
        step_grad_val,
        step_next_checkpoint,
        index,
        val_candidate,
        grad_val,
        grad_body_fun,
    ):
        del step_val, step_next_checkpoint
        step_grad_val, index = nonbatchable((step_grad_val, index))
        grad_body_fun_update, grad_val2 = vjp_fn(grad_val)
        grad_body_fun2 = jtu.tree_map(operator.add, grad_body_fun, grad_body_fun_update)
        step_grad_val2 = step_grad_val - 1
        step_val2, val2, index2 = _load_from_checkpoint(
            step_grad_val2, index, residual_steps, residuals, checkpoints, max_steps
        )
        step_next_checkpoint2 = _calc_next_checkpoint(
            step_val2, step_grad_val2, index2, checkpoints
        )
        step_val2, step_grad_val2, step_next_checkpoint2, index2 = nonbatchable(
            (step_val2, step_grad_val2, step_next_checkpoint2, index2)
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


@_checkpointed_while_loop.def_bwd
def _checkpointed_while_loop_bwd(
    remainders,
    grad_final_val,
    perturbed,
    vjp_arg,
    cond_fun,
    checkpoints,
    buffers,
    max_steps,
):
    """Time for the complicated bit: iterate backward through a checkpointed while loop,
    loading values from checkpoints and using treeverse to toggle between forward and
    backward steps.
    """
    perturb_init_val, perturb_body_fun = perturbed
    _, body_fun = vjp_arg
    grad_final_body_fun = jtu.tree_map(
        lambda x, p: jnp.zeros_like(x) if p else None, body_fun, perturb_body_fun
    )
    del cond_fun, perturbed
    num_steps, init_residual_steps, init_residuals, filled_buffers = remainders
    # `allow_constant_across_batch` to handle
    # https://github.com/patrick-kidger/optimistix/issues/67
    num_steps, init_residual_steps = nonbatchable(
        (num_steps, init_residual_steps), allow_constant_across_batch=True
    )

    def _cond_fun(carry):
        _, step_grad_val, *_ = carry
        # step_grad_val is the location of our cotangent. We want to keep going until
        # this has got all the way to the start.
        step_grad_val = nonbatchable(step_grad_val)
        return step_grad_val > 0

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
        (
            step_val,
            step_grad_val,
            step_next_checkpoint,
            index,
            residual_steps,
        ) = nonbatchable(
            (step_val, step_grad_val, step_next_checkpoint, index, residual_steps)
        )

        msg = (
            "Internal run-time error when backpropagating through "
            "`equinox.internal.checkpointed_while_loop`. "
            "Please raise an issue at https://github.com/patrick-kidger/equinox"
        )
        if getattr(typing, "TESTING", False):
            step_val = error_if(step_val, step_val >= step_grad_val, msg)

        #
        # First either propagate our primal state forward, or make a U-turn if the
        # primal state has caught up to the cotangent state.
        #

        # Use `filter_vjp` to neatly handle floating-point arrays.
        #
        # We pass in `body_fun` as an argument as it contains its closed-over values
        # in its PyTree structure, and we do want to compute cotangents wrt these.
        val_buffers = tree_at(buffers(_is_none), val, filled_buffers, is_leaf=_is_none)
        grad_val_has_ct = jtu.tree_map(is_array, grad_val, is_leaf=_is_none)
        body_fun_has_ct = jtu.tree_map(is_array, grad_body_fun, is_leaf=_is_none)
        diff_val, nondiff_val = partition(val_buffers, grad_val_has_ct)
        diff_body, nondiff_body = partition(body_fun, body_fun_has_ct)

        def _to_vjp(_diff_body, _diff_val):
            _val_buffers = combine(_diff_val, nondiff_val)
            _body_fun = combine(_diff_body, nondiff_body)
            _out = _body_fun(_val_buffers)
            return partition(_out, grad_val_has_ct)

        dynamic_val_candidate, vjp_fn, static_val_candidate = jax.vjp(
            _to_vjp, diff_body, diff_val, has_aux=True
        )
        val_candidate = combine(dynamic_val_candidate, static_val_candidate)
        val_candidate = tree_at(buffers(None), val_candidate, replace_fn=lambda _: None)

        if checkpoints == max_steps:
            # Skip calculating this at runtime; we know it must be true.
            # This means we can inline the U-turn branch of the `cond`.
            perform_u_turn = True
        else:
            perform_u_turn = step_val + 1 == step_grad_val
            perform_u_turn = nonbatchable(perform_u_turn)
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
            _make_u_turn(vjp_fn, residual_steps, residuals, checkpoints, max_steps),
            _fwd,
            step_val,
            step_grad_val,
            step_next_checkpoint,
            index,
            val_candidate,
            grad_val,
            grad_body_fun,
        )

        #
        # Second, decide whether to store our current primal state in a checkpoint.
        # Note that this can only actually trigger on `_fwd` and not on `_u_turn`, as
        # if `_u_turn` happens then `step_val2 < step_next_checkpoint2`, but
        # `_maybe_save_to_checkpoint` has a `step_val2 == step_next_checkpoint2` check.
        # (We could maybe move this inside `_fwd`? It was originally outside for
        # efficiency reasons that I don't think apply any more.)
        #

        if checkpoints == max_steps:
            # In this case we never need to save any extra checkpoints on the backward
            # pass, so skip this.
            residual_steps2 = residual_steps
            residuals2 = residuals
        else:
            (
                index2,
                step_next_checkpoint2,
                residual_steps2,
                residuals2,
            ) = _maybe_save_to_checkpoint(
                step_val2,
                step_grad_val2,
                step_next_checkpoint2,
                index2,
                val2,
                residual_steps,
                residuals,
                checkpoints,
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
    init_index = jnp.minimum(num_steps, checkpoints)
    init_step_grad_val = num_steps
    init_step_val, init_val, init_index = _load_from_checkpoint(
        init_step_grad_val,
        init_index,
        init_residual_steps,
        init_residuals,
        checkpoints,
        max_steps,
    )
    init_step_next_checkpoint = _calc_next_checkpoint(
        init_step_val, init_step_grad_val, init_index, checkpoints
    )

    #
    # Okay, this next bit is a bit complicated.
    # We need to figure out exactly which parts of the carry that we're going to compute
    # cotangents for.
    # There are three interacting pieces:
    # (1) Only some inputs even need their cotangents computed (those that are marked
    #     as perturbed). We should avoid computing intermediate cotangents that aren't
    #     needed.
    # (2) Some cotangents may be symbolic zeros. We should avoid materialising any that
    #     don't need to be materialised.
    # (3) Some parts of the body function may not be differentiable. (E.g. they call to
    #     `eqxi.nondifferentiable` or `eqxi.nondifferentiable_backward`.) In particular,
    #     some inexact arrays may be marked with these. (Anything else is automatically
    #     nondifferentiable.) As these raise an error eagerly, we can't just take an
    #     approach of "differentiate everything, then DCE".
    #
    # Our strategy is as follows. First, consider those inputs that are perturbed, and
    # iterate to find all the carries that become perturbed as a result. We obtain a
    # fixed point.
    # * This mimics what JAX does in JVP rules for scans/whiles. And in fact we also
    #   track perturbation information forward via JVPs (which input tangents affect
    #   which output tangents), rather than backward via VJPs (which output cotangents
    #   affect which input cotangents), as this is what JAX does here and we'd like to
    #   be consistent. (Also, it seems like there might be a catch-22 trying to do this
    #   with VJPs -- how do you decide which inputs to perturb in your VJP?) This is
    #   basically handling criteria (1).
    # * This also means that we're perturbing the minimal amount possible. We don't even
    #   run the JVP rule for something that isn't differentiated. This is part of
    #   addressing criteria (3).
    #
    # Second, we ignore all cotangents (of our forward output) that aren't on a
    # perturbed carry. Clearly they're not going to affect anything. So we switch them
    # to symbolic zeros.
    # * This is part of addressing criteria (3) -- we avoid propagating cotangents
    #   that we don't need to. Maybe there's an `eqxi.nondifferentiable_backward` in
    #   there.
    #
    # Third, we find another fixed point, this time for the symbolic zeros. Track which
    # cotangents affect which cotangents, and materialise the minimal amount possible.
    # * This addresses criteria (2).
    # * Note that symbolic zeros are doing double-duty here. We avoid materialising them
    #   for simple efficiency reasons, sure -- but we also want to keep as many
    #   unmaterialised as possible, in case there's an `eqxi.nondifferentiable_backward`
    #   in there, which only accepts symbolic zero cotangents.
    #
    # Phew! At that point, job done: we know which cotangents we're propagating and so
    # we can run our loop.
    #
    # The first fixed point basically mimics what JAX does in the JVP rule for a while
    # or scan, figuring out which values are perturbed. The second fixed point basically
    # mimics what JAX does in the transpose rule of a scan, figuring out which
    # cotangents are symbolic zeros.
    #

    # Do this inside the dynamic context of a `jax.eval_shape` so that it's as cheap as
    # possible, without extra tracing.
    def _resolve_perturb_val():
        # TODO: is this function still necessary now that we have
        # _stop_gradient_on_unperturbed_jvp?
        init_val_buffers = tree_at(
            buffers(_is_none), init_val, filled_buffers, is_leaf=_is_none
        )
        perturb_val = perturb_init_val
        assert jtu.tree_structure(init_val_buffers) == jtu.tree_structure(perturb_val)

        while True:
            # `body_fun` is included so that we also track perturbatations on that.
            perturb_pair = (perturb_body_fun, perturb_val)
            dynamic, static = partition((body_fun, init_val_buffers), perturb_pair)
            new_perturb_val = sentinel = object()

            @jax.custom_jvp
            def _record_symbolic_zeros(_dynamic_out):
                return _dynamic_out

            def _record_symbolic_zeros_jvp(primals, tangents):
                (primals,) = primals
                (tangents,) = tangents
                nonlocal new_perturb_val
                new_perturb_val = jtu.tree_map(_not_symbolic_zero, tangents)
                return primals, tangents

            _record_symbolic_zeros.defjvp(
                _record_symbolic_zeros_jvp, symbolic_zeros=True
            )

            def _to_linearize(_dynamic):
                _body_fun, _val = combine(_dynamic, static)
                _out = _body_fun(_val)
                _dynamic_out, _static_out = partition(_out, is_inexact_array)
                _dynamic_out = _record_symbolic_zeros(_dynamic_out)
                _out = combine(_dynamic_out, _static_out)
                return _out

            # Not `jax.jvp`, so as not to error if `body_fun` has any `custom_vjp`s.
            jax.linearize(_to_linearize, dynamic)
            assert new_perturb_val is not sentinel
            assert jtu.tree_structure(
                perturb_val, is_leaf=_is_none
            ) == jtu.tree_structure(new_perturb_val, is_leaf=_is_none)
            new_perturb_val = jtu.tree_map(
                lambda x, y: False if (x is False and y is None) else y,
                perturb_val,
                new_perturb_val,
            )
            assert jtu.tree_structure(perturb_val) == jtu.tree_structure(
                new_perturb_val
            )
            if tree_equal(perturb_val, new_perturb_val):
                jtu.tree_map(_assert_bool, perturb_val)
                if getattr(typing, "TESTING", False):
                    print("perturb_val", perturb_val)
                return Static(perturb_val)
            else:
                perturb_val = jtu.tree_map(operator.or_, perturb_val, new_perturb_val)

    # Find a fixed point of which values we need to perturb. (Not the same as whether
    # or not they have symbolic zero cotangents! All unperturbed values must have
    # symbolic zero cotangents, but some perturbed values may have these as well.)
    perturb_val = jax.eval_shape(_resolve_perturb_val).value

    # Do this inside the dynamic context of a `jax.eval_shape` so that it's as cheap as
    # possible, without extra tracing.
    def _resolve_symbolic_zeros(grad_val):
        symbolic_zero_gradient = jtu.tree_map(_is_none, grad_val, is_leaf=_is_none)
        init_val_buffers = tree_at(
            buffers(_is_none), init_val, filled_buffers, is_leaf=_is_none
        )
        dynamic_init_val, static_init_val = partition(init_val_buffers, perturb_val)

        while True:
            new_symbolic_zero_gradient = sentinel = object()

            # Some hackery to extract which inputs still have symbolic zero cotangents.
            @jax.custom_vjp
            def _record_symbolic_zero(_dynamic_val):
                return _dynamic_val

            def _record_symbolic_zero_fwd(_dynamic_val):
                return jtu.tree_map(_get_value, _dynamic_val), None

            def _record_symbolic_zero_bwd(_, grad_dynamic_val):
                nonlocal new_symbolic_zero_gradient
                new_symbolic_zero_gradient = jtu.tree_map(
                    _is_symbolic_zero_or_none, grad_dynamic_val, is_leaf=_is_none
                )
                return (grad_dynamic_val,)

            _record_symbolic_zero.defvjp(
                _record_symbolic_zero_fwd,
                _record_symbolic_zero_bwd,
                symbolic_zeros=True,
            )

            def _to_vjp(_dynamic_val):
                _dynamic_val = _record_symbolic_zero(_dynamic_val)
                val = combine(_dynamic_val, static_init_val)
                return filter(body_fun(val), symbolic_zero_gradient, inverse=True)

            _, vjp_fn = jax.vjp(_to_vjp, dynamic_init_val)
            vjp_fn(grad_val)
            assert new_symbolic_zero_gradient is not sentinel
            if tree_equal(symbolic_zero_gradient, new_symbolic_zero_gradient):
                jtu.tree_map(_assert_bool, symbolic_zero_gradient)
                if getattr(typing, "TESTING", False):
                    print("symbolic_zero_gradient", symbolic_zero_gradient)
                return Static(symbolic_zero_gradient)
            else:
                symbolic_zero_gradient = jtu.tree_map(
                    operator.and_, symbolic_zero_gradient, new_symbolic_zero_gradient
                )
                grad_val = jtu.tree_map(
                    _materialise_symbolic_zero,
                    symbolic_zero_gradient,
                    grad_val,
                    init_val_buffers,
                )

    grad_final_val = filter(grad_final_val, perturb_val)
    # If all provided cotangents end up being irrelevant -- i.e. the perturbed inputs
    # depend only on those outputs which have symbolic zero cotangents -- then we can
    # skip the whole computation.
    if len(jtu.tree_leaves(grad_final_val)) == 0:
        return jtu.tree_map(lambda _: None, (grad_final_val, body_fun))
    symbolic_zero_gradient = jax.eval_shape(
        _resolve_symbolic_zeros, grad_final_val
    ).value
    init_val_buffers = tree_at(
        buffers(_is_none), init_val, filled_buffers, is_leaf=_is_none
    )
    grad_final_val = jtu.tree_map(
        _materialise_symbolic_zero,
        symbolic_zero_gradient,
        grad_final_val,
        init_val_buffers,
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
    # Note that the saved checkpoints hold both (a) values computed on the forward
    # pass, and (b) checkpoints recomputed on the backward pass. (We don't need to
    # distinguish them.)
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
    # Checkpoints
    # -------
    # `residual_steps` is the memory holding the `step` for each checkpoint
    # `residuals` is the memory holding the `val` for each checkpoint

    # Not reverse mode autodifferentiable, but it is forward-mode autodifferentiable!
    # That means we can do Hessians, at least.
    while_loop = jax.named_call(lax.while_loop, name="checkpointed-bwd")
    final_carry = while_loop(_cond_fun, _body_fun, init_carry)
    *_, grad_init_val, grad_body_fun, _, _ = final_carry
    out = grad_init_val, grad_body_fun
    return out


def _resolve_perturb_val(final_val, body_fun, perturb_final_val, perturb_body_fun):
    def _resolve_perturb_val_impl():
        perturb_val = perturb_final_val
        assert jtu.tree_structure(final_val) == jtu.tree_structure(perturb_val)

        while True:
            # `body_fun` is included so that we also track perturbatations on that.
            perturb_pair = (perturb_body_fun, perturb_val)
            dynamic, static = partition((body_fun, final_val), perturb_pair)
            new_perturb_val = sentinel = object()

            @jax.custom_jvp
            def _record_symbolic_zeros(_dynamic_out):
                return _dynamic_out

            def _record_symbolic_zeros_jvp(primals, tangents):
                (primals,) = primals
                (tangents,) = tangents
                nonlocal new_perturb_val
                new_perturb_val = jtu.tree_map(_not_symbolic_zero, tangents)
                return primals, tangents

            _record_symbolic_zeros.defjvp(
                _record_symbolic_zeros_jvp, symbolic_zeros=True
            )

            def _to_linearize(_dynamic):
                _body_fun, _val = combine(_dynamic, static)
                _out = _body_fun(_val)
                _dynamic_out, _static_out = partition(_out, is_inexact_array)
                _dynamic_out = _record_symbolic_zeros(_dynamic_out)
                _out = combine(_dynamic_out, _static_out)
                return _out

            # Not `jax.jvp`, so as not to error if `body_fun` has any `custom_vjp`s.
            jax.linearize(_to_linearize, dynamic)
            if new_perturb_val is sentinel:
                # `_dynamic_out` in `_to_linearize` had no JVP tracers at all, despite
                # `_dynamic` having them. Presumably the user's `_body_fun` has no
                # differentiable dependency whatsoever.
                # This can happen if all the autograd is happening through
                # `perturb_body_fun`.
                return Static(perturb_val)
            assert jtu.tree_structure(
                perturb_val, is_leaf=_is_none
            ) == jtu.tree_structure(new_perturb_val, is_leaf=_is_none)
            new_perturb_val = jtu.tree_map(
                lambda x, y: False if (x is False and y is None) else y,
                perturb_val,
                new_perturb_val,
            )
            assert jtu.tree_structure(perturb_val) == jtu.tree_structure(
                new_perturb_val
            )
            if tree_equal(perturb_val, new_perturb_val):
                jtu.tree_map(_assert_bool, perturb_val)
                if getattr(typing, "TESTING", False):
                    print("stop_gradient_perturb_val", perturb_val)
                return Static(perturb_val)
            else:
                perturb_val = jtu.tree_map(operator.or_, perturb_val, new_perturb_val)

    perturb_val = jax.eval_shape(_resolve_perturb_val_impl).value
    return perturb_val


@filter_custom_jvp
def _stop_gradient_on_unperturbed(init_val, final_val, body_fun):
    del body_fun
    return final_val


def _perturb_to_tang(t, p):
    if t is None:
        return None
    elif p is None:
        return None
    elif p is False:
        return None
    elif p is True:
        return t
    else:
        assert False


@_stop_gradient_on_unperturbed.def_jvp
def _stop_gradient_on_unperturbed_jvp(primals, tangents):
    init_val, final_val, body_fun = primals
    t_init_val, t_final_val, t_body_fun = tangents
    del primals, tangents
    perturb_val, perturb_body_fun = jtu.tree_map(
        lambda _, t: t is not None, (init_val, body_fun), (t_init_val, t_body_fun)
    )
    perturb_val = _resolve_perturb_val(
        init_val, body_fun, perturb_val, perturb_body_fun
    )
    t_final_val = jtu.tree_map(
        _perturb_to_tang, t_final_val, perturb_val, is_leaf=_is_none
    )
    return final_val, t_final_val
