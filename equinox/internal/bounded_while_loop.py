import math
from typing import Any

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Bool

from ..module import Module
from .unvmap import unvmap_any


def bounded_while_loop(cond_fun, body_fun, init_val, max_steps, base=16):
    """Reverse-mode autodifferentiable while loop.

    Mostly as `lax.while_loop`, with a few small changes.

    Arguments:
        cond_fun: function `a -> a`
        body_fun: function `a -> b -> a`, where `b` is a function that should be used
            instead of performing in-place updates with .at[].set() etc; see below.
        init_val: pytree with structure `a`.
        max_steps: integer or `None`.
        base: integer.

    Limitations with in-place updates.:
        The single big limitation is around making in-place updates. Done naively then
        the XLA compiler will fail to treat these as in-place and will make a copy
        every time. (See JAX issue #8192.)

        Working around this is a bit of a hassle -- as follows -- and it is for this
        reason that `body_fun` takes a second argument.

        If you ever have:
        - an inplace update...
        - ...made to the input to the body_fun...
        - ...whose result is returned from the body_fun...
        ...then you should use

        ```python
        x = inplace(x).at[i].set(u)
        x = HadInplaceUpdate(x)
        ```

        in place of

        ```python
        x = x.at[i].set(u)
        ```

        where `inplace` is the second argument to `body_fun`, and `HadInplaceUpdate` is
        available at `diffrax.misc.HadInplaceUpdate`.

        Internally, `bounded_while_loop` will treat things so as to work around this
        limitation of XLA.

        !!! faq

            `HadInplaceUpdate` is available separately (instead of being returned
            automatically from `inplace().at[].set()`) in case the in-place update
            takes place inside e.g. a `lax.scan` or similar, and you need to maintain
            PyTree structures. Just place the `HadInplaceUpdate` at the very end of
            `body_fun`. (And applied only to those array(s) that actually had in-place
            update(s), if the state is a PyTree.)

        !!! note

            If you need to nest `bounded_while_loop`s, then the two `inplace` functions
            can be merged:

            ```python
            def body_fun(val, inplace):
                ...  # stuff (use inplace)

                def inner_body_fun(_val, _inplace):
                    _inplace = _inplace.merge(inplace)
                    ...  # stuff (use _inplace)

                bounded_while_loop(body_fun=inner_body_fun, ...)

                ... # stuff (use inplace)

            bounded_while_loop(body_fun=body_fun, ...)
            ```

        !!! note

            In-place updates to arrays that are _created_ inside of `body_fun` can be
            made as normal. It's just those arrays that are part of the state (that is
            passed in and out) that need to be treated specially.

    Note the extra `max_steps` argument. If this is `None` then `bounded_while_loop`
    will fall back to `lax.while_loop` (which is not reverse-mode autodifferentiable).
    If it is a non-negative integer then this is the maximum number of steps which may
    be taken in the loop, after which the loop will exit unconditionally.

    Note the extra `base` argument.
    - Run time will increase slightly as `base` increases.
    - Compilation time will decrease substantially as
      `math.ceil(math.log(max_steps, base))` decreases. (Which happens as `base`
      increases.)
    """

    init_val = jtu.tree_map(jnp.asarray, init_val)

    if max_steps is None:

        def _make_update(_new_val):
            if isinstance(_new_val, HadInplaceUpdate):
                return _new_val.val
            else:
                return _new_val

        def _body_fun(_val):
            inplace = lambda x: x
            inplace.pred = True
            _new_val = body_fun(_val, inplace)
            return jtu.tree_map(
                _make_update,
                _new_val,
                is_leaf=lambda x: isinstance(x, HadInplaceUpdate),
            )

        return lax.while_loop(cond_fun, _body_fun, init_val)

    if not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a non-negative integer")
    if max_steps == 0:
        return init_val

    def _cond_fun(val, step):
        return cond_fun(val) & (step < max_steps)

    init_data = (cond_fun(init_val), init_val, 0)
    rounded_max_steps = base ** int(math.ceil(math.log(max_steps, base)))
    _, val, _ = _while_loop(_cond_fun, body_fun, init_data, rounded_max_steps, base)
    return val


class _InplaceUpdate(Module):
    pred: Bool[Array, "..."]

    def __call__(self, val: Array):
        return _InplaceUpdateInner(self.pred, val)

    def merge(self, other: "_InplaceUpdate") -> "_InplaceUpdate":
        return _InplaceUpdate(self.pred & other.pred)


class _InplaceUpdateInner(Module):
    pred: Bool[Array, "..."]
    val: Array

    @property
    def at(self):
        return _InplaceUpdateInnerInner(self.pred, self.val)


class _InplaceUpdateInnerInner(Module):
    pred: Bool[Array, "..."]
    val: Array

    def __getitem__(self, index: Any):
        return _InplaceUpdateInnerInnerInner(self.pred, self.val, index)


class _InplaceUpdateInnerInnerInner(Module):
    pred: Bool[Array, "..."]
    val: Array
    index: Any

    # TODO: implement other .add() etc. methods if required.

    def set(self, update: Array, **kwargs) -> Array:
        old = self.val[self.index]
        new = lax.select(self.pred, update, old)
        return self.val.at[self.index].set(new, **kwargs)


class HadInplaceUpdate(Module):
    val: Array


# There's several tricks happening here to work around various limitations of JAX.
# (Also see https://github.com/google/jax/issues/2139#issuecomment-1039293633)
# 1. `unvmap_any` prior to using `lax.cond`. JAX has a problem in that vmap-of-cond
#    is converted to a `lax.select`, which executes both branches unconditionally.
#    Thus writing this naively, using a plain `lax.cond`, will mean the loop always
#    runs to `max_steps` when executing under vmap. Instead we run (only) until every
#    batch element has finished.
# 2. Treating in-place updates specially in the body_fun. Specifically we need to
#    `lax.select` the update-to-make, not the updated buffer. This is because the
#    latter instead results in XLA:CPU failing to determine that the buffer can be
#    updated in-place, and instead it makes a copy. c.f. JAX issue #8192.
#    This is done through the extra `inplace` argument provided to `body_fun`.
# 3. The use of the `@jax.checkpoint` decorator. Backpropagating through a
#    `bounded_while_loop` will otherwise run in θ(max_steps) time, rather than
#    θ(number of steps actually taken). See
#    https://docs.kidger.site/diffrax/devdocs/bounded_while_loop/
# 4. The use of `base`. In theory `base=2` is optimal at run time, as it implies the
#    fewest superfluous operations. In practice this implies quite deep recursion in
#    the construction of the bounded while loop, and this slows down the jaxpr
#    creation and the XLA compilation. We choose `base=16` as a reasonable-looking
#    compromise between compilation time and run time.
def _while_loop(cond_fun, body_fun, data, max_steps, base):
    if max_steps == 1:
        pred, val, step = data

        inplace_update = _InplaceUpdate(pred)
        new_val = body_fun(val, inplace_update)

        def _make_update(_new_val, _val):
            if isinstance(_new_val, HadInplaceUpdate):
                return _new_val.val
            else:
                return lax.select(pred, _new_val, _val)

        new_val = jtu.tree_map(
            _make_update,
            new_val,
            val,
            is_leaf=lambda x: isinstance(x, HadInplaceUpdate),
        )
        new_step = step + 1
        return cond_fun(new_val, new_step), new_val, new_step
    else:

        def _call(_data):
            return _while_loop(cond_fun, body_fun, _data, max_steps // base, base)

        def _scan_fn(_data, _):
            _pred, _, _ = _data
            _unvmap_pred = unvmap_any(_pred)
            return lax.cond(_unvmap_pred, _call, lambda x: x, _data), None

        # Don't put checkpointing on the lowest level
        if max_steps != base:
            _scan_fn = jax.checkpoint(_scan_fn, prevent_cse=False)

        return lax.scan(_scan_fn, data, xs=None, length=base)[0]
