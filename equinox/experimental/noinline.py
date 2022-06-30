import functools as ft
from typing import Callable, Union
from typing_extensions import Literal

import jax
import jax.experimental.host_callback as hcb
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.numpy as jnp

from ..compile_utils import hashable_combine, hashable_partition
from ..custom_types import PyTree, sentinel, TreeDef
from ..filters import is_array
from ..module import Module, static_field


class _NoInlineArg(Module):
    dynamic: PyTree
    args: PyTree
    kwargs: PyTree


# Monkey-patch the batching rule for host_callback.call to work with noinline
_have_monkey_patched = False


# To explain how this works.
#
# `noinline` is built on top of `host_callback.call`; we use it as a convenient
# way to produce a separate XLA computation graph.
#
# However `host_callback` works by rewriting computation graphs; in particular
# this means that it only works with those primitives which already exist
# within JAX, and adding a new primitive isn't really supported. (It would be
# very difficult to add support for a new primitive that copy-pastes
# `host_callback` and makes the appropriate modifications.)
#
# So we're stuck with `host_callback.call` as our only mechanism. However, this
# doesn't support JVPs, batching etc, which we need here.
#
# This means we need to monkey-patch in support for JVPs etc. into
# `host_callback.call`, and we use `_NoInlineArg` to detect when we're calling
# `noinline` rather than any other `host_callback.call`; we should fall back to
# the default behaviour for generic `host_callback.call`s. (In particular if
# anyone else is doing similar monkey-patching; e.g.
# `eqx.experimental.stateful`.
#
# The implementation of each rule is essentially straightforward: we just use
# the public JAX API applied to the function we're wrapping, and then wrap that
# in a new `noinline` so that e.g. `jvp(noinline(fn)` becomes
# `noinline(jvp(fn))`. (It might actually be a bit nicer to use the internal
# JAX APIs for this purpose, especially if/when this ever gets upstreamed into
# core JAX.)
#
# The only thing to be careful of here are all these `Module`s being used to
# wrap every transformation.
#
# Suppose we have a `noinline(fn)` that we use multiple times in a jaxpr, and that we
# subsequently e.g. apply `jax.linear_transpose` to this jaxpr.
# Done naively we might convert `jax.linear_transpose(noinline(fn), x)` into
# `noinline(jax.linear_transpose(fn), x)` each time.
# However, each time `jax.linear_transpose(fn)` is called it would produce a new
# Python function object, with a different hash/equality to every other call to
# `jax.linear_transpose(fn)`. Meanwhile, `noinline` will JIT its argument (as
# part of the external secondary computation graph that is the whole point of
# using `noinline`). As each time it sees a result of `jax.linear_transpose(fn)`
# it would believe it has a different function, it would have to re-JIT each
# case separately.
#
# This would be unacceptable: the whole point of `noinline` is to avoid
# re-JIT'ing each time we see the same function!
#
# Callable `Module`s are instead used to provide equality and hashing, so that
# transposing `fn` produces an identical (wrt hash/equality) PyTree each time.
def _monkey_patch():
    global _have_monkey_patched
    if not _have_monkey_patched:
        _have_monkey_patched = True

        _old_outside_call_jvp_rule = ad.primitive_jvps[hcb.outside_call_p]
        _old_outside_call_transpose_rule = ad.primitive_transposes[hcb.outside_call_p]
        _old_outside_call_batching_rule = batching.primitive_batchers[
            hcb.outside_call_p
        ]

        class _Jvp(Module):
            fn: Callable

            def __call__(self, primal, tangent):
                _, out = jax.jvp(self.fn, (primal,), (tangent,))
                return out

        def _outside_call_jvp_rule(
            primal_arg_flat,
            tangent_arg_flat,
            *,
            arg_treedef,
            result_treedef,
            callback,
            **params
        ):
            primal_arg = jax.tree_unflatten(arg_treedef, primal_arg_flat)
            if type(primal_arg) is _NoInlineArg:
                # TODO: avoid instantiating symbolic zeros
                tangent_arg_flat = [ad.instantiate_zeros(t) for t in tangent_arg_flat]
                tangent_arg = jax.tree_unflatten(arg_treedef, tangent_arg_flat)
                invoke = callback.callback_func
                assert isinstance(invoke, _Invoke)
                fn = hashable_combine(
                    primal_arg.dynamic, invoke.static_leaves, invoke.static_treedef
                )
                eval_shape = lambda *_, **__: _get_shape(invoke, primal_arg)

                # Can't get primal_out during the jvp because the noinline
                # smushes the primal and tangent together, so the unknown (i.e.
                # PartialVal) tangents produce unknown (PartialVal) primals.
                # This is verboten - known input primals must produce known
                #
                # TODO: this is inefficient as it means we compute the primal
                # twice
                primal_out = noinline(fn, eval_shape)(
                    *primal_arg.args, **primal_arg.kwargs
                )
                tangent_out = noinline(_Jvp(invoke), eval_shape)(
                    primal_arg, tangent_arg
                )
                # Also note that we wrote `primal_out = ...` rather than
                # `primal_out = noinline(invoke, out_shape)(primal_arg)`. This latter
                # option is semantically correct but triggers extra compilation
                # as `invoke` and `fn` are different things (despite computing
                # the same output).

                primal_out_flat, primal_out_treedef = jax.tree_flatten(primal_out)
                tangent_out_flat, tangent_out_treedef = jax.tree_flatten(tangent_out)
                assert primal_out_treedef == result_treedef
                assert tangent_out_treedef == result_treedef
                return tuple(primal_out_flat), tuple(tangent_out_flat)
            else:
                return _old_outside_call_jvp_rule(
                    primal_arg_flat,
                    tangent_arg_flat,
                    arg_treedef=arg_treedef,
                    result_treedef=result_treedef,
                    callback=callback,
                    **params
                )

        class _LinearTranspose(Module):
            fn: Callable
            undefined_arg_flat: list
            defined_arg_flat: list
            arg_treedef: TreeDef

            def __call__(self, cts):
                def _fn(*_undefined_arg_flat):
                    _arg_flat = [
                        u if d is None else d
                        for u, d in zip(_undefined_arg_flat, self.defined_arg_flat)
                    ]
                    _arg = jax.tree_unflatten(self.arg_treedef, _arg_flat)
                    return self.fn(_arg)

                return jax.linear_transpose(_fn, *self.undefined_arg_flat)(cts)

        def _outside_call_transpose_rule(
            cts_flat, *args_flat, arg_treedef, result_treedef, callback, **params
        ):
            arg = jax.tree_unflatten(arg_treedef, args_flat)
            if type(arg) is _NoInlineArg:
                # TODO: avoid instantiating symbolic zeros
                cts_flat = [ad.instantiate_zeros(ct) for ct in cts_flat]
                cts = jax.tree_unflatten(result_treedef, cts_flat)
                invoke = callback.callback_func
                assert isinstance(invoke, _Invoke)
                arg_flat, arg_treedef = jax.tree_flatten(
                    arg, is_leaf=ad.is_undefined_primal
                )
                undefined_arg_flat = [
                    x.aval if ad.is_undefined_primal(x) else None for x in arg_flat
                ]
                defined_arg_flat = [
                    None if ad.is_undefined_primal(x) else x for x in arg_flat
                ]
                eval_shape = lambda *_: tuple(undefined_arg_flat)
                return noinline(
                    _LinearTranspose(
                        invoke, undefined_arg_flat, defined_arg_flat, arg_treedef
                    ),
                    eval_shape,
                )(cts)
            else:
                return _old_outside_call_transpose_rule(
                    cts,
                    *args_flat,
                    arg_treedef=arg_treedef,
                    result_treedef=result_treedef,
                    callback=callback**params
                )

        class _Vmap(Module):
            fn: Callable
            arg_treedef: TreeDef = static_field()
            in_axes_leaves: tuple = static_field()
            in_axes_treedef: PyTree = static_field()

            def __call__(self, arg_flat):
                def fn_flat(*_arg_flat):
                    _arg = jax.tree_unflatten(self.arg_treedef, _arg_flat)
                    return self.fn(_arg)

                # TODO: support not batching all outputs (would need to dig into
                # batching.batch_jaxpr etc., and adjust our output shape
                # calculations)
                in_axes = jax.tree_unflatten(self.in_axes_treedef, self.in_axes_leaves)
                return jax.vmap(fn_flat, in_axes=in_axes)(*arg_flat)

        def _index(a, b):
            if b is None:
                return a
            else:
                # like doing `a[b]`, except that `a` might be an aval so we have
                # to do this instead.
                shape = a.shape[:b] + a.shape[b + 1 :]
                return jax.ShapeDtypeStruct(shape=shape, dtype=a.dtype)

        def _prepend(axis_size):
            def _prepend_impl(struct):
                return jax.ShapeDtypeStruct(
                    shape=(axis_size,) + struct.shape, dtype=struct.dtype
                )

            return _prepend_impl

        def _outside_call_batching_rule(
            arg_flat,
            batch_axes_flat,
            *,
            arg_treedef,
            result_treedef,
            callback,
            **params
        ):
            arg = jax.tree_unflatten(arg_treedef, arg_flat)
            if type(arg) is _NoInlineArg:
                invoke = callback.callback_func
                assert isinstance(invoke, _Invoke)

                def eval_shape(_):
                    axis_size = None
                    for a, b in zip(arg_flat, batch_axes_flat):
                        if b is not None:
                            assert isinstance(b, int)
                            axis_size = a.shape[b]
                            break
                    if axis_size is None:
                        # Very rare that we end up on this branch: the user is
                        # calling `jax.vmap` without arguments, and specifying
                        # `jax.vmap(..., axis_size=...)` instead. In this case we
                        # can't identify the axis size based on the arguments, so we
                        # bite the bullet and go via `jax.eval_shape` instead.
                        return sentinel
                    else:
                        # More commmon case: we can figure out the output shape
                        # based on the output shape of the unbatched version, and
                        # knowledge of the axis size. We can skip the cost
                        # of evaluating `jax.eval_shape`.
                        single_arg_flat = jax.tree_map(
                            _index, arg_flat, batch_axes_flat
                        )
                        single_arg = jax.tree_unflatten(arg_treedef, single_arg_flat)
                        single_out_shape = _get_shape(invoke, single_arg)
                        return jax.tree_map(_prepend(axis_size), single_out_shape)

                batch_axes_flat_leaves, batch_axes_flat_treedef = jax.tree_flatten(
                    batch_axes_flat
                )
                batch_axes_flat_leaves = tuple(batch_axes_flat_leaves)
                out = noinline(
                    _Vmap(
                        invoke,
                        arg_treedef,
                        batch_axes_flat_leaves,
                        batch_axes_flat_treedef,
                    ),
                    eval_shape,
                )(arg_flat)
                out_flat, out_treedef = jax.tree_flatten(out)
                assert out_treedef == result_treedef
                return out_flat, [0] * len(out_flat)
            else:
                return _old_outside_call_batching_rule(
                    arg_flat,
                    batch_axes_flat,
                    arg_treedef=arg_treedef,
                    result_treedef=result_treedef,
                    callback=callback**params,
                )

        ad.primitive_jvps[hcb.outside_call_p] = _outside_call_jvp_rule
        ad.primitive_transposes[hcb.outside_call_p] = _outside_call_transpose_rule
        batching.primitive_batchers[hcb.outside_call_p] = _outside_call_batching_rule


class _Invoke(Module):
    static_leaves: PyTree = static_field()
    static_treedef: PyTree = static_field()

    # Not using eqx.filter_jit, as that has a higher overhead due to the
    # filtering. It's important that we keep overheads as low as possible as we
    # call _Invoke many times at runtime.
    @ft.partial(jax.jit, static_argnums=0)
    def __call__(self, arg):
        fn = hashable_combine(arg.dynamic, self.static_leaves, self.static_treedef)
        return fn(*arg.args, **arg.kwargs)


_out_shape_cache = {}


def _make_lookup(invoke: _Invoke, arg: _NoInlineArg):
    in_shape = jax.tree_map(lambda x: (x.shape, x.dtype), arg)
    lookup_leaves, lookup_treedef = jax.tree_flatten((in_shape, invoke))
    return tuple(lookup_leaves), lookup_treedef


def _get_shape(invoke: _Invoke, arg: _NoInlineArg):
    lookup = _make_lookup(invoke, arg)
    return _out_shape_cache[lookup]


class _NoInlineFn(Module):
    invoke: _Invoke
    dynamic: PyTree
    eval_shape: Union[Literal[sentinel], Callable]

    def __call__(__self, *args, **kwargs):
        args, kwargs = jax.tree_map(jnp.asarray, (args, kwargs))
        arg = _NoInlineArg(__self.dynamic, args, kwargs)
        lookup = _make_lookup(__self.invoke, arg)
        try:
            out_shape = _out_shape_cache[lookup]
        except KeyError:
            if __self.eval_shape is sentinel:
                out_shape = jax.eval_shape(__self.invoke, arg)
            else:
                out_shape = __self.eval_shape(*args, **kwargs)
                if out_shape is sentinel:
                    out_shape = jax.eval_shape(__self.invoke, arg)
            _out_shape_cache[lookup] = out_shape
        return hcb.call(__self.invoke, arg, result_shape=out_shape)


def noinline(fn, eval_shape=sentinel):
    """Compiles `fn` into a separate XLA computation graph, without inlining.

    By default, JAX will trace your entire program into a single computation graph, and
    then compile the entire graph in one go. This is usually great for efficiency:
    every operation you perform is "inlined" into a single computation graph, which
    allows for many compiler optimisations to occur.

    However, it can have some drawbacks: if you call the same operation multiple
    times then each copy is treated independently by the compiler. This can sometimes
    produce very long compile times. (Calling something multiple times because it's in
    a `lax.scan` etc. is fine -- it's if the function has to be traced multiple times
    that compile times slow down.)

    This decorator marks its function as instead having a separate computation graph,
    which can be optimised just once and reused each time. This produces faster compile
    times, at the expense of some runtime performance. (At runtime this introduces two
    sources of overhead: (1) the no-inline'd function cannot have compiler
    optimisations applied jointly with its surrounding computation graph. (2) there is
    the overhead of calling the sub-computation-graph from the main one.)

    **Arguments:**

    - `fn`: The callable to not inline. It need not be a pure Python function,
      e.g. it can be an instance of `eqx.Module` etc. as well. It must satisfy:

      1. It must be a function from (a PyTree of) JAX arrays to (a PyTree of) JAX arrays.
         That is to say, arbitrary Python inputs and outputs are not supported. Python
         int/float/bool/complex will be cast to JAX arrays.

      2. The result of `eqx.filter(fn, is_array, inverse=True)` must be hashable.
         This is what is looked up to get the same sub-computation-graph each time `fn`
         is called.

    - `eval_shape`: Optional. If passed, this should satisfy
      `eval_shape(*args, **kwargs) == jax.eval_shape(fn, *args, **kwargs)`, where
      `*args` and `**kwargs` denote whatever args and kwargs are passed at runtime.
      This argument can be used if the output shape is known to the end user, so that
      the call to `jax.eval_shape` can be skipped.

    **Returns**

    A version of `fn` that will be compiled only once when used inside a `jax.jit`
    region. It will not be inlined into the surrounding computation graph.

    !!! warning

        A `noinline`'d function is excuted on the CPU. If your main computation is on
        the GPU then this may be slow, and will necessitate GPU->CPU and CPU->GPU
        copies. As such `noinline` is only recommended for use in CPU-based computations.

    !!! example

        Differential equation solvers (here we use
        [Diffrax](https://github.com/patrick-kidger/diffrax)) often evaluate the vector
        field multiple times in different places. (Precisely, in the code below: twice
        to determine the initial step size; once to determine the initial state of the
        solver. Then another 19 times for making a step! (6 non-FSAL stages * 3 traces
        in the Newton optimiser + 1 for a Jacobian evaluation.) This makes them a great
        application for `noinline`. See
        [the benchmark](https://github.com/patrick-kidger/equinox/blob/main/benchmarks/noinline.py)
        for a code example.
    """

    _monkey_patch()
    dynamic, static_leaves, static_treedef = hashable_partition(fn, is_array)
    invoke = _Invoke(static_leaves, static_treedef)
    return _NoInlineFn(invoke, dynamic, eval_shape)
