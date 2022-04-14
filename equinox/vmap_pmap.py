import functools as ft
import inspect
from typing import Any, Callable, Union

import jax
import jax.interpreters.batching as batching

from .compile_utils import (
    hashable_combine,
    hashable_partition,
    Static,
    strip_wrapped_partial,
)
from .custom_types import BoolAxisSpec, PyTree, ResolvedBoolAxisSpec, sentinel
from .doc_utils import doc_strip_annotations
from .filters import combine, is_array, partition


ResolvedMapAxisSpec = Union[None, int]
MapAxisSpec = Union[ResolvedMapAxisSpec, Callable[[Any], ResolvedMapAxisSpec]]
#
ResolvedAxisSpec = Union[ResolvedBoolAxisSpec, ResolvedMapAxisSpec]
AxisSpec = Union[ResolvedAxisSpec, Callable[[Any], ResolvedAxisSpec]]


def _is_none(x: Any) -> bool:
    return x is None


def _resolve_axis(axis_spec: AxisSpec, elem: Any) -> PyTree[ResolvedAxisSpec]:
    if axis_spec is None or isinstance(axis_spec, (bool, int)):
        return axis_spec
    if callable(axis_spec):
        return jax.tree_map(axis_spec, elem)
    else:
        raise ValueError(
            "`in_axes` and `out_axes` must consist of None, bools, ints, and callables only."
        )


def _resolve_axes(
    pytree: PyTree[Any], axes_spec: PyTree[AxisSpec]
) -> PyTree[ResolvedAxisSpec]:
    return jax.tree_map(_resolve_axis, axes_spec, pytree, is_leaf=_is_none)


def _jit_axis(axis: ResolvedAxisSpec) -> BoolAxisSpec:  # not necessarily resolved
    if isinstance(axis, bool):
        return axis
    elif isinstance(axis, int):
        return True
    elif axis is None:
        return is_array
    else:
        assert False


def _map_axis(axis: ResolvedAxisSpec) -> ResolvedMapAxisSpec:
    if isinstance(axis, bool):
        return None
    elif isinstance(axis, int):
        return axis
    elif axis is None:
        return None
    else:
        assert False


def _jit_axes(axes: PyTree[ResolvedAxisSpec]) -> PyTree[BoolAxisSpec]:
    return jax.tree_map(_jit_axis, axes, is_leaf=_is_none)


def _map_axes(axes: PyTree[ResolvedAxisSpec]) -> PyTree[ResolvedMapAxisSpec]:
    return jax.tree_map(_map_axis, axes, is_leaf=_is_none)


class _VmapFilter:
    def __init__(self, axis: AxisSpec):
        self.axis = axis


_have_monkey_patched = False


def _monkey_patch():
    global _have_monkey_patched
    if not _have_monkey_patched:
        _have_monkey_patched = True

        _old_from_elt = batching.from_elt

        def from_elt(trace, axis_size, x, spec):
            if isinstance(spec, _VmapFilter):
                spec = _resolve_axis(spec.axis, x)
                spec = _map_axis(spec)
            return _old_from_elt(trace, axis_size, x, spec)

        batching.from_elt = from_elt
        batching.spec_types.add(_VmapFilter)


def _zero_if_array_else_none(x: Any) -> ResolvedMapAxisSpec:
    return 0 if is_array(x) else None


# Note the use of AxisSpec rather than MapAxisSpec.
# This is to support seamlessly switching out filter_pmap for filter_vmap.
@doc_strip_annotations
def filter_vmap(
    fun: Callable = sentinel,
    *,
    default: AxisSpec = _zero_if_array_else_none,
    fn: PyTree[AxisSpec] = None,
    args: PyTree[AxisSpec] = (),
    kwargs: PyTree[AxisSpec] = None,
    # `out` default would ideally be _zero_if_array_else_none but that hits
    # experimental behaviour, so it's not a good default.
    # As a bonus, this also keeps the default the same as filter_pmap.
    out: PyTree[AxisSpec] = 0,
    **vmapkwargs
) -> Callable:
    """Wraps together [`equinox.partition`][] and `jax.vmap`.

    **Arguments:**

    In each of the following cases, then `int` indicates an array axis to vectorise
    over, `None` indicates that an argument should be broadcast (not vectorised
    over), and functions `Leaf -> Union[None, int]` are mapped and evaluated on every
    leaf of their subtree.

    (This is the same semantics as `jax.vmap(in_axes=..., out_axes=...)`.)

    `None` may be used for non-JAX-array arguments. It is an error to try and specify
    an integer axis for a non-JAX-array.

    - `fun` is a pure function to vectorise.
    - `default` should be a `Union[None, int]` or a function
        `Leaf -> Union[None, int]`, and is applied by default to every argument and
        keyword argument to `fun`.
    - `args` is an optional per-argument override for `default`, and should be a tuple
        of PyTrees with leaves that are either `Union[None, int]`s or functions
        `Leaf -> Union[None, int]`. The PyTree structures should be prefixes of the
        corresponding input to `fun`.
    - `kwargs` is an optional per-keyword-argument override for `default` and should be
        a dictionary, whose keys are the names of arguments to `fun`, and whose values
        are PyTrees with leaves that either `Union[None, int]`s or functions
        `Leaf -> Union[None, int]`. The PyTree structures should be prefixes of the
        corresponding input to `fun`.
    - `out` should be a PyTree with leaves that either `Union[None, int]`s or functions
        `Leaf -> Union[None, int]`. The PyTree structure should be a prefix of the
        output of `fun`.
    - `fn` should be a PyTree with leaves that either `Union[None, int]`s or functions
        `Leaf -> Union[None, int]`. The PyTree structure should be a prefix of `fun`
        itself. (Note that `fun` may be any callable, e.g. a bound method, or a class
        implementing `__call__`, and doesn't have to be a normal Python function.)
    - `**vmapkwargs` are any other keyword arguments to `jax.vmap`.

    When `args`, `kwargs`, `out`, `fn` are prefixes of the corresponding input, their
    value will be mapped over the input PyTree.

    **Returns:**

    The vectorised version of `fun`.

    !!! info

        By default, all JAX arrays are vectorised down their leading axis (i.e. axis
        index 0), and all other types are not vectorised.

    !!! info

        In fact, besides `None`, `int` and `Leaf -> Union[None, int]`, then boolean
        types are also supported, and treated identically to `None`. This is to support
        seamlessly switching between [`equinox.filter_pmap`][] and
        [`equinox.filter_vmap`][] if desired.

    !!! warning

        Using functions `Leaf -> Union[None, int]` in `out` (as opposed to just values
        of type `Union[None, int]`) is considered experimental, and may change.

    !!! example

        ```python
        import equinox as eqx
        import jax.numpy as jnp

        @eqx.filter_vmap
        def f(x, y):
            return x + y

        @eqx.filter_vmap(kwargs=dict(x=1))
        def g(x, y):
            return x + y

        @eqx.filter_vmap(args=(None,))
        def h(x, y):
            return x + y

        f(jnp.array([1, 2]), jnp.array([3, 4]))  # both args vectorised down axis 0
        f(jnp.array([1, 2]), 3)  # first arg vectorised down axis 0
                                 # second arg broadcasted

        g(jnp.array([[1, 2]]), jnp.array([3, 4]))  # first arg vectorised down axis 1
                                                   # second arg vectorised down axis 0

        h(jnp.array(1), jnp.array([2, 3]))  # first arg broadcasted
                                            # second arg vectorised down axis 0
        ```

    !!! example

        `filter_vmap` can be used to easily create ensembles of models. For example, here's an
        ensemble of eight MLPs:

        ```python
        import equinox as eqx
        import jax.random as jr

        key = jr.PRNGKey(0)
        keys = jr.split(key, 8)

        # Create an ensemble of models

        @eqx.filter_vmap(out=lambda x: 0 if eqx.is_array(x) else None)
        def make_ensemble(key):
            return eqx.nn.MLP(2, 2, 2, 2, key=key)

        mlp_ensemble = make_ensemble(keys)

        # Evaluate each member of the ensemble on the same data

        @eqx.filter_vmap(kwargs=dict(x=None))
        def evaluate_ensemble(model, x):
            return model(x)

        evaluate_ensemble(mlp_ensemble, jr.normal(key, (2,)))

        # Evaluate each member of the ensemble on different data

        @eqx.filter_vmap
        def evaluate_per_ensemble(model, x):
            return model(x)

        evaluate_per_ensemble(mlp_ensemble, jr.normal(key, (8, 2)))
        ```

        Here, `make_ensemble` works because [`equinox.nn.MLP`][] is a PyTree, and so it
        is a valid output from a `filter_vmap`. This PyTree includes some JAX arrays
        (the weights and biases) and some non-JAX-arrays (e.g. activation functions).
        `filter_vmap` will vectorise the JAX arrays (with separate weights for each
        member of the ensemble) whilst leaving the non-JAX-arrays alone.

        Note that as the weights in `mlp_ensemble` now have a leading batch dimension
        -- that the weights of `eqx.nn.MLP` instances do not typically have -- then it
        cannot be called directly. It must instead be passed back into a vectorised
        region to be called.
    """

    if fun is sentinel:
        return ft.partial(
            filter_vmap,
            default=default,
            fn=fn,
            args=args,
            kwargs=kwargs,
            out=out,
            **vmapkwargs
        )

    if kwargs is None:
        kwargs = {}

    signature = inspect.signature(fun)

    signature_default = signature.replace(
        parameters=[p.replace(default=default) for p in signature.parameters.values()]
    )
    bound = signature_default.bind_partial(*args, **kwargs)
    bound.apply_defaults()

    if any(callable(o) for o in jax.tree_leaves(out)):
        # Experimental behaviour
        _monkey_patch()
        callable_out_axes = True
    else:
        callable_out_axes = False

    def _fun_wrapper(_fun, _args, _kwargs):
        result = _fun(*_args, **_kwargs)
        out_axes = _resolve_axes(result, out)
        out_axes = _map_axes(out_axes)
        out_axes_is_none = jax.tree_map(_is_none, out_axes, is_leaf=_is_none)
        nonvmapd, vmapd = partition(result, out_axes_is_none)
        return vmapd, Static(nonvmapd)

    @ft.wraps(fun)
    def fun_wrapper(*_args, **_kwargs):
        _bound = signature.bind(*_args, **_kwargs)
        _bound.apply_defaults()
        in_axes = _resolve_axes(
            (fun, _bound.args, _bound.kwargs), (fn, bound.args, bound.kwargs)
        )
        in_axes = _map_axes(in_axes)
        if callable_out_axes:  # `out` of type AxisSpec
            out_axes = (jax.tree_map(_VmapFilter, out), None)
        else:  # `out` of type ResolvedAxisSpec
            out_axes = _map_axes(out)
        vmapd, nonvmapd = jax.vmap(
            _fun_wrapper, in_axes=in_axes, out_axes=out_axes, **vmapkwargs
        )(fun, _bound.args, _bound.kwargs)
        return combine(vmapd, nonvmapd.value)

    return fun_wrapper


@ft.lru_cache(maxsize=None)
def _filter_pmap_cache(unwrapped_fun_treedef, unwrapped_fun_leaves, **pmapkwargs):
    unwrapped_fun = jax.tree_unflatten(unwrapped_fun_treedef, unwrapped_fun_leaves)

    @ft.partial(jax.pmap, **pmapkwargs)
    @ft.wraps(unwrapped_fun)
    def fun_wrapped(dynamic, static_leaves, static_treedef, jit_out_axes):
        _fun, _args, _kwargs = hashable_combine(dynamic, static_leaves, static_treedef)
        _out = _fun(*_args, **_kwargs)
        _dynamic, _static = partition(_out, jit_out_axes)
        return _dynamic, Static(_static)

    return fun_wrapped


@doc_strip_annotations
def filter_pmap(
    fun: Callable = sentinel,
    axis_name=None,
    *,
    default: AxisSpec = _zero_if_array_else_none,
    fn: PyTree[AxisSpec] = None,
    args: PyTree[AxisSpec] = (),
    kwargs: PyTree[AxisSpec] = None,
    out: PyTree[AxisSpec] = 0,
    **pmapkwargs
) -> Callable:
    """Wraps together [`equinox.partition`][] and `jax.pmap`.

    **Arguments:**

    In each of the following cases, then `int` indicates an array axis to split down,
    `None` indicates that an argument should be broadcast to each device (not split
    across devices), and functions `Leaf -> Union[None, bool, int]` are mapped and
    evaluated on every leaf of their subtree.

    Note that `jax.pmap`, and thus `equinox.filter_pmap`, also JIT-compile their
    function in the same way as `jax.jit`. By default, all JAX arrays are traced and
    all other arrays are treated as static inputs. This may be controlled explicitly
    -- instead of just passing `None` -- by passing either `True` (traced) or
    `False` (static).

    (For `None` and `int`, this is the same semantics as
    `jax.vmap(in_axes=..., out_axes=...)`.)

    `None`, `False` and `True` may be used for non-JAX-array arguments. It is an error
    to try and specify an integer axis for a non-JAX-array.

    - `fun` is a pure function to parallelise.
    - `default` should be a `Union[None, bool, int]` or a function
        `Leaf -> Union[None, bool, int]`, and is applied by default to every argument
        and keyword argument to `fun`.
    - `args` is an optional per-argument override for `default`, and should be a tuple
        of PyTrees with leaves that are either `Union[None, bool, int]`s or functions
        `Leaf -> Union[None, bool, int]`. The PyTree structures should be prefixes of
        the corresponding input to `fun`.
    - `kwargs` is an optional per-keyword-argument override for `default` and should be
        a dictionary, whose keys are the names of arguments to `fun`, and whose values
        are PyTrees with leaves that either `Union[None, bool, int]`s or functions
        `Leaf -> Union[None, bool, int]`. The PyTree structures should be prefixes of
        the corresponding input to `fun`.
    - `out` should be a PyTree with leaves that either `Union[None, bool, int]`s or
        functions `Leaf -> Union[None, bool, int]`. The PyTree structure should be a
        prefix of the output of `fun`. `True` indicates a tracer, `False` indicates any
        auxiliary information to return.
    - `fn` should be a PyTree with leaves that either `Union[None, bool, int]`s or
        functions `Leaf -> Union[None, bool, int]`. The PyTree structure should be a
        prefix of `fun` itself. (Note that `fun` may be any callable, e.g. a bound
        method, or a class implementing `__call__`, and doesn't have to be a normal
        Python function.)
    - `**pmapkwargs` are any other keyword arguments to `jax.pmap`.

    When `args`, `kwargs`, `out`, `fn` are prefixes of the corresponding input, their
    value will be mapped over the input PyTree.

    **Returns:**

    The parallelised version of `fun`.

    !!! info

        By default, the computation is parallelised by splitting all JAX arrays down
        their leading axis (i.e. axis index 0), and broadcasting all other types to
        each replica.

    !!! example

        ```python
        import equinox as eqx
        import jax.numpy as jnp

        @eqx.filter_pmap
        def f(x, y):
            return x + y

        @eqx.filter_pmap(kwargs=dict(x=1))
        def g(x, y):
            return x + y

        @eqx.filter_pmap(args=(None,))
        def h(x, y):
            return x + y

        @eqx.filter_pmap
        def apply(fun, x):
            return fun(x)

        f(jnp.array([1, 2]), jnp.array([3, 4]))  # both args split down axis 0
        f(jnp.array([1, 2]), 3)  # first arg split down axis 0
                                 # second arg broadcasted

        g(jnp.array([[1, 2]]), jnp.array([3, 4]))  # first arg split down axis 1
                                                   # second arg split down axis 0

        h(jnp.array(1), jnp.array([2, 3]))  # first arg broadcasted
                                            # second arg split down axis 0

        apply(lambda x: x + 1, jnp.array([2, 3]))  # first arg broadcasted (as it's not
                                                   # a JAX array)
                                                   # second arg split down axis 0
        ```
    """

    if fun is sentinel:
        return ft.partial(
            filter_vmap,
            axis_name=axis_name,
            default=default,
            fn=fn,
            args=args,
            kwargs=kwargs,
            out=out,
            **pmapkwargs
        )

    if kwargs is None:
        kwargs = {}

    signature = inspect.signature(fun)

    signature_default = signature.replace(
        parameters=[p.replace(default=default) for p in signature.parameters.values()]
    )
    bound = signature_default.bind_partial(*args, **kwargs)
    bound.apply_defaults()

    _, unwrapped_fun_leaves, unwrapped_fun_treedef = hashable_partition(
        strip_wrapped_partial(fun), fn
    )

    def _fun_wrapper(is_lower, _args, _kwargs):
        _bound = signature.bind(*_args, **_kwargs)
        _bound.apply_defaults()
        in_axes = _resolve_axes(
            (fun, _bound.args, _bound.kwargs), (fn, bound.args, bound.kwargs)
        )
        jit_in_axes = _jit_axes(in_axes)
        map_in_axes = _map_axes(in_axes)
        if any(callable(o) for o in jax.tree_leaves(out)):
            # In practice we demand `out` be of type `PyTree[ResolvedAxisSpec]`.
            raise NotImplementedError(
                "`filter_pmap(out_axes=...)` does not support filter functions (only None, bool, int)"
            )
        jit_out_axes = _jit_axes(out)
        map_out_axes = _map_axes(out)

        cached = _filter_pmap_cache(
            unwrapped_fun_treedef,
            unwrapped_fun_leaves,
            in_axes=(map_in_axes, None, None),
            out_axes=(map_out_axes, None),
            static_broadcasted_argnums=(1, 2, 3),
            **pmapkwargs
        )

        dynamic, static_leaves, static_treedef = hashable_partition(
            (fun, _bound.args, _bound.kwargs), jit_in_axes
        )
        if is_lower:
            return cached.lower(dynamic, static_leaves, static_treedef, jit_out_axes)
        else:
            dynamic_out, static_out = cached(
                dynamic, static_leaves, static_treedef, jit_out_axes
            )
            return combine(dynamic_out, static_out.value)

    @ft.wraps(fun)
    def fun_wrapper(*_args, **_kwargs):
        return _fun_wrapper(False, _args, _kwargs)

    def lower(*_args, **_kwargs):
        return _fun_wrapper(True, _args, _kwargs)

    fun_wrapper.lower = lower

    return fun_wrapper
