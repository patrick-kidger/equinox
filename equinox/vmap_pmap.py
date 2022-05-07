import functools as ft
import inspect
from typing import Any, Callable, Dict, Union

import jax
import jax.interpreters.batching as batching
import jax.numpy as jnp

from .compile_utils import (
    compile_cache,
    hashable_combine,
    hashable_partition,
    Static,
    strip_wrapped_partial,
)
from .custom_types import BoolAxisSpec, PyTree, ResolvedBoolAxisSpec, sentinel
from .doc_utils import doc_strip_annotations
from .filters import combine, filter, is_array, partition
from .module import Module, module_update_wrapper


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


class _VmapWrapper(Module):
    _signature: inspect.Signature
    _fun: Callable
    _default: AxisSpec
    _fn: PyTree[AxisSpec]
    _args: PyTree[AxisSpec]
    _kwargs: PyTree[AxisSpec]
    _out: PyTree[AxisSpec]
    _callable_out_axes: bool
    _vmapkwargs: Dict[str, Any]

    def __call__(__self, *args, **kwargs):
        def _fun_wrapper(_fun, _args, _kwargs):
            result = _fun(*_args, **_kwargs)
            out_axes = _resolve_axes(result, __self._out)
            out_axes = _map_axes(out_axes)
            out_axes_is_none = jax.tree_map(_is_none, out_axes, is_leaf=_is_none)
            nonvmapd, vmapd = partition(result, out_axes_is_none)
            return vmapd, Static(nonvmapd)

        bound = __self._signature.bind(*args, **kwargs)
        del args, kwargs
        bound.apply_defaults()
        _args = __self._args + (__self._default,) * (
            len(bound.args) - len(__self._args)
        )
        _kwargs = {
            key: __self._kwargs.get(key, __self._default) for key in bound.kwargs
        }
        in_axes = _resolve_axes(
            (__self._fun, bound.args, bound.kwargs), (__self._fn, _args, _kwargs)
        )
        in_axes = _map_axes(in_axes)
        if __self._callable_out_axes:  # `out` of type AxisSpec
            out_axes = (jax.tree_map(_VmapFilter, __self._out), None)
        else:  # `out` of type ResolvedAxisSpec
            out_axes = _map_axes(__self._out)
        vmapd, nonvmapd = jax.vmap(
            _fun_wrapper, in_axes=in_axes, out_axes=out_axes, **__self._vmapkwargs
        )(__self._fun, bound.args, bound.kwargs)
        return combine(vmapd, nonvmapd.value)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return jax.tree_util.Partial(self, instance)


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

    !!! info

        By default, all JAX arrays are vectorised down their leading axis (i.e. axis
        index 0), and all other types are not vectorised.

    **Arguments:**

    In each of the following cases, then `int` indicates an array axis to vectorise
    over, `None` indicates that an argument should be broadcast (not vectorised
    over), and functions `Leaf -> Union[None, int]` are mapped and evaluated on every
    leaf of their subtree. `None` should be used for non-JAX-array arguments.

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
        are PyTrees with leaves that are either `Union[None, int]`s or functions
        `Leaf -> Union[None, int]`. The PyTree structures should be prefixes of the
        corresponding input to `fun`.
    - `out` should be a PyTree with leaves that are either `Union[None, int]`s or
        functions `Leaf -> Union[None, int]`. The PyTree structure should be a prefix
        of the output of `fun`.
    - `fn` should be a PyTree with leaves that are either `Union[None, int]`s or
        functions `Leaf -> Union[None, int]`. The PyTree structure should be a prefix of
        `fun` itself. (Note that `fun` may be any callable, e.g. a bound method, or a
        class implementing `__call__`, and doesn't have to be a normal Python function.)
    - `**vmapkwargs` are any other keyword arguments to `jax.vmap`.

    When `args`, `kwargs`, `out`, `fn` are prefixes of the corresponding input, their
    value will be mapped over the input PyTree.

    **Returns:**

    The vectorised version of `fun`.

    !!! info

        In fact, besides `None`, `int` and `Leaf -> Union[None, int]`: boolean
        types are also supported and treated identically to `None`. This is to support
        seamlessly switching between [`equinox.filter_pmap`][] and
        [`equinox.filter_vmap`][] if desired.

    !!! warning

        Using functions `Leaf -> Union[None, int]` in `out` is considered experimental,
        and may change.

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
        parameters=[
            p
            if p.kind
            in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            else p.replace(default=default)
            for p in signature.parameters.values()
        ]
    )
    bound = signature_default.bind_partial(*args, **kwargs)
    del args, kwargs
    bound.apply_defaults()

    if any(callable(o) for o in jax.tree_leaves(out)):
        # Experimental behaviour
        _monkey_patch()
        callable_out_axes = True
    else:
        callable_out_axes = False

    vmap_wrapper = _VmapWrapper(
        _signature=signature,
        _fun=fun,
        _default=default,
        _fn=fn,
        _args=bound.args,
        _kwargs=bound.kwargs,
        _out=out,
        _callable_out_axes=callable_out_axes,
        _vmapkwargs=vmapkwargs,
    )
    return module_update_wrapper(vmap_wrapper, fun)


@compile_cache
def _filter_pmap_cache(unwrapped_fun, **pmapkwargs):
    @ft.partial(jax.pmap, **pmapkwargs)
    @ft.wraps(unwrapped_fun)
    def fun_wrapped(dynamic, static_leaves, static_treedef, jit_out_axes):
        _fun, _args, _kwargs, _maybe_dummy = hashable_combine(
            dynamic, static_leaves, static_treedef
        )
        del _maybe_dummy
        _out = _fun(*_args, **_kwargs)
        _dynamic, _static = partition(_out, jit_out_axes)
        return _dynamic, Static(_static)

    return fun_wrapped


class _PmapWrapper(Module):
    _signature: inspect.Signature
    _fun: Callable
    _default: AxisSpec
    _fn: PyTree[AxisSpec]
    _args: PyTree[AxisSpec]
    _kwargs: PyTree[AxisSpec]
    _out: PyTree[AxisSpec]
    _unwrapped_fun: Any
    _pmapkwargs: Dict[str, Any]

    def _fun_wrapper(self, is_lower, args, kwargs):
        bound = self._signature.bind(*args, **kwargs)
        del args, kwargs
        bound.apply_defaults()
        _args = self._args + (self._default,) * (len(bound.args) - len(self._args))
        _kwargs = {key: self._kwargs.get(key, self._default) for key in bound.kwargs}
        try:
            axis_size = self._pmapkwargs["axis_size"]
        except KeyError:
            maybe_dummy = 0  # hashable non-array object
        else:
            # Work around JAX bug #9252
            maybe_dummy = jnp.empty(axis_size)
        in_axes = _resolve_axes(
            (self._fun, bound.args, bound.kwargs, maybe_dummy),
            (self._fn, _args, _kwargs, _zero_if_array_else_none),
        )
        jit_in_axes = _jit_axes(in_axes)
        map_in_axes = _map_axes(in_axes)
        jit_out_axes = _jit_axes(self._out)
        map_out_axes = _map_axes(self._out)

        cached = _filter_pmap_cache(
            self._unwrapped_fun,
            in_axes=(map_in_axes, None, None),
            out_axes=(map_out_axes, None),
            static_broadcasted_argnums=(1, 2, 3),
            **self._pmapkwargs
        )

        dynamic, static_leaves, static_treedef = hashable_partition(
            (self._fun, bound.args, bound.kwargs, maybe_dummy), jit_in_axes
        )
        if is_lower:
            return cached.lower(dynamic, static_leaves, static_treedef, jit_out_axes)
        else:
            dynamic_out, static_out = cached(
                dynamic, static_leaves, static_treedef, jit_out_axes
            )
            return combine(dynamic_out, static_out.value)

    def __call__(__self, *args, **kwargs):
        return __self._fun_wrapper(False, args, kwargs)

    def lower(__self, *args, **kwargs):
        return __self._fun_wrapper(True, args, kwargs)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return jax.tree_util.Partial(self, instance)


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

    !!! info

        By default, the computation is parallelised by splitting all JAX arrays down
        their leading axis (i.e. axis index 0), and broadcasting all other types to
        each replica.

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

    `None`, `False` and `True` should be used for non-JAX-array arguments.

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
        are PyTrees with leaves that are either `Union[None, bool, int]`s or functions
        `Leaf -> Union[None, bool, int]`. The PyTree structures should be prefixes of
        the corresponding input to `fun`.
    - `out` should be a PyTree with leaves that are either `Union[None, bool, int]`s or
        functions `Leaf -> Union[None, bool, int]`. The PyTree structure should be a
        prefix of the output of `fun`. `True` indicates a tracer, `False` indicates any
        auxiliary information to return.
    - `fn` should be a PyTree with leaves that are either `Union[None, bool, int]`s or
        functions `Leaf -> Union[None, bool, int]`. The PyTree structure should be a
        prefix of `fun` itself. (Note that `fun` may be any callable, e.g. a bound
        method, or a class implementing `__call__`, and doesn't have to be a normal
        Python function.)
    - `**pmapkwargs` are any other keyword arguments to `jax.pmap`.

    When `args`, `kwargs`, `out`, `fn` are prefixes of the corresponding input, their
    value will be mapped over the input PyTree.

    **Returns:**

    The parallelised version of `fun`.

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
            filter_pmap,
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
        parameters=[
            p
            if p.kind
            in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            else p.replace(default=default)
            for p in signature.parameters.values()
        ]
    )
    bound = signature_default.bind_partial(*args, **kwargs)
    del args, kwargs
    bound.apply_defaults()

    unwrapped_fun = filter(strip_wrapped_partial(fun), _jit_axes(fn), inverse=True)

    if any(callable(o) for o in jax.tree_leaves(out)):
        # In practice we demand `out` be of type `PyTree[ResolvedAxisSpec]`.
        raise NotImplementedError(
            "`filter_pmap(out_axes=...)` does not support filter functions (only None, bool, int)"
        )

    pmap_wrapper = _PmapWrapper(
        _signature=signature,
        _fun=fun,
        _default=default,
        _fn=fn,
        _args=bound.args,
        _kwargs=bound.kwargs,
        _out=out,
        _unwrapped_fun=unwrapped_fun,
        _pmapkwargs=pmapkwargs,
    )

    return module_update_wrapper(pmap_wrapper, fun)
