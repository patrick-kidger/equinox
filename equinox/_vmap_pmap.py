import dataclasses
import functools as ft
import inspect
import warnings
from collections.abc import Callable, Hashable
from typing import Any, Literal, Optional, overload, Union

import jax
import jax._src.traceback_util as traceback_util
import jax.core
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PyTree

from ._compile_utils import (
    compile_cache,
    hashable_combine,
    hashable_partition,
    Lowered,
)
from ._custom_types import sentinel
from ._deprecate import deprecated_0_10
from ._doc_utils import doc_remove_args
from ._filters import combine, filter, is_array, is_array_like, partition
from ._module import Module, module_update_wrapper, Partial, Static


traceback_util.register_exclusion(__file__)


ResolvedAxisSpec = Optional[int]
AxisSpec = Union[ResolvedAxisSpec, Callable[[Any], ResolvedAxisSpec]]


def _is_none(x: Any) -> bool:
    return x is None


def _resolve_axis(axis_spec: AxisSpec, elem: Any) -> PyTree[ResolvedAxisSpec]:
    if axis_spec is None or isinstance(axis_spec, int):
        return jtu.tree_map(lambda _: axis_spec, elem)
    elif callable(axis_spec):
        return jtu.tree_map(axis_spec, elem)
    else:
        raise ValueError(
            "`in_axes` and `out_axes` must consist of None, ints, and callables only."
        )


def _resolve_axes(
    pytree: PyTree[Any], axes_spec: PyTree[AxisSpec]
) -> PyTree[ResolvedAxisSpec]:
    return jtu.tree_map(_resolve_axis, axes_spec, pytree, is_leaf=_is_none)


@dataclasses.dataclass(frozen=True)  # not a pytree
class if_array:
    """Returns a callable that returns the specified integer if evaluated on an array.
    Otherwise, it returns `None`.

    !!! Example

        ```python
        fn = if_array(1)
        # Evaluate on an array, return the integer.
        fn(jax.numpy.array([0, 1, 2]))  # 1
        # Evaluate on not-an-array, return None.
        fn(True)  # None
        ```
    """

    axis: int

    def __call__(self, x: Any) -> Optional[int]:
        return self.axis if is_array(x) else None


def _moveaxis(array, axis):
    return jnp.moveaxis(array, 0, axis)


def _named_in_axes(fun, in_axes, args):
    if isinstance(in_axes, dict):
        in_axes = dict(in_axes)
        new_in_axes = []
        default = if_array(0)
        params = inspect.signature(fun).parameters
        # We may have that len(args) < len(params) due to default arguments.
        # Truncate to considering just the arguments that have been passed.
        #
        # (If len(args) > len(params) then we'll get the usual error later when
        # attempting to call with the wrong number of arguments.)
        for _, param_name in zip(args, params):
            new_in_axes.append(in_axes.pop(param_name, default))
        if len(in_axes) != 0:
            raise ValueError(
                "The following `in_axes` did not correspond to any argument: "
                f"{tuple(in_axes.keys())}"
            )
        # Note that this requires all named arguments to be passed; they cannot take
        # default values. That is, we deliberately don't allow something like
        # ```python
        # @eqx.filter_vmap(in_axes=dict(foo=0))
        # def fn(foo=default_value):
        #     ...
        #
        # fn()
        # ```
        # This is because it is ambiguous whether the default value is vectorised or
        # not.
        in_axes = tuple(new_in_axes)
    return in_axes


class _VmapWrapper(Module):
    _fun: Callable
    _in_axes: PyTree[AxisSpec]
    _out_axes: PyTree[AxisSpec]
    _axis_name: Optional[Hashable]
    _axis_size: Optional[int]
    _vmapkwargs: dict[str, Any]

    @property
    def __wrapped__(self):
        return self._fun

    def __call__(self, /, *args, **kwargs):
        if len(kwargs) != 0:
            raise RuntimeError(
                "keyword arguments cannot be used with functions wrapped with "
                "`filter_vmap`"
            )
        del kwargs

        in_axes = _named_in_axes(self._fun, self._in_axes, args)

        # JAX only actually supports passing array-typed values as inputs.
        # Interestingly it's usually still fine to pass non-array-typed values as
        # broadcast inputs. *Unless* you mess up with an inconsistent axis size, in
        # which case it tries to get the axis sizes of all inputs and then fails on the
        # non-array inputs.
        # e.g.:
        # ```
        # jax.vmap(lambda x, y, z: 0,
        #          in_axes=(0, 0, None))(jnp.arange(3), jnp.arange(5), object())
        # ```
        in_axes = _resolve_axes(args, in_axes)
        unmapped_axis = jtu.tree_map(_is_none, in_axes, is_leaf=_is_none)
        static_args, dynamic_args = partition(args, unmapped_axis)

        def _fun_wrapper(_dynamic_args):
            _args = combine(_dynamic_args, static_args)
            _out = self._fun(*_args)
            _out_axes = _resolve_axes(_out, self._out_axes)
            _none_axes = jtu.tree_map(_is_none, _out_axes, is_leaf=_is_none)
            _nonvmapd, _vmapd = partition(_out, _none_axes, is_leaf=_is_none)
            _nonvmapd_arr, _nonvmapd_static = partition(_nonvmapd, is_array)
            return _vmapd, _nonvmapd_arr, Static((_nonvmapd_static, _out_axes))

        if len(jtu.tree_leaves(in_axes)) == 0 and self._axis_size is None:
            vmapd, nonvmapd_arr, static = _fun_wrapper(dynamic_args)
            if len(jtu.tree_leaves(vmapd)) != 0:
                raise ValueError(
                    "Cannot resolve batch dimension. Non-`None` `out_axes` requires "
                    "either `in_axes` or `axis_size` to be not `None`."
                )
        else:
            vmapd, nonvmapd_arr, static = jax.vmap(
                _fun_wrapper,
                in_axes=(in_axes,),
                out_axes=(0, None, None),
                axis_name=self._axis_name,
                axis_size=self._axis_size,
                **self._vmapkwargs,
            )(dynamic_args)

        nonvmapd_static, out_axes = static.value
        nonvmapd = combine(nonvmapd_arr, nonvmapd_static)

        assert jtu.tree_structure(vmapd) == jtu.tree_structure(out_axes)
        vmapd = jtu.tree_map(_moveaxis, vmapd, out_axes)

        return combine(vmapd, nonvmapd)

    def __get__(self, instance, owner):
        del owner
        if instance is None:
            return self
        return Partial(self, instance)


@overload
def filter_vmap(
    *,
    in_axes: PyTree[AxisSpec] = if_array(0),
    out_axes: PyTree[AxisSpec] = if_array(0),
    axis_name: Hashable = None,
    axis_size: Optional[int] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...


@overload
def filter_vmap(
    fun: Callable[..., Any],
    *,
    in_axes: PyTree[AxisSpec] = if_array(0),
    out_axes: PyTree[AxisSpec] = if_array(0),
    axis_name: Hashable = None,
    axis_size: Optional[int] = None,
) -> Callable[..., Any]: ...


@doc_remove_args("vmapkwargs")
def filter_vmap(
    fun=sentinel,
    *,
    in_axes: PyTree[AxisSpec] = if_array(0),
    out_axes: PyTree[AxisSpec] = if_array(0),
    axis_name: Hashable = None,
    axis_size: Optional[int] = None,
    **vmapkwargs,
):
    """Vectorises a function. By default, all JAX/NumPy arrays are vectorised down their
    leading axis (i.e. axis index 0), and all other types are broadcast.

    **Arguments:**

    For both `in_axes` and `out_axes`, then `int` indicates an array axis to vectorise
    over, `None` indicates that an argument should be broadcast (not vectorised
    over), and callables `Leaf -> Union[None, int]` are mapped and evaluated on every
    leaf of their subtree. `None` should be used for non-JAX-array arguments.

    - `fun` is a pure function to vectorise. Should be of the form `fun(*args)`; that
        is to say it cannot accept keyword arguments.
    - `in_axes` indicates which axes of the input arrays should be vectorised over.
        It should be a PyTree of `None`, `int`, or callables `Leaf -> Union[None, int]`.
        Its tree structure should either be:
        1. a prefix of the input tuple of `args`.
        2. a dictionary, in which case the named arguments use the specified indices
            to vectorise over, and all other arguments will have the default
            `eqx.if_array(0)`.
    - `out_axes` indicates which axis of the output arrays the mapped axis should appear
        at. It should be a PyTree of `None`, `int`, or callables
        `Leaf -> Union[None, int]`, and its tree structure should be a prefix of the
        output `fun(*args)`.
    - `axis_name` is an optional hashable Python object used to identify the mapped
        axis so that parallel collectives (e.g. `jax.lax.psum`) can be applied.
    - `axis_size` is an optional `int` describing the size of the axis mapped. This
        only needs to be passed if none of the input arguments are vectorised, as else
        it can be deduced by looking at the argument shapes.

    **Returns:**

    The vectorised version of `fun`.

    !!! tip

        To vectorise all JAX/NumPy arrays down their `j`th axis, and broadcast all other
        types, then you can use `equinox.if_array(j)`, which returns a callable
        `leaf -> j if is_array(leaf) else None`. For example: the default values of
        `in_axes` and `out_axes` are both `equinox.if_array(0)`.

    !!! example

        ```python
        import equinox as eqx
        import jax.numpy as jnp

        @eqx.filter_vmap
        def f(x, y):
            return x + y

        @eqx.filter_vmap(in_axes=(None, 1))
        def g(x, y):
            return x + y

        f(jnp.array([1, 2]), jnp.array([3, 4]))  # both args vectorised down axis 0

        f(jnp.array([1, 2]), 3)                  # first arg vectorised down axis 0
                                                 # second arg broadcasted

        g(jnp.array(1), jnp.array([[2, 3]]))     # first arg broadcasted
                                                 # second arg vectorised down axis 1
        ```

    !!! example

        `filter_vmap` can be used to easily create ensembles of models. For example,
        here's an ensemble of eight MLPs:

        ```python
        import equinox as eqx
        import jax.random as jr

        key = jr.PRNGKey(0)
        keys = jr.split(key, 8)

        # Create an ensemble of models

        @eqx.filter_vmap
        def make_ensemble(key):
            return eqx.nn.MLP(2, 2, 2, 2, key=key)

        mlp_ensemble = make_ensemble(keys)

        # Evaluate each member of the ensemble on the same data

        @eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
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
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
            **vmapkwargs,
        )

    deprecated_0_10(vmapkwargs, "default")
    deprecated_0_10(vmapkwargs, "fn")
    deprecated_0_10(vmapkwargs, "args")
    deprecated_0_10(vmapkwargs, "kwargs")
    deprecated_0_10(vmapkwargs, "out")

    vmap_wrapper = _VmapWrapper(
        _fun=fun,
        _in_axes=in_axes,
        _out_axes=out_axes,
        _axis_name=axis_name,
        _axis_size=axis_size,
        _vmapkwargs=vmapkwargs,
    )
    return module_update_wrapper(vmap_wrapper)


@compile_cache
def _filter_pmap_cache(
    fun_names, static, struct, in_axes, axis_name, axis_size, pmapkwargs
):
    def fun_abstract(_dynamic):
        fun, args, _, _ = combine(_dynamic, static)
        out = fun(*args)
        return filter(out, is_array_like)

    fun_abstract = jax.vmap(
        fun_abstract, in_axes=(in_axes,), axis_name=axis_name, axis_size=axis_size
    )
    struct_out = jax.eval_shape(fun_abstract, struct)
    max_out_size = jtu.tree_reduce(lambda x, y: max(x, y.ndim), struct_out, 0)
    del fun_abstract, struct, struct_out

    def _check_map_out_axis(x: Optional[int]):
        if isinstance(x, int):
            if x < -max_out_size or x >= max_out_size:
                raise ValueError(
                    "integers in filter_pmap(..., out_axes=...) must correspond to a "
                    "dimension of the output array"
                )
        elif x is not None:
            raise ValueError(
                "filter_pmap(..., out_axes=...) must contain only integers and Nones"
            )

    def fun_wrapped(_dynamic):
        _fun, _args, _, _out_axes = combine(_dynamic, static)
        _out = _fun(*_args)
        _out_axes = _resolve_axes(_out, _out_axes)
        jtu.tree_map(_check_map_out_axis, _out_axes)
        _pmapd = []
        for i in range(-max_out_size, max_out_size):
            _i_axes = jtu.tree_map(lambda a: a == i, _out_axes, is_leaf=_is_none)
            _pmapd.append(filter(_out, _i_axes, is_leaf=_is_none))
        _none_axes = jtu.tree_map(_is_none, _out_axes, is_leaf=_is_none)
        _nonpmapd = filter(_out, _none_axes, is_leaf=_is_none)
        _dynamic_nonpmapd, _static_nonpmapd = hashable_partition(_nonpmapd, is_array)
        return _pmapd, _dynamic_nonpmapd, Static(_static_nonpmapd)

    fun_name, fun_qualname = fun_names
    fun_wrapped.__name__ = fun_name
    fun_wrapped.__qualname__ = fun_qualname

    out_axes = (list(range(-max_out_size, max_out_size)), None, None)

    return jax.pmap(
        fun_wrapped,
        in_axes=(in_axes,),  # pyright: ignore
        out_axes=out_axes,  # pyright: ignore
        axis_name=axis_name,
        axis_size=axis_size,
        **pmapkwargs,
    )


def _common_preprocess(axis_size, kwargs):
    if len(kwargs) != 0:
        raise RuntimeError(
            "keyword arguments cannot be used with functions wrapped with "
            "`filter_pmap`"
        )
    if axis_size is None:
        return 0  # hashable non-array object
    else:
        # Work around JAX issue #9252
        return np.broadcast_to(0, axis_size)


def _preprocess(info, args, kwargs):
    fun, out_axes, axis_size = info
    maybe_dummy = _common_preprocess(axis_size, kwargs)
    del kwargs
    dynamic = filter((fun, args, maybe_dummy, out_axes), is_array)
    return (dynamic,)


def _postprocess(out):
    (pmapd, dynamic_nonpmapd, static_nonpmapd) = out
    nonpmapd = hashable_combine(dynamic_nonpmapd, static_nonpmapd.value)
    return combine(*pmapd, nonpmapd)


class _PmapWrapper(Module):
    _fun: Callable
    _in_axes: PyTree[AxisSpec]
    _out_axes: PyTree[AxisSpec]
    _axis_name: Optional[Hashable]
    _axis_size: Optional[int]
    _filter_warning: bool
    _pmapkwargs: dict[str, Any]

    @property
    def __wrapped__(self):
        return self._fun

    def _call(self, is_lower, args, kwargs):
        maybe_dummy = _common_preprocess(self._axis_size, kwargs)
        del kwargs

        in_axes = _named_in_axes(self._fun, self._in_axes, args)
        in_axes = _resolve_axes(args, in_axes)
        in_axes = (None, in_axes, 0, None)

        dynamic, static = partition(
            (self._fun, args, maybe_dummy, self._out_axes), is_array
        )
        struct = jtu.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), dynamic)

        cached = _filter_pmap_cache(
            self._fun,
            static,
            struct,
            in_axes,
            self._axis_name,
            self._axis_size,
            self._pmapkwargs,
        )

        if is_lower:
            return Lowered(
                cached.lower(dynamic),
                (self._fun, self._out_axes, self._axis_size),
                _preprocess,  # pyright: ignore
                _postprocess,  # pyright: ignore
            )
        else:
            if self._filter_warning is True:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="Some donated buffers were not usable*"
                    )
                    out = cached(dynamic)
            else:
                out = cached(dynamic)
            return _postprocess(out)

    def __call__(self, /, *args, **kwargs):
        return self._call(False, args, kwargs)

    def lower(self, /, *args, **kwargs) -> Lowered:
        return self._call(True, args, kwargs)

    def __get__(self, instance, owner):
        del owner
        if instance is None:
            return self
        return Partial(self, instance)


# Deliberately using `Callable[..., Any]` as `filter_pmap` does change the input and
# out args in ways not expressible in the static type system (changing the number of
# axes).
@overload
def filter_pmap(
    *,
    in_axes: PyTree[AxisSpec] = if_array(0),
    out_axes: PyTree[AxisSpec] = if_array(0),
    axis_name: Hashable = None,
    axis_size: Optional[int] = None,
    donate: Literal["all", "warn", "none"] = "none",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...


@overload
def filter_pmap(
    fun: Callable[..., Any],
    *,
    in_axes: PyTree[AxisSpec] = if_array(0),
    out_axes: PyTree[AxisSpec] = if_array(0),
    axis_name: Hashable = None,
    axis_size: Optional[int] = None,
    donate: Literal["all", "warn", "none"] = "none",
) -> Callable[..., Any]: ...


@doc_remove_args("pmapkwargs")
def filter_pmap(
    fun=sentinel,
    *,
    in_axes: PyTree[AxisSpec] = if_array(0),
    out_axes: PyTree[AxisSpec] = if_array(0),
    axis_name: Hashable = None,
    axis_size: Optional[int] = None,
    donate: Literal["all", "warn", "none"] = "none",
    **pmapkwargs,
):
    """
    !!! warning

        JAX has now added more powerful parallelism APIs directly to the JIT interface.
        As such, using [`equinox.filter_jit`][] with sharded inputs is now recommended
        over `filter_pmap`. See also the
        [parallelism example](../../examples/parallelism/).

    Parallelises a function. By default, all JAX/NumPy arrays are parallelised down
    their leading axis (i.e. axis index 0), and all other types are broadcast.

    `jax.pmap`, and thus `equinox.filter_pmap`, also compiles their function in the same
    way as `jax.jit`. By default, all JAX arrays are traced, and all other arguments are
    treated as static inputs.

    **Arguments:**

    For both `in_axes` and `out_axes`, then `int` indicates an array axis to parallelise
    over, `None` indicates that an argument should be broadcast (not parallelise
    over), and callables `Leaf -> Union[None, int]` are mapped and evaluated on every
    leaf of their subtree. `None` should be used for non-JAX-array arguments.

    - `fun` is a pure function to parallelise. Should be of the form `fun(*args)`; that
        is to say it cannot accept keyword arguments.
    - `in_axes` indicates which axes of the input arrays should be parallelised over.
        It should be a PyTree of `None`, `int`, or callables `Leaf -> Union[None, int]`.
        Its tree structure should either be:
        1. a prefix of the input tuple of `args`.
        2. a dictionary, in which case the named arguments use the specified indices
            to parallelise over, and all other arguments will have the default
            `eqx.if_array(0)`.
    - `out_axes` indicates which axis of the output arrays the mapped axis should appear
        at. It should be a PyTree of `None`, `int`, or callables
        `Leaf -> Union[None, int]`, and its tree structure should be a prefix of the
        output `fun(*args)`.
    - `axis_name` is an optional hashable Python object used to identify the mapped
        axis so that parallel collectives (e.g. `jax.lax.psum`) can be applied.
    - `axis_size` is an optional `int` describing the size of the axis mapped. This
        only needs to be passed if none of the input arguments are vectorised, as else
        it can be deduced by looking at the argument shapes.
    - `donate` indicates whether the buffers of JAX arrays are donated or not, it
        should either be:
        - `'all'`: donate all arrays and suppress all warnings about
            unused buffers;
        - `'warn'`: as above, but don't suppress unused buffer warnings;
        - `'none'`: the default, disables buffer donation.

    **Returns:**

    The parallelised version of `fun`.

    !!! tip

        To parallelise all JAX/NumPy arrays down their `j`th axis, and broadcast all
        other types, then you can use `equinox.if_array(j)`, which returns a callable
        `leaf -> j if is_array(leaf) else None`. For example: the default values of
        `in_axes` and `out_axes` are both `equinox.if_array(0)`.

    !!! example

        ```python
        import equinox as eqx
        import jax.numpy as jnp

        @eqx.filter_pmap
        def f(x, y):
            return x + y

        @eqx.filter_pmap(in_axes=(None, 1))
        def g(x, y):
            return x + y

        f(jnp.array([1, 2]), jnp.array([3, 4]))  # both args parallelised down axis 0

        f(jnp.array([1, 2]), 3)                  # first arg parallelised down axis 0
                                                 # second arg broadcasted (as it's not
                                                 # a JAX array)

        g(jnp.array(1), jnp.array([[2, 3]]))     # first arg broadcasted
                                                 # second arg parallelised down axis 1
        ```
    """

    if fun is sentinel:
        return ft.partial(
            filter_pmap,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
            donate=donate,
            **pmapkwargs,
        )

    deprecated_0_10(pmapkwargs, "default")
    deprecated_0_10(pmapkwargs, "fn")
    deprecated_0_10(pmapkwargs, "args")
    deprecated_0_10(pmapkwargs, "kwargs")
    deprecated_0_10(pmapkwargs, "out")
    if any(x in pmapkwargs for x in ("static_broadcasted_argnums", "donate_argnums")):
        raise ValueError(
            "`pmapkwargs` cannot contain either 'static_broadcasted_argnums' or "
            "'donate_argnums'"
        )

    if donate == "arrays":
        warnings.warn(
            "The `donate='arrays'` option to `filter_pmap` has been renamed to "
            "`donate='all'`",
            DeprecationWarning,
        )
        donate = "all"
    if donate not in {"all", "warn", "none"}:
        raise ValueError(
            "`filter_jit(..., donate=...)` must be one of 'all', 'warn', or 'none'"
        )
    filter_warning = True if donate == "all" else False
    if donate != "none":
        pmapkwargs["donate_argnums"] = (0,)

    pmap_wrapper = _PmapWrapper(
        _fun=fun,
        _in_axes=in_axes,
        _out_axes=out_axes,
        _axis_name=axis_name,
        _axis_size=axis_size,
        _filter_warning=filter_warning,
        _pmapkwargs=pmapkwargs,
    )
    return module_update_wrapper(pmap_wrapper)
