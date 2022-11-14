import functools as ft
import types
import typing
import warnings
from typing import Any, Callable, Dict

import jax
import jax.interpreters.ad as ad
import jax.tree_util as jtu
from jaxtyping import Array, PyTree

from .custom_types import BoolAxisSpec, sentinel
from .doc_utils import doc_strip_annotations
from .filters import (
    combine,
    is_array,
    is_inexact_array,
    is_inexact_array_like,
    partition,
)
from .make_jaxpr import filter_make_jaxpr
from .module import Module, module_update_wrapper, Static


class _ValueAndGradWrapper(Module):
    _fun: Callable
    _arg: PyTree[BoolAxisSpec]
    _gradkwargs: Dict[str, Any]

    # Try to avoid clashes with existing argument names.
    # TODO: use "/" once we're on Python 3.8.
    def __call__(__self, __x, *args, **kwargs):
        @ft.partial(jax.value_and_grad, argnums=0, **__self._gradkwargs)
        def fun_value_and_grad(_diff_x, _nondiff_x, *_args, **_kwargs):
            _x = combine(_diff_x, _nondiff_x)
            return __self._fun(_x, *_args, **_kwargs)

        diff_x, nondiff_x = partition(__x, __self._arg)
        return fun_value_and_grad(diff_x, nondiff_x, *args, **kwargs)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return jtu.Partial(self, instance)


class _GradWrapper(Module):
    _fun_value_and_grad: _ValueAndGradWrapper
    _has_aux: bool

    def __call__(__self, *args, **kwargs):
        value, grad = __self._fun_value_and_grad(*args, **kwargs)
        if __self._has_aux:
            _, aux = value
            return grad, aux
        else:
            return grad

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return jtu.Partial(self, instance)


@doc_strip_annotations
def filter_value_and_grad(
    fun: Callable = sentinel,
    *,
    arg: PyTree[BoolAxisSpec] = is_inexact_array,
    **gradkwargs,
) -> Callable:
    """As [`equinox.filter_grad`][], except that it is `jax.value_and_grad` that is
    wrapped.
    """

    if fun is sentinel:
        return ft.partial(filter_value_and_grad, arg=arg, **gradkwargs)

    filter_spec = gradkwargs.pop("filter_spec", None)
    if filter_spec is not None:
        warnings.warn("For brevity the `filter_spec` argument has been renamed `arg`")
        arg = filter_spec

    argnums = gradkwargs.pop("argnums", None)
    if argnums is not None:
        raise ValueError(
            "`argnums` should not be passed. If you need to differentiate "
            "multiple objects then collect them into a tuple and pass that "
            "as the first argument."
        )

    return module_update_wrapper(_ValueAndGradWrapper(fun, arg, gradkwargs), fun)


@doc_strip_annotations
def filter_grad(
    fun: Callable = sentinel,
    *,
    arg: PyTree[BoolAxisSpec] = is_inexact_array,
    **gradkwargs,
):
    """As `jax.grad`, but accepts arbitrary PyTrees as inputs. (Not just JAXable types.)

    !!! info

        By default, all inexact (floating-point) JAX arrays are differentiated. Any
        nondifferentiable leaves will have `None` as the gradient.


    **Arguments:**

    - `fun` is a pure function to JIT compile.
    - `arg` is a PyTree whose structure should be a prefix of the structure of
        the **first** argument to `fun`. It behaves as the `filter_spec` argument to
        [`equinox.filter`][]. Truthy values will be differentiated; falsey values will
        not.
    - `**gradkwargs` are any other keyword arguments to `jax.grad`.

    **Returns:**

    A function computing the derivative of `fun` with respect to its first input. Any
    nondifferentiable leaves will have `None` as the gradient. See
    [`equinox.apply_updates`][] for a convenience function that will only attempt to
    apply non-`None` updates.

    !!! tip

        If you need to differentiate multiple objects, then put them together into a
        tuple and pass that through the first argument:
        ```python
        # We want to differentiate `func` with respect to both `x` and `y`.
        def func(x, y):
            ...

        @equinox.filter_grad
        def grad_func(x__y):
            x, y = x__y
            return func(x, y)
        ```
    """

    if fun is sentinel:
        return ft.partial(filter_grad, arg=arg, **gradkwargs)

    has_aux = gradkwargs.get("has_aux", False)

    fun_value_and_grad = filter_value_and_grad(fun, arg=arg, **gradkwargs)
    return module_update_wrapper(_GradWrapper(fun_value_and_grad, has_aux), fun)


def _is_none(x):
    return x is None


def _is_jvp_tracer(x):
    return isinstance(x, ad.JVPTracer)


def filter_jvp(fn, primals, tangents):
    """Like `jax.jvp`, but accepts arbitrary PyTrees. (Not just JAXable types.)

    **Arguments:**

    - `fn`: Function to be differentiated. Its arguments can be Python objects, and
        its return type can be any Python object.
    - `primals`: The primal values at which `fn` should be evaluated. Should be a
        sequence of arguments, and its length should be equal to the number of
        positional parameter of `fn`.
    - `tangents`: The tangent vector for which the Jacobian-vector product should be
        calculated. Should be a PyTree with the same structure as `primals`. The leaves
        of `tangents` must be either floating-point JAX arrays, or Python floats, or
        `None`s. The tangent must be `None` for any primal which is not itself a
        floating-point JAX array or Python float.

    **Returns:**

    A pair `(primals_out, tangents_out)` is returned,
    where `primals_out = fn(*primals)` and `tangents_out` is the Jacobian-vector
    product of `fn` evaluated at `primals` with `tangents`.

    The `tangents_out` has the same structure as `primals_out`, but has `None` for
    any leaves that aren't differentiable.

    !!! Tip

        Unlike `jax.jvp`, this function does not support a `has_aux` argument. It isn't
        needed, as unlike `jax.jvp` the output of this function can be of arbitrary type.
    """
    if jtu.tree_structure(primals, is_leaf=_is_none) != jtu.tree_structure(
        tangents, is_leaf=_is_none
    ):
        raise ValueError("primals and tangents must have the same pytree structure")
    filter_spec = jtu.tree_map(_is_none, tangents, is_leaf=_is_none)
    static_primals, dynamic_primals = partition(primals, filter_spec)
    flat_dynamic_primals, treedef = jtu.tree_flatten(dynamic_primals)
    flat_tangents = jtu.tree_leaves(tangents)  # all non-None tangents are dynamic

    def _fn(*_flat_dynamic):
        _dynamic = jtu.tree_unflatten(treedef, _flat_dynamic)
        _in = combine(_dynamic, static_primals)
        _out = fn(*_in)
        _dynamic_out, _static_out = partition(_out, _is_jvp_tracer)
        return _dynamic_out, Static(_static_out)

    primal_out, tangent_out = jax.jvp(_fn, flat_dynamic_primals, flat_tangents)
    dynamic_primal_out, static_primal_out = primal_out
    primal_out = combine(dynamic_primal_out, static_primal_out.value)
    tangent_out, _ = tangent_out

    return primal_out, tangent_out


def filter_vjp(fun, *primals, has_aux=False):
    """Filtered version of `jax.vjp`.

    **Arguments:**

    - `fun`: The function to be differentiated. Will be called as `fun(*primals)`. Can
        return an arbitrary PyTree.
    - `primals`: The arguments at which `fun` will be evaluated and differentiated.
        Can be arbitrary PyTrees.
    - `has_aux`: Indicates whether `fun` returns a pair, with the first element the
        output to be differentiated, and the latter auxiliary data. Defaults to `False`.

    **Returns:**

    If `has_aux is False` then returns a `(primals_out, vjpfun)` pair, where
    `primals_out = fun(*primals)` and `vjpfun` is a function from a cotangent vector
    with the same shape as `primals_out` to a tuple of cotangent vectors with the same
    shape as `primals`, representing the vector-Jacobian product of `fun` evaluated at
    `primals`.

    If `has_aux is True` then returns a tuple `(primals_out, vjpfun, aux)`, where `aux`
    is the auxiliary data returned from `fun`.

    The cotangent passed to `vjpfun` should have arrays corresponding to all
    floating-point arrays in `primals_out`, and `None` for all other PyTree leaves. The
    cotangents returned from `vjpfun` will likewise have arrays for all `primals` that
    are floating-point arrays, and `None` for all other PyTree leaves.
    """
    diff, nondiff = partition(primals, is_inexact_array)

    def diff_fun(*_diff):
        _primals = combine(_diff, nondiff)
        _out = fun(*_primals)
        if has_aux:
            _out, _aux = _out
        else:
            _aux = None
        _diff_out, _nondiff_out = partition(_out, is_inexact_array)
        return _diff_out, (_nondiff_out, _aux)

    diff_out, vjp_fn, (nondiff_out, aux) = jax.vjp(diff_fun, *diff, has_aux=True)
    out = combine(diff_out, nondiff_out)
    if has_aux:
        return out, vjp_fn, aux
    else:
        return out, vjp_fn


class _ClosureConvert(Module):
    jaxpr: jax.core.Jaxpr
    consts: PyTree[Array]  # Captured in the PyTree structure of _ClosureConvert
    out_dynamic_struct: PyTree[jax.ShapeDtypeStruct]
    out_static: PyTree[Any]

    def __call__(self, *args, **kwargs):
        dynamic = filter((args, kwargs), is_array)
        dynamic_flat = jtu.tree_leaves(dynamic)
        out_dynamic_flat = jax.core.eval_jaxpr(self.jaxpr, self.consts, *dynamic_flat)
        out_dynamic_struct_flat, out_dynamic_treedef = jtu.tree_flatten(
            self.out_dynamic_struct
        )
        assert len(out_dynamic_flat) == len(out_dynamic_struct_flat)
        for o1, o2 in zip(out_dynamic_flat, out_dynamic_struct_flat):
            assert o1.shape == o2.shape
            assert o1.dtype == o2.dtype
        out = jtu.tree_unflatten(out_dynamic_treedef, out_dynamic_flat)
        out = combine(out, self.out_static)
        return out


def filter_closure_convert(fn, *args, **kwargs):
    """As `jax.closure_convert`, but works on functions accepting and returning
    arbtirary PyTree objects. In addition, all JAX arrays are hoisted into constants
    (not just floating point arrays).

    This is useful for explicitly capturing any closed-over JAX tracers
    before crossing an API boundary, such as `jax.grad`, `jax.custom_vjp`, or the
    rule of a custom primitive.

    **Arguments:**

    - `fn`: The function to call. Will be called as `fun(*args, **kwargs)`.
    - `args`, `kwargs`: Example arguments at which to call the function. The function is
        not actually evaluated on these arguments; all JAX arrays are subsituted for
        tracers. Note that Python builtins (`bool`, `int`, `float`, `complex`) are
        not substituted for tracers and are passed through as-is.

    **Returns:**

    A new function, which can be called in the same way, using `*args` and `**kwargs`.
    Will contain all closed-over tracers of `fn` as part of its PyTree structure.

    !!! Example

        ```python
        @jax.grad
        def f(x, y):
            z = x + y
            g = lambda a: z + a  # closes over z
            g2 = filter_closure_convert(g, 1)
            assert [id(b) for b in g2.consts] == [id(z)]
            return z

        f(1., 1.)
        ```
    """
    if fn.__closure__ is None:
        # In this case, it's not possible to have any closed-over tracers.
        return fn
    closed_jaxpr, out_dynamic_struct, out_static = filter_make_jaxpr(fn)(
        *args, **kwargs
    )
    jaxpr = closed_jaxpr.jaxpr
    consts = closed_jaxpr.consts
    return _ClosureConvert(jaxpr, consts, out_dynamic_struct, out_static)


class filter_custom_jvp:
    """Filtered version of `jax.custom_jvp`.

    Works in the same way as `jax.custom_jvp`, except that you do not need to specify
    `nondiff_argnums`. Instead, arguments are automatically split into differentiable
    and nondifferentiable based on whether or not they are a floating-point JAX array.

    The tangents of the nondifferentiable arguments will be passed as `None`.

    The return types must still all be JAX types.

    Example:
    ```python
    @equinox.filter_custom_jvp
    def call(fn, x):
        return fn(x)

    @call.defjvp
    def call_jvp(primals, tangents):
        fn, x = primals
        _, tx = tangents
        primal_out = call(fn, x)
        tangent_out = tx**2
        return primal_out, tangent_out
    ```
    """

    def __init__(self, fn):
        def fn_wrapper(static, dynamic):
            return fn(*combine(dynamic, static))

        self.fn = jax.custom_jvp(fn_wrapper, nondiff_argnums=(0,))

    def defjvp(self, fn_jvp):
        def fn_jvp_wrapper(static, dynamic, tangents):
            (dynamic,) = dynamic
            (tangents,) = tangents
            primals = combine(dynamic, static)
            return fn_jvp(primals, tangents)

        self.fn.defjvp(fn_jvp_wrapper)

    def defjvps(self, *a, **kw):
        raise NotImplementedError("filter_custom_jvp().defjvps is not implemented")

    def __call__(self, *args):
        dynamic, static = partition(args, is_inexact_array_like)
        return self.fn(static, dynamic)


class filter_custom_vjp:
    """As `jax.custom_vjp`, but with a nicer interface.

    Usage is:
    ```python
    @equinox.filter_custom_vjp
    def fn(vjp_arg, *args, **kwargs):
        # vjp_arg is some PyTree of arbitrary Python objects.
        # args, kwargs contain arbitrary Python objects.
        ...
        return obj  # some PyTree of arbitrary Python objects.

    def fn_fwd(vjp_arg, *args, **kwargs):
        ...
        # Should return `obj` as before. `residuals` can be any collection of JAX
        # arrays you want to keep around for the backward pass.
        return obj, residuals

    def fn_bwd(residuals, grad_obj, vjp_arg, *args, **kwargs):
        # grad_obj will have `None` as the gradient for any leaves of `obj` that were
        # not JAX arrays
        ...
        # grad_vjp_arg should have `None` as the gradient for any leaves of `vjp_arg`
        # that were not JAX arrays.
        return grad_vjp_arg

    fn.defvjp(fn_fwd, fn_bwd)
    ```

    The key differences to `jax.custom_vjp` are that:

    - Only the gradient of the first argument, `vjp_arg`, should be computed on the
        backward pass. Everything else will automatically have zero gradient.
    - You do not need to distinguish differentiable from nondifferentiable manually.
        Instead you should return gradients for all inexact JAX arrays in the first
        argument. (And just put `None` on every other leaf of the PyTree.)
    - As a convenience, all of the inputs from the forward pass are additionally made
        available to you on the backward pass.

    !!! tip

        If you need gradients with respect to multiple arguments, then just pack them
        together as a tuple via the first argument `vjp_arg`. (See also
        [`equinox.filter_grad`][] for a similar trick.)
    """

    def __init__(self, fn):
        self.fn = fn
        self.fn_wrapped = None

    def defvjp(self, fn_fwd, fn_bwd):
        def fn_wrapped(
            nonarray_vjp_arg,
            nonarray_args_kwargs,
            diff_array_vjp_arg,
            nondiff_array_vjp_arg,
            array_args_kwargs,
        ):
            vjp_arg = combine(
                nonarray_vjp_arg, diff_array_vjp_arg, nondiff_array_vjp_arg
            )
            args, kwargs = combine(nonarray_args_kwargs, array_args_kwargs)
            out = self.fn(vjp_arg, *args, **kwargs)
            array_out, nonarray_out = partition(out, is_array)
            diff_array_out, nondiff_array_out = partition(array_out, is_inexact_array)
            return diff_array_out, nondiff_array_out, Static(nonarray_out)

        def fn_fwd_wrapped(
            nonarray_vjp_arg,
            nonarray_args_kwargs,
            diff_array_vjp_arg,
            nondiff_array_vjp_arg,
            array_args_kwargs,
        ):
            vjp_arg = combine(
                nonarray_vjp_arg, diff_array_vjp_arg, nondiff_array_vjp_arg
            )
            args, kwargs = combine(nonarray_args_kwargs, array_args_kwargs)
            out, residuals = fn_fwd(vjp_arg, *args, **kwargs)
            array_out, nonarray_out = partition(out, is_array)
            diff_array_out, nondiff_array_out = partition(array_out, is_inexact_array)
            out = diff_array_out, nondiff_array_out, Static(nonarray_out)
            return out, (
                residuals,
                diff_array_vjp_arg,
                nondiff_array_vjp_arg,
                array_args_kwargs,
            )

        def fn_bwd_wrapped(nonarray_vjp_arg, nonarray_args_kwargs, residuals, grad_out):
            (
                residuals,
                diff_array_vjp_arg,
                nondiff_array_vjp_arg,
                array_args_kwargs,
            ) = residuals
            vjp_arg = combine(
                nonarray_vjp_arg, diff_array_vjp_arg, nondiff_array_vjp_arg
            )
            args, kwargs = combine(nonarray_args_kwargs, array_args_kwargs)
            grad_diff_array_out, _, _ = grad_out
            out = fn_bwd(residuals, grad_diff_array_out, vjp_arg, *args, **kwargs)
            if jtu.tree_structure(out) != jtu.tree_structure(diff_array_vjp_arg):
                raise RuntimeError(
                    "custom_vjp gradients must have the same structure as "
                    "`equinox.filter(vjp_arg, equinox.is_inexact_array)`, where "
                    "`vjp_arg` is the first argument used in the forward pass."
                )
            # None is the gradient through nondiff_array_vjp_arg and array_args_kwargs
            return out, None, None

        fn_wrapped = jax.custom_vjp(fn_wrapped, nondiff_argnums=(0, 1))
        fn_wrapped.defvjp(fn_fwd_wrapped, fn_bwd_wrapped)
        self.fn_wrapped = fn_wrapped

    def __call__(__self, __vjp_arg, *args, **kwargs):
        # Try and avoid name collisions with the arguments of the wrapped function.
        # TODO: once we switch to Python 3.8, use (self, vjp_arg, /, *args, **kwargs).
        self = __self
        vjp_arg = __vjp_arg
        del __self, __vjp_arg
        if self.fn_wrapped is None:
            raise RuntimeError(f"defvjp not yet called for {self.fn.__name__}")
        array_vjp_arg, nonarray_vjp_arg = partition(vjp_arg, is_array)
        diff_array_vjp_arg, nondiff_array_vjp_arg = partition(
            array_vjp_arg, is_inexact_array
        )
        array_args_kwargs, nonarray_args_kwargs = partition((args, kwargs), is_array)
        out = self.fn_wrapped(
            nonarray_vjp_arg,
            nonarray_args_kwargs,
            diff_array_vjp_arg,
            nondiff_array_vjp_arg,
            array_args_kwargs,
        )
        diff_array_out, nondiff_array_out, nonarray_out = out
        return combine(diff_array_out, nondiff_array_out, nonarray_out.value)


if getattr(typing, "GENERATING_DOCUMENTATION", False):
    _filter_custom_jvp_doc = filter_custom_jvp.__doc__
    _filter_custom_vjp_doc = filter_custom_vjp.__doc__

    def defjvp(fn_jvp):
        pass

    def filter_custom_jvp(fn):
        return types.SimpleNamespace(defjvp=defjvp)

    def defvjp(fn_fwd, fn_bwd):
        pass

    def filter_custom_vjp(fn):
        return types.SimpleNamespace(defvjp=defvjp)

    filter_custom_jvp.__doc__ = _filter_custom_jvp_doc
    filter_custom_vjp.__doc__ = _filter_custom_vjp_doc
