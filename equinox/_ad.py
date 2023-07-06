import functools as ft
import types
import typing
import warnings
from collections.abc import Callable, Sequence
from typing import (
    Any,
    cast,
    Literal,
    Optional,
    overload,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
from typing_extensions import ParamSpec

import jax
import jax.core
import jax.interpreters.ad as ad
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, Complex, Float, PyTree, PyTreeDef

from ._custom_types import sentinel
from ._deprecate import deprecated_0_10
from ._doc_utils import doc_remove_args
from ._filters import (
    combine,
    is_array,
    is_inexact_array,
    partition,
)
from ._make_jaxpr import filter_make_jaxpr
from ._module import field, Module, module_update_wrapper, Partial, Static
from ._tree import tree_equal


_P = ParamSpec("_P")
_T = TypeVar("_T")
_S = TypeVar("_S")


class _ValueAndGradWrapper(Module):
    _fun: Callable
    _has_aux: bool
    _gradkwargs: dict[str, Any]

    @property
    def __wrapped__(self):
        return self._fun

    def __call__(self, x, /, *args, **kwargs):
        @ft.partial(jax.value_and_grad, has_aux=self._has_aux, **self._gradkwargs)
        def fun_value_and_grad(_diff_x, _nondiff_x, *_args, **_kwargs):
            _x = combine(_diff_x, _nondiff_x)
            return self._fun(_x, *_args, **_kwargs)

        diff_x, nondiff_x = partition(x, is_inexact_array)
        return fun_value_and_grad(diff_x, nondiff_x, *args, **kwargs)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return Partial(self, instance)


class _GradWrapper(Module):
    _fun_value_and_grad: _ValueAndGradWrapper
    _has_aux: bool

    @property
    def __wrapped__(self):
        return self._fun_value_and_grad

    def __call__(self, /, *args, **kwargs):
        value, grad = self._fun_value_and_grad(*args, **kwargs)
        if self._has_aux:
            _, aux = value
            return grad, aux
        else:
            return grad

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return Partial(self, instance)


_Scalar = Union[float, complex, Float[ArrayLike, ""], Complex[ArrayLike, ""]]


@overload
def filter_value_and_grad(
    *, has_aux: Literal[False] = False
) -> Callable[[Callable[_P, _Scalar]], Callable[_P, tuple[_Scalar, PyTree]]]:
    ...


@overload
def filter_value_and_grad(
    fun: Callable[_P, _Scalar], *, has_aux: Literal[False] = False
) -> Callable[_P, tuple[_Scalar, PyTree]]:
    ...


@overload
def filter_value_and_grad(
    *, has_aux: Literal[True] = True
) -> Callable[
    [Callable[_P, tuple[_Scalar, _T]]], Callable[_P, tuple[tuple[_Scalar, _T], PyTree]]
]:
    ...


@overload
def filter_value_and_grad(
    fun: Callable[_P, tuple[_Scalar, _T]], *, has_aux: Literal[True] = True
) -> Callable[_P, tuple[tuple[_Scalar, _T], PyTree]]:
    ...


@overload
def filter_value_and_grad(
    fun: Callable[_P, _T], *, has_aux: bool = False
) -> Callable[_P, tuple[_T, PyTree]]:
    ...


@doc_remove_args("gradkwargs")
def filter_value_and_grad(
    fun=sentinel, *, has_aux: bool = False, **gradkwargs
) -> Callable:
    """Creates a function that evaluates both `fun` and the gradient of `fun`.

    The gradient will be computed with respect to all floating-point JAX/NumPy arrays
    in the first argument. (Which should be a PyTree.)

    Any nondifferentiable leaves in the first argument will have `None` as the gradient.

    **Arguments:**

    - `fun` is a pure function to differentiate.
    - `has_aux`: if `True` then `fun` should return a pair; the first element is the
        output to be differentiated and the second element is auxiliary data.

    **Returns:**

    A function with the same arguments as `fun`, that evaluates both `fun` and computes
    the derivative of `fun` with respect to its first input. Any nondifferentiable
    leaves will have `None` as the gradient.

    If `has_aux` is `True` then a nested tuple `((value, aux), gradient)` is returned.
    If `has_aux` is `False` then the pair `(value, gradient)` is returned.
    """

    if fun is sentinel:
        return ft.partial(filter_value_and_grad, has_aux=has_aux, **gradkwargs)

    deprecated_0_10(gradkwargs, "arg")
    deprecated_0_10(gradkwargs, "filter_spec")
    argnums = gradkwargs.pop("argnums", None)
    if argnums is not None:
        raise ValueError(
            "`argnums` should not be passed. If you need to differentiate "
            "multiple objects then collect them into a tuple and pass that "
            "as the first argument."
        )

    return module_update_wrapper(_ValueAndGradWrapper(fun, has_aux, gradkwargs), fun)


@overload
def filter_grad(
    *, has_aux: Literal[False] = False
) -> Callable[[Callable[_P, _Scalar]], Callable[_P, PyTree[Float[Array, "..."]]]]:
    ...


@overload
def filter_grad(
    fun: Callable[_P, _Scalar], *, has_aux: Literal[False] = False
) -> Callable[_P, PyTree[Float[Array, "..."]]]:
    ...


@overload
def filter_grad(
    *, has_aux: Literal[True] = True
) -> Callable[
    [Callable[_P, tuple[_Scalar, _T]]],
    Callable[_P, tuple[PyTree[Float[Array, "..."]], _T]],
]:
    ...


@overload
def filter_grad(
    fun: Callable[_P, tuple[_Scalar, _T]], *, has_aux: Literal[True] = True
) -> Callable[_P, tuple[PyTree[Float[Array, "..."]], _T]]:
    ...


@overload
def filter_grad(fun: Callable[_P, Any], *, has_aux: bool = False) -> Callable[_P, Any]:
    ...


@doc_remove_args("gradkwargs")
def filter_grad(fun=sentinel, *, has_aux: bool = False, **gradkwargs):
    """Creates a function that computes the gradient of `fun`.

    The gradient will be computed with respect to all floating-point JAX/NumPy arrays
    in the first argument. (Which should be a PyTree.)

    Any nondifferentiable leaves in the first argument will have `None` as the gradient.

    **Arguments:**

    - `fun` is a pure function to differentiate.
    - `has_aux`: if `True` then `fun` should return a pair; the first element is the
        output to be differentiated and the second element is auxiliary data.

    **Returns:**

    A function with the same arguments as `fun`, that computes the derivative of `fun`
    with respect to its first input. Any nondifferentiable leaves will have `None` as
    the gradient.

    If `has_aux` is `True` then a pair `(gradient, aux)` is returned. If `has_aux` is
    `False` then just the `gradient` is returned.

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

    !!! info

        See also [`equinox.apply_updates`][] for a convenience function that applies
        non-`None` gradient updates to a model.

    """

    if fun is sentinel:
        return ft.partial(filter_grad, has_aux=has_aux, **gradkwargs)

    fun_value_and_grad = filter_value_and_grad(fun, has_aux=has_aux, **gradkwargs)
    fun_value_and_grad = cast(_ValueAndGradWrapper, fun_value_and_grad)
    return module_update_wrapper(_GradWrapper(fun_value_and_grad, has_aux), fun)


def _is_none(x):
    return x is None


def _is_jvp_tracer(x):
    return isinstance(x, ad.JVPTracer)


def filter_jvp(
    fn: Callable[..., _T], primals: Sequence, tangents: Sequence
) -> tuple[_T, PyTree]:
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
        needed, as unlike `jax.jvp` the output of this function can be of arbitrary
        type.
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


@overload
def filter_vjp(
    fun: Callable[..., _T], *primals, has_aux: Literal[False] = False
) -> tuple[_T, Callable[..., tuple[PyTree, ...]]]:
    ...


@overload
def filter_vjp(
    fun: Callable[..., tuple[_T, _S]], *primals, has_aux: Literal[True] = True
) -> tuple[_T, Callable[..., tuple[PyTree, ...]], _S]:
    ...


def filter_vjp(fun, *primals, has_aux: bool = False):
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


def _is_struct(x):
    return is_array(x) or isinstance(x, jax.ShapeDtypeStruct)


def _unflatten(flat_pytree):
    leaves, treedef = flat_pytree
    return jtu.tree_unflatten(treedef, leaves)


_T = TypeVar("_T")
_FlatPyTree = tuple[list[_T], PyTreeDef]


class _ClosureConvert(Module):
    # Important that `jaxpr` be a leaf (and not static), so that it is a tuple element
    # when passing through `filter_primitive_bind` and thus visible to
    # `jax.core.subjaxprs`
    jaxpr: jax.core.Jaxpr
    consts: PyTree[Array]  # Captured in the PyTree structure of _ClosureConvert
    in_dynamic_struct: _FlatPyTree[jax.ShapeDtypeStruct] = field(static=True)
    out_dynamic_struct: _FlatPyTree[jax.ShapeDtypeStruct] = field(static=True)
    in_static: _FlatPyTree[Any] = field(static=True)
    out_static: _FlatPyTree[Any] = field(static=True)

    def __call__(self, *args, **kwargs):
        self_in_dynamic_struct = _unflatten(self.in_dynamic_struct)
        self_out_dynamic_struct = _unflatten(self.out_dynamic_struct)
        self_in_static = _unflatten(self.in_static)
        self_out_static = _unflatten(self.out_static)
        in_dynamic, in_static = partition((args, kwargs), is_array)
        in_dynamic_struct = jax.eval_shape(lambda: in_dynamic)
        # `is` because `tree_equal` may return a tracer
        if tree_equal(in_dynamic_struct, self_in_dynamic_struct) is not True:
            raise ValueError(
                "Closure-converted function called with different dynamic arguments to "
                "the example arguments provided."
            )
        if tree_equal(in_static, self_in_static) is not True:
            raise ValueError(
                "Closure-converted function called with different static arguments to "
                "the example arguments provided."
            )
        in_dynamic_flat = jtu.tree_leaves(in_dynamic)
        out_dynamic_flat = jax.core.eval_jaxpr(
            self.jaxpr, self.consts, *in_dynamic_flat
        )
        out_dynamic_struct_flat, out_dynamic_treedef = jtu.tree_flatten(
            self_out_dynamic_struct
        )
        assert len(out_dynamic_flat) == len(out_dynamic_struct_flat)
        for o1, o2 in zip(out_dynamic_flat, out_dynamic_struct_flat):
            assert jnp.shape(o1) == jnp.shape(o2)
            assert jnp.result_type(o1) == jnp.result_type(o2)
        out = jtu.tree_unflatten(out_dynamic_treedef, out_dynamic_flat)
        out = combine(out, self_out_static)
        return out


def filter_closure_convert(fn: Callable[_P, _T], *args, **kwargs) -> Callable[_P, _T]:
    """As `jax.closure_convert`, but works on functions accepting and returning
    arbitrary PyTree objects. In addition, all JAX arrays are hoisted into constants
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
    if isinstance(fn, types.FunctionType) and fn.__closure__ is None:
        # In this case, it's not possible to have any closed-over tracers.
        # Skip jaxpr tracing for efficiency.
        return fn
    closed_jaxpr, out_dynamic_struct, out_static = filter_make_jaxpr(fn)(
        *args, **kwargs
    )  # pyright: ignore
    in_dynamic, in_static = partition((args, kwargs), _is_struct)
    in_dynamic_struct = jax.eval_shape(lambda: in_dynamic)
    jaxpr = closed_jaxpr.jaxpr
    consts = closed_jaxpr.consts
    in_dynamic_struct = jtu.tree_flatten(in_dynamic_struct)
    out_dynamic_struct = jtu.tree_flatten(out_dynamic_struct)
    in_static = jtu.tree_flatten(in_static)
    out_static = jtu.tree_flatten(out_static)
    closure_converted = _ClosureConvert(
        jaxpr, consts, in_dynamic_struct, out_dynamic_struct, in_static, out_static
    )
    closure_converted = cast(Callable[_P, _T], closure_converted)
    return closure_converted


def _materialise_symbolic_zero(x, grad_x):
    if grad_x is None and is_inexact_array(x):
        return jnp.zeros_like(x)
    else:
        return grad_x


def _drop_nondiff(tangent, primal):
    if isinstance(tangent, jax.custom_derivatives.SymbolicZero):
        return None
    elif jnp.issubdtype(jnp.result_type(primal), jnp.inexact):
        # Work around JAX issue #16000
        return tangent
    else:
        return None


class filter_custom_jvp:
    """Filtered version of `jax.custom_jvp`.

    Works in the same way as `jax.custom_jvp`, except that you do not need to specify
    `nondiff_argnums`. Instead, arguments are automatically split into differentiable
    and nondifferentiable. (Everything that is not a floating-point array is necessarily
    nondifferentiable. In addition, some floating-point arrays may happen not to have
    been differentiated.)

    The tangents of the nondifferentiable arguments will be passed as `None`.

    The return types must still all be JAX types.

    Supports keyword arguments, which are always treated as nondifferentiable.

    !!! Example

        ```python
        @equinox.filter_custom_jvp
        def call(x, y, *, fn):
            return fn(x, y)

        @call.def_jvp
        def call_jvp(primals, tangents, *, fn):
            x, y = primals
            tx, ty = tangents
            primal_out = call(x, y, fn=fn)
            tangent_out = tx**2 + ty
            return primal_out, tangent_out
        ```
    """

    def __init__(self, fn):
        def fn_wrapper(static, dynamic):
            args, kwargs = combine(dynamic, static)
            return fn(*args, **kwargs)

        self.fn = jax.custom_jvp(fn_wrapper, nondiff_argnums=(0,))

    def defjvp(self, fn_jvp):
        warnings.warn(
            "As of Equinox 0.10.7, `equinox.filter_custom_jvp.defjvp` is deprecated in "
            "favour of `.def_jvp`. This new API supports symbolic zeros, which allow "
            "for more efficient autodifferentiation rules. In particular:, `None` was "
            "previously passed to indicate a symbolic zero tangent for all objects "
            "that weren't inexact arrays, but all inexact arrays always had an "
            "array-valued tangent. Now, `None` may also be passed to indicate that an "
            "inexact array has a symbolic zero tangent."
        )

        def _fn_jvp(args, t_args, **kwargs):
            t_args = jtu.tree_map(_materialise_symbolic_zero, args, t_args)
            return fn_jvp(args, t_args, **kwargs)

        self.def_jvp(_fn_jvp)

    def def_jvp(self, fn_jvp):
        def fn_jvp_wrapper(static, dynamic, tangents):
            (dynamic,) = dynamic
            (tangents,) = tangents
            d_args, _ = dynamic
            t_args, t_kwargs = tangents
            if len(jtu.tree_leaves(t_kwargs)) > 0:
                raise ValueError("Received keyword tangent")
            t_args = jtu.tree_map(_drop_nondiff, t_args, d_args)
            args, kwargs = combine(dynamic, static)
            return fn_jvp(args, t_args, **kwargs)

        self.fn.defjvp(fn_jvp_wrapper, symbolic_zeros=True)

    def defjvps(self, *a, **kw):
        raise NotImplementedError(
            "`equinox.filter_custom_jvp.defjvps` is not implemented"
        )

    def __call__(self, *args, **kwargs):
        dynamic, static = partition((args, kwargs), is_array)
        return self.fn(static, dynamic)


@ft.partial(jax.custom_jvp, nondiff_argnums=(0,))
def _nondifferentiable(msg: str, x: PyTree[Array]):
    return x


@_nondifferentiable.defjvp
def _nondifferentiable_jvp(msg: str, primals, tangents):
    raise RuntimeError(msg)


def nondifferentiable(
    x: PyTree, *, name: Optional[str] = None, msg: Optional[str] = None
) -> PyTree:
    """Identity function, which raises an error if it is differentiated (in forward or
    reverse mode).
    """
    dynamic, static = partition(x, is_array)
    if msg is None:
        if name is None:
            name = "This operation"
        msg = f"Unexpected tangent. {name} cannot be autodifferentiated."
    dynamic = _nondifferentiable(msg, dynamic)
    return combine(dynamic, static)


def _get_perturbed(x):
    assert type(x) is jax.custom_derivatives.CustomVJPPrimal
    return x.perturbed


def _get_value(x):
    assert type(x) is jax.custom_derivatives.CustomVJPPrimal
    return x.value


def _get_value_assert_unperturbed(x):
    assert type(x) is jax.custom_derivatives.CustomVJPPrimal
    assert x.perturbed is False
    return x.value


def _zero_to_none(ct):
    if isinstance(ct, jax.custom_derivatives.SymbolicZero):
        return None
    else:
        return ct


def _none_to_zero(ct, x):
    if ct is None:
        if x is None:
            return None
        else:
            aval = jax.core.get_aval(x).at_least_vspace()
            return jax.custom_derivatives.SymbolicZero(aval)
    else:
        return ct


class filter_custom_vjp:
    """As `jax.custom_vjp`, but with a nicer interface.

    Usage is:
    ```python
    @equinox.filter_custom_vjp
    def fn(vjp_arg, *args, **kwargs):
        # `vjp_arg` is some PyTree of arbitrary Python objects.
        # `args`, `kwargs` contain arbitrary Python objects.
        ...
        return out  # some PyTree of arbitrary Python objects.

    @fn.def_fwd
    def fn_fwd(perturbed, vjp_arg, *args, **kwargs):
        # `perturbed` is a pytree with the same structure as `vjp_arg`. Every leaf is
        # either `True` or `False`, indicating whether that leaf is being
        # differentiated. (All leaves that are not floating-point arrays will
        # necessarily have `False`. Some floating-point arrays might happen not to be
        # differentiated either.)
        ...
        # Should return `out` as before. `residuals` can be any collection of JAX
        # arrays you want to keep around for the backward pass.
        return out, residuals

    @fn.def_bwd
    def fn_bwd(residuals, grad_obj, perturbed, vjp_arg, *args, **kwargs):
        # `grad_obj` will have `None` as the gradient for any leaves of `out` that were
        # not differentiated.
        ...
        # `grad_vjp_arg` should be a pytree with the same structure as `vjp_arg`.
        # It can have `None` leaves to indicate that that argument has zero gradient.
        # (E.g. if the leaf was not a JAX array.)
        return grad_vjp_arg
    ```

    The key differences to `jax.custom_vjp` are that:

    - Only the gradient of the first argument, `vjp_arg`, should be computed on the
        backward pass. Everything else will automatically have zero gradient.
    - You do not need to distinguish differentiable from nondifferentiable manually.
        Instead you should return gradients for all perturbed arrays in the first
        argument. (And just put `None` on every other leaf of the PyTree.)
    - As a convenience, all of the inputs from the forward pass are additionally made
        available to you on the backward pass.
    - As a convenience, you can declare forward and backward passes using `def_fwd` and
        `def_bwd`, rather than a single `defvjp` as in core JAX.

    !!! tip

        If you need gradients with respect to multiple arguments, then just pack them
        together as a tuple via the first argument `vjp_arg`. (See also
        [`equinox.filter_grad`][] for a similar trick.)
    """

    def __init__(self, fn):
        self.fn = fn
        self.fn_fwd: Optional[Callable] = None
        self.fn_bwd: Optional[Callable] = None
        self.fn_wrapped = None

    def def_fwd(self, fn_fwd):
        self.fn_fwd = fn_fwd
        if self.fn_bwd is not None:
            self._defvjp()

    def def_bwd(self, fn_bwd):
        self.fn_bwd = fn_bwd
        if self.fn_fwd is not None:
            self._defvjp()

    def defvjp(self, fn_fwd, fn_bwd):
        warnings.warn(
            "As of Equinox 0.10.7, `equinox.filter_custom_vjp.defvjp` is deprecated in "
            "favour of `.def_fwd` and `.def_bwd`. This new API supports symbolic "
            "zeros, which allow for more efficient autodifferentiation rules. In "
            "particular:\n"
            "- the fwd and bwd functions take an extra `perturbed` argument, which "
            "    indicates which primals actually need a gradient. You can use this "
            "    to skip computing the gradient for any unperturbed value. (You can "
            "    also safely just ignore this if you wish.)\n"
            "- `None` was previously passed to indicate a symbolic zero gradient for "
            "    all objects that weren't inexact arrays, but all inexact arrays "
            "    always had an array-valued gradient. Now, `None` may also be passed "
            "    to indicate that an inexact array has a symbolic zero gradient."
        )

        def _fn_fwd(perturbed, vjp_arg, *args, **kwargs):
            del perturbed
            out, residuals = fn_fwd(vjp_arg, *args, **kwargs)
            return out, (out, residuals)

        def _fn_bwd(
            residuals, grad_diff_array_out, perturbed, vjp_arg, *args, **kwargs
        ):
            del perturbed
            out, residuals = residuals
            grad_diff_array_out = jtu.tree_map(
                _materialise_symbolic_zero, out, grad_diff_array_out
            )
            return fn_bwd(residuals, grad_diff_array_out, vjp_arg, *args, **kwargs)

        self.def_fwd(_fn_fwd)
        self.def_bwd(_fn_bwd)

    def _defvjp(self):
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
            assert self.fn_fwd is not None
            nonarray_perturbed = jtu.tree_map(lambda _: False, nonarray_vjp_arg)
            nondiff_array_perturbed = jtu.tree_map(
                lambda _: False, nondiff_array_vjp_arg
            )
            diff_array_perturbed = jtu.tree_map(_get_perturbed, diff_array_vjp_arg)
            perturbed = combine(
                nonarray_perturbed, nondiff_array_perturbed, diff_array_perturbed
            )
            diff_array_vjp_arg = jtu.tree_map(_get_value, diff_array_vjp_arg)
            nondiff_array_vjp_arg = jtu.tree_map(
                _get_value_assert_unperturbed, nondiff_array_vjp_arg
            )
            array_args_kwargs = jtu.tree_map(
                _get_value_assert_unperturbed, array_args_kwargs
            )
            vjp_arg = combine(
                nonarray_vjp_arg, diff_array_vjp_arg, nondiff_array_vjp_arg
            )
            args, kwargs = combine(nonarray_args_kwargs, array_args_kwargs)
            out, residuals = self.fn_fwd(perturbed, vjp_arg, *args, **kwargs)
            array_out, nonarray_out = partition(out, is_array)
            diff_array_out, nondiff_array_out = partition(array_out, is_inexact_array)
            out = diff_array_out, nondiff_array_out, Static(nonarray_out)
            return out, (
                residuals,
                diff_array_vjp_arg,
                nondiff_array_vjp_arg,
                array_args_kwargs,
                perturbed,
            )

        def fn_bwd_wrapped(nonarray_vjp_arg, nonarray_args_kwargs, residuals, grad_out):
            assert self.fn_bwd is not None
            (
                residuals,
                diff_array_vjp_arg,
                nondiff_array_vjp_arg,
                array_args_kwargs,
                perturbed,
            ) = residuals
            vjp_arg = combine(
                nonarray_vjp_arg, diff_array_vjp_arg, nondiff_array_vjp_arg
            )
            args, kwargs = combine(nonarray_args_kwargs, array_args_kwargs)
            grad_diff_array_out, _, _ = grad_out
            grad_diff_array_out = jtu.tree_map(_zero_to_none, grad_diff_array_out)
            out = self.fn_bwd(
                residuals, grad_diff_array_out, perturbed, vjp_arg, *args, **kwargs
            )
            if jtu.tree_structure(out, is_leaf=_is_none) != jtu.tree_structure(
                diff_array_vjp_arg, is_leaf=_is_none
            ):
                raise RuntimeError(
                    "custom_vjp gradients must have the same structure as "
                    "`equinox.filter(vjp_arg, equinox.is_inexact_array)`, where "
                    "`vjp_arg` is the first argument used in the forward pass."
                )
            out = jtu.tree_map(_none_to_zero, out, diff_array_vjp_arg, is_leaf=_is_none)
            # None is the gradient through nondiff_array_vjp_arg and array_args_kwargs
            return out, None, None

        fn_wrapped = jax.custom_vjp(fn_wrapped, nondiff_argnums=(0, 1))
        fn_wrapped.defvjp(fn_fwd_wrapped, fn_bwd_wrapped, symbolic_zeros=True)
        self.fn_wrapped = fn_wrapped

    def __call__(self, vjp_arg, /, *args, **kwargs):
        if self.fn_wrapped is None:
            raise RuntimeError(
                f"`def_fwd` or `def_bwd` not yet called for {self.fn.__name__}"
            )
        array_vjp_arg, nonarray_vjp_arg = partition(vjp_arg, is_array)
        diff_array_vjp_arg, nondiff_array_vjp_arg = partition(
            array_vjp_arg, is_inexact_array
        )
        array_args_kwargs, nonarray_args_kwargs = partition((args, kwargs), is_array)
        array_args_kwargs = nondifferentiable(
            array_args_kwargs, name="`*args` and `**kwargs` to `filter_custom_vjp`"
        )
        out = self.fn_wrapped(
            nonarray_vjp_arg,
            nonarray_args_kwargs,
            diff_array_vjp_arg,
            nondiff_array_vjp_arg,
            array_args_kwargs,
        )
        diff_array_out, nondiff_array_out, nonarray_out = out
        return combine(diff_array_out, nondiff_array_out, nonarray_out.value)


if getattr(typing, "GENERATING_DOCUMENTATION", False) and not TYPE_CHECKING:
    _filter_custom_jvp_doc = filter_custom_jvp.__doc__
    _filter_custom_vjp_doc = filter_custom_vjp.__doc__

    def def_jvp(fn_jvp):
        pass

    def defjvp(fn_jvp):
        pass

    def filter_custom_jvp(fn):
        return types.SimpleNamespace(def_jvp=def_jvp, defjvp=defjvp)

    def def_fwd(fn_fwd):
        pass

    def def_bwd(fn_bwd):
        pass

    def defvjp(fn_fwd, fn_bwd):
        pass

    def filter_custom_vjp(fn):
        return types.SimpleNamespace(def_fwd=def_fwd, def_bwd=def_bwd, defvjp=defvjp)

    filter_custom_jvp.__doc__ = _filter_custom_jvp_doc
    filter_custom_vjp.__doc__ = _filter_custom_vjp_doc
