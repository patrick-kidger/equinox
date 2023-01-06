import functools as ft

import jax
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.tree_util as jtu

from ..filters import combine, is_array, partition


_identity = lambda x, *, name: x


def _make_error(opname):
    def _error(*args, name):
        raise RuntimeError(f"Detected {opname} of {name}")

    return _error


nontraceable_p = jax.core.Primitive("nontraceable")
nontraceable_p.def_impl(_identity)
nontraceable_p.def_abstract_eval(_identity)
ad.primitive_jvps[nontraceable_p] = _make_error("differentiation")
ad.primitive_transposes[nontraceable_p] = _make_error("transposition")
batching.primitive_batchers[nontraceable_p] = _make_error("batching")
mlir.register_lowering(
    nontraceable_p, mlir.lower_fun(_identity, multiple_results=False)
)


def nontraceable(x, *, name="nontraceable operation"):
    """Identity function, which raises an error if it is transformed in any way. (i.e.
    in `jax.grad`, `jax.vmap` etc.)

    This is useful at the end of the `impl` rule for higher-order final-style
    primitives, for checking that no other tracers were captured via closure.
    """
    dynamic, static = partition(x, is_array)
    bind = ft.partial(nontraceable_p.bind, name=name)
    dynamic = jtu.tree_map(bind, dynamic)
    return combine(dynamic, static)
