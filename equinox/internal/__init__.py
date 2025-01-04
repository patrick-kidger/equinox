# Backward compatibility: expose via `equinox.internal`. Now available under `equinox`.
from .._better_abstract import (
    AbstractClassVar as AbstractClassVar,
    AbstractVar as AbstractVar,
)
from .._compile_utils import (
    hashable_combine as hashable_combine,
    hashable_partition as hashable_partition,
)
from .._doc_utils import doc_remove_args as doc_remove_args, doc_repr as doc_repr

# Backward compatibility: expose via `equinox.internal`. Now available under `equinox`.
from .._enum import Enumeration as Enumeration

# Backward compatibility: expose via `equinox.internal`. Now available under `equinox`.
from .._errors import (
    assert_dce as assert_dce,
    branched_error_if as branched_error_if,
    error_if as error_if,
)
from .._eval_shape import cached_filter_eval_shape as cached_filter_eval_shape
from .._misc import left_broadcast_to as left_broadcast_to
from .._module import Static as Static
from .._unvmap import (
    unvmap_all as unvmap_all,
    unvmap_all_p as unvmap_all_p,
    unvmap_any as unvmap_any,
    unvmap_any_p as unvmap_any_p,
    unvmap_max as unvmap_max,
    unvmap_max_p as unvmap_max_p,
)

# Backward compatibility: expose via `equinox.internal`. Now available under
# `equinox.debug`.
from ..debug import (
    announce_transform as announce_transform,
    backward_nan as debug_backward_nan,  # noqa: F401
    breakpoint_if as breakpoint_if,
    inspect_dce as inspect_dce,
    store_dce as store_dce,
)
from ..debug._announce_transform import announce_jaxpr_p as announce_jaxpr_p
from ._closure_to_pytree import closure_to_pytree as closure_to_pytree
from ._finalise_jaxpr import (
    finalise_eval_jaxpr as finalise_eval_jaxpr,
    finalise_fn as finalise_fn,
    finalise_jaxpr as finalise_jaxpr,
    finalise_jaxpr_as_fn as finalise_jaxpr_as_fn,
    finalise_make_jaxpr as finalise_make_jaxpr,
    primitive_finalisations as primitive_finalisations,
    register_impl_finalisation as register_impl_finalisation,
)
from ._getkey import GetKey as GetKey
from ._loop import (
    buffer_at_set as buffer_at_set,
    maybe_set_p as maybe_set_p,
    MaybeBuffer as MaybeBuffer,
    scan as scan,
    select_if_vmap_p as select_if_vmap_p,
    while_loop as while_loop,
)
from ._misc import (
    ContainerMeta as ContainerMeta,
    eval_empty as eval_empty,
    eval_zero as eval_zero,
    scan_trick as scan_trick,
)
from ._nextafter import nextafter as nextafter, prevbefore as prevbefore
from ._noinline import noinline as noinline, noinline_p as noinline_p
from ._nontraceable import (
    nonbatchable as nonbatchable,
    nonbatchable_p as nonbatchable_p,
    nondifferentiable as nondifferentiable,
    nondifferentiable_backward as nondifferentiable_backward,
    nondifferentiable_backward_p as nondifferentiable_backward_p,
    nontraceable as nontraceable,
    nontraceable_p as nontraceable_p,
)
from ._omega import ω as ω
from ._onnx import to_onnx as to_onnx
from ._primitive import (
    create_vprim as create_vprim,
    filter_primitive_batching as filter_primitive_batching,
    filter_primitive_bind as filter_primitive_bind,
    filter_primitive_def as filter_primitive_def,
    filter_primitive_jvp as filter_primitive_jvp,
    filter_primitive_transpose as filter_primitive_transpose,
    materialise_zeros as materialise_zeros,
)
from ._str2jax import str2jax as str2jax
