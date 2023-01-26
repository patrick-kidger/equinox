from ..better_abc import abstractattribute
from ..compile_utils import hashable_combine, hashable_partition
from ..doc_utils import doc_repr
from ..module import Static
from ..pretty_print import tree_pp
from ..vmap_pmap import if_mapped
from .checkpointed_while_loop import Buffer, checkpointed_while_loop
from .debug import announce_jaxpr_p, announce_transform, debug_backward_nan
from .errors import branched_error_if, branched_error_p, error_if
from .finalise_jaxpr import (
    finalise_eval_jaxpr,
    finalise_fn,
    finalise_jaxpr,
    finalise_jaxpr_as_fn,
    finalise_make_jaxpr,
    primitive_finalisations,
    register_impl_finalisation,
)
from .misc import ContainerMeta, left_broadcast_to
from .nextafter import nextafter, prevbefore
from .noinline import noinline, noinline_p
from .nontraceable import (
    nonbatchable,
    nonbatchable_p,
    nondifferentiable,
    nondifferentiable_backward,
    nondifferentiable_backward_p,
    nontraceable,
    nontraceable_p,
)
from .omega import ω
from .onnx import to_onnx
from .primitive import (
    create_vprim,
    filter_primitive_batching,
    filter_primitive_bind,
    filter_primitive_def,
    filter_primitive_jvp,
    filter_primitive_transpose,
    materialise_zeros,
)
from .str2jax import str2jax
from .unvmap import (
    unvmap_all,
    unvmap_all_p,
    unvmap_any,
    unvmap_any_p,
    unvmap_max,
    unvmap_max_p,
)
