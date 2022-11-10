from ..compile_utils import hashable_combine, hashable_partition
from ..doc_utils import doc_repr, doc_strip_annotations
from ..module import Static
from .ad import nondifferentiable, nondifferentiable_backward
from .debug import announce_transform, debug_backward_nan
from .errors import branched_error_if, error_if
from .misc import ContainerMeta, left_broadcast_to
from .nextafter import nextafter, prevbefore
from .noinline import noinline
from .omega import ω
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
from .unvmap import unvmap_all, unvmap_any, unvmap_max
