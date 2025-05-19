import importlib.metadata

from . import debug as debug, internal as internal, nn as nn
from ._ad import (
    filter_checkpoint as filter_checkpoint,
    filter_closure_convert as filter_closure_convert,
    filter_custom_jvp as filter_custom_jvp,
    filter_custom_vjp as filter_custom_vjp,
    filter_grad as filter_grad,
    filter_hessian as filter_hessian,
    filter_jacfwd as filter_jacfwd,
    filter_jacrev as filter_jacrev,
    filter_jvp as filter_jvp,
    filter_value_and_grad as filter_value_and_grad,
    filter_vjp as filter_vjp,
)
from ._better_abstract import (
    AbstractClassVar as AbstractClassVar,
    AbstractVar as AbstractVar,
)
from ._caches import clear_caches as clear_caches
from ._callback import filter_pure_callback as filter_pure_callback
from ._enum import Enumeration as Enumeration
from ._errors import (
    branched_error_if as branched_error_if,
    EquinoxTracetimeError as EquinoxTracetimeError,
    error_if as error_if,
)
from ._eval_shape import filter_eval_shape as filter_eval_shape
from ._filters import (
    combine as combine,
    filter as filter,
    is_array as is_array,
    is_array_like as is_array_like,
    is_inexact_array as is_inexact_array,
    is_inexact_array_like as is_inexact_array_like,
    partition as partition,
)
from ._jit import EquinoxRuntimeError as EquinoxRuntimeError, filter_jit as filter_jit
from ._make_jaxpr import filter_make_jaxpr as filter_make_jaxpr
from ._module import (
    field as field,
    Module as Module,
    module_update_wrapper as module_update_wrapper,
    Partial as Partial,
    static_field as static_field,
    StrictConfig as StrictConfig,
)
from ._pretty_print import tree_pformat as tree_pformat, tree_pprint as tree_pprint
from ._serialisation import (
    default_deserialise_filter_spec as default_deserialise_filter_spec,
    default_serialise_filter_spec as default_serialise_filter_spec,
    tree_deserialise_leaves as tree_deserialise_leaves,
    tree_serialise_leaves as tree_serialise_leaves,
)
from ._sharding import filter_shard as filter_shard
from ._tree import (
    tree_at as tree_at,
    tree_check as tree_check,
    tree_equal as tree_equal,
    tree_flatten_one_level as tree_flatten_one_level,
)
from ._update import apply_updates as apply_updates
from ._vmap_pmap import (
    filter_pmap as filter_pmap,
    filter_vmap as filter_vmap,
    if_array as if_array,
)
from .nn import inference_mode as tree_inference  # noqa: F401 - backward compatibility


__version__ = importlib.metadata.version("equinox")
