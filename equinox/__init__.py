from . import experimental, internal, nn
from .callback import filter_pure_callback
from .eval_shape import filter_eval_shape
from .filters import (
    combine,
    filter,
    is_array,
    is_array_like,
    is_inexact_array,
    is_inexact_array_like,
    partition,
)
from .grad import (
    filter_closure_convert,
    filter_custom_jvp,
    filter_custom_vjp,
    filter_grad,
    filter_jvp,
    filter_value_and_grad,
    filter_vjp,
)
from .jit import filter_jit
from .make_jaxpr import filter_make_jaxpr
from .module import Module, module_update_wrapper, static_field
from .pretty_print import tree_pformat, tree_pprint
from .serialisation import (
    default_deserialise_filter_spec,
    default_serialise_filter_spec,
    tree_deserialise_leaves,
    tree_serialise_leaves,
)
from .tree import tree_at, tree_equal, tree_inference
from .update import apply_updates
from .vmap_pmap import filter_pmap, filter_vmap, if_array


# import jaxlib
# if tuple(int(x) for x in jaxlib.__version__.split(".")) < (0, 4, 2):
#     raise ImportError("Equinox requires jaxlib>=0.4.2. You should run `pip install --upgrade jaxlib`.")


__version__ = "0.10.0"
