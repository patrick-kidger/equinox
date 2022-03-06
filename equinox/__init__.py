from . import nn
from .filters import (
    combine,
    filter,
    is_array,
    is_array_like,
    is_inexact_array,
    is_inexact_array_like,
    merge,
    partition,
    split,
)
from .grad import (
    filter_custom_vjp,
    filter_grad,
    filter_value_and_grad,
    gradf,
    value_and_grad_f,
)
from .jit import filter_jit, jitf
from .module import Module, static_field
from .tree import tree_at, tree_equal
from .update import apply_updates


__version__ = "0.2.0"
