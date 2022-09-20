from typing import Any, Callable, Union

import jax.tree_util as jtu

from .doc_utils import doc_repr


sentinel = doc_repr(object(), "sentinel")
TreeDef = type(jtu.tree_structure(0))
ResolvedBoolAxisSpec = bool
BoolAxisSpec = Union[ResolvedBoolAxisSpec, Callable[[Any], ResolvedBoolAxisSpec]]
