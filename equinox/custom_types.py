from typing import Any, TYPE_CHECKING

import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array

from .doc_utils import doc_repr


sentinel: Any = doc_repr(object(), "sentinel")
TreeDef = type(jtu.tree_structure(0))
if TYPE_CHECKING:
    PRNGKey = jr.KeyArray
else:
    PRNGKey = doc_repr(Array, "jax.random.PRNGKey")
