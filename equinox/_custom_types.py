from typing import Any, TYPE_CHECKING
from typing_extensions import TypeAlias

import jax.random as jr
from jaxtyping import Array

from ._doc_utils import doc_repr


sentinel: Any = doc_repr(object(), "sentinel")
if TYPE_CHECKING:
    PRNGKey: TypeAlias = jr.KeyArray
else:
    PRNGKey = doc_repr(Array, "jax.random.PRNGKey")
