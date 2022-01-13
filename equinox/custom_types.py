import typing
from typing import Any

import jax
import jax.numpy as jnp


if getattr(typing, "GENERATING_DOCUMENTATION", True):
    Array = "jax.numpy.ndarray"
else:
    Array = jnp.ndarray

PyTree = Any

TreeDef = type(jax.tree_structure(0))
