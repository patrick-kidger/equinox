import typing
from typing import Any

import jax
import jax.numpy as jnp


if getattr(typing, "GENERATING_DOCUMENTATION", False):
    Array = "jax.numpy.ndarray"
    PyTree = "PyTree"
else:
    Array = jnp.ndarray
    PyTree = Any

TreeDef = type(jax.tree_structure(0))
