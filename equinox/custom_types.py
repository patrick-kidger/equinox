from typing import Any

import jax
import jax.numpy as jnp


Array = jnp.ndarray

PyTree = Any

TreeDef = type(jax.tree_structure(0))
