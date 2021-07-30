import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from typing import Any, Union


Array = Union[jax.core.Tracer, jaxlib.xla_extension.DeviceArray, jnp.ndarray, np.ndarray]

PyTree = Any

TreeDef = jaxlib.xla_extension.PyTreeDef
