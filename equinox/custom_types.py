import jax
import jaxlib
import numpy as np
from typing import Any, Union


Array = Union[jax.core.Tracer, jaxlib.xla_extension.DeviceArray, np.ndarray]

PyTree = Any

TreeDef = Any
