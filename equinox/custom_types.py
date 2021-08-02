from typing import Any, Union

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np


# JAX arrays
Array = Union[jax.core.Tracer, jaxlib.xla_extension.DeviceArray]
# JAX arrays + numpy arrays
# jnp.ndarray counts as a numpy array: `assert isinstance(np.array(1.), jnp.ndarray)`
MoreArrays = Union[Array, np.ndarray, jnp.ndarray]

PyTree = Any

TreeDef = jaxlib.xla_extension.PyTreeDef
