from typing import Any

import jax
import jax.core
import jax.lax as lax
from jaxtyping import PyTree

from ._compile_utils import hashable_partition, hashable_combine
from ._filters import is_array


def filter_shard(
    x: PyTree[Any],
    device_or_shardings: jax.Device | jax.sharding.Sharding,  # pyright: ignore[reportInvalidTypeForm]
):
    """Filtered transform combining `jax.lax.with_sharding_constraint`
    and `jax.device_put`.

    Enforces sharding within a JIT'd computation (That is, how an array is
    split between multiple devices, i.e. multiple GPUs/TPUs.), or outside a
    JIT'd region moves `x` to a device.

    **Arguments:**

    - `x`: A PyTree, with potentially a mix of arrays and non-arrays on the leaves.
        They will have their shardings constrained.
    - `device_or_shardings`: Either a singular device (e.g. CPU or GPU) or PyTree of
        sharding specifications. The structure should be a prefix of `x`.

    **Returns:**

    A copy of `x` with the specified sharding constraints.

    !!! Example
        See also the [autoparallelism example](../examples/parallelism.ipynb).
    """
    if isinstance(device_or_shardings, jax.Device):
        shardings = jax.sharding.SingleDeviceSharding(device_or_shardings)
    else:
        shardings = device_or_shardings
    dynamic, static = hashable_partition(x, is_array)
    if len(dynamic) > 0:
        test_array = dynamic[0]
        if isinstance(test_array, jax.core.Tracer):
            dynamic = lax.with_sharding_constraint(dynamic, shardings)
        else:
            dynamic = jax.device_put(dynamic, shardings)
    return hashable_combine(dynamic, static)
