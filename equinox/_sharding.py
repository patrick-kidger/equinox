from typing import Any, Union

import jax
import jax.lax as lax
from jaxlib.xla_extension import Device
from jaxtyping import PyTree

from ._filters import combine, is_array, partition


def filter_shard(
    x: PyTree[Any], device_or_shardings: Union[Device, jax.sharding.Sharding]
):
    """Filtered transform combining `jax.lax.with_sharding_constraint`
    and `jax.device_put`.

    Enforces sharding within a JIT'd computation (That is, how an array is
    split between multiple devices, i.e. multiple GPUs/TPUs.), or moves `x` to
    a device.

    **Arguments:**

    - `x`: A PyTree, with potentially a mix of arrays and non-arrays on the leaves.
        They will have their shardings constrained.
    - `device_or_shardings`: Either a singular device (e.g. CPU or GPU) or PyTree of
        sharding specifications. The structure should be a prefix of `x`.

    **Returns:**

    A copy of `x` with the specified sharding constraints.

    !!! Example
        See also the [autoparallelism example](../../examples/parallelism).
    """
    if isinstance(device_or_shardings, Device):
        shardings = jax.sharding.SingleDeviceSharding(device_or_shardings)
    else:
        shardings = device_or_shardings
    dynamic, static = partition(x, is_array)
    dynamic = lax.with_sharding_constraint(dynamic, shardings)
    return combine(dynamic, static)
