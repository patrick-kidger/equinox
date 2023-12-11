from typing import Any

import jax
import jax.lax as lax
from jaxtyping import PyTree

from ._filters import combine, is_array, partition


def filter_with_sharding_constraint(x: PyTree[Any], shardings):
    """Filtered version of `jax.lax.with_sharding_constraint`. Enforces sharding within
    a JIT'd computation. (That is, how an array is split between multiple devices, i.e.
    multiple GPUs/TPUs.)

    This should always be called *inside* of a JIT'd computation.

    This is a strict constraint for the XLA compiler, and not just a hint. It is
    typically placed on the inputs of JIT'd computations to assert that they are sharded
    in the correct way, and on the output of JIT'd computations to specify how they
    should be sharded.

    **Arguments:**

    - `x`: A PyTree, with potentially a mix of arrays and non-arrays on the leaves. They
        will have their shardings constrained.
    - `shardings`: a PyTree of sharding specifications. The structure should be a prefix
        of `x`.

    **Returns:**

    A copy of `x` with the specified sharding constraints.

    !!! Example

        See also the [autoparallelism example](../../../examples/parallelism).
    """
    dynamic, static = partition(x, is_array)
    dynamic = lax.with_sharding_constraint(dynamic, shardings)
    return combine(dynamic, static)


def filter_device_put(x: PyTree[Any], device):
    """Filtered version of `jax.device_put`. Places all arrays in `x` on the device.
    Non-arrays are unchanged.

    This should always be called *outside* of a JIT'd computation.

    **Arguments:**

    - `x`: A PyTree, with potentially a mix of arrays and non-arrays on the leaves.
    - `device`: A specification for how to place `x` on a device. Most typically this is
        either a `Device` (as returned by `jax.local_devices`) or a sharding (usually a
        `jax.sharding.NamedSharding` or `jax.sharding.PositionalSharding`).

    **Returns:**

    A copy of `x` that resides on `device`.

    !!! Example

        See also the [autoparallelism example](../../../examples/parallelism).
    """
    dynamic, static = partition(x, is_array)
    dynamic = jax.device_put(dynamic, device)
    return combine(dynamic, static)
