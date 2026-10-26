import functools as ft

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P


pytestmark = pytest.mark.skipif(
    not hasattr(jax, "shard_map") or jax.default_backend() != "cpu",
    reason="Requires the public shard_map API and simulated CPU devices.",
)


def test_closure_convert_constant_after_sharding():
    mesh = jax.make_mesh((2,), ("i",))

    @ft.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=P("i"),
        out_specs=P("i"),
        check_vma=False,
    )
    def run(x):
        constant = jnp.array(1.0)
        converted = eqx.filter_closure_convert(lambda y: x * y, constant)
        # Adding zero attaches the manual mesh without changing the value.
        return converted(constant + 0)

    x = jax.device_put(jnp.array([2.0, 3.0]), NamedSharding(mesh, P("i")))
    np.testing.assert_array_equal(jax.jit(run)(x), [2.0, 3.0])
