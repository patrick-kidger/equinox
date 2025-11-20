import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.sharding as jshard
import jax.tree_util as jtu
import pytest


# Info on the skips: on CPU-only, JAX can simulate mulitple devices
# but as soon as you have a GPU(s), JAX will use those, so if you have
# only a single GPU (i.e. a single device) it makes no sense to "test"
# sharding


@pytest.mark.skipif(
    condition=len(jax.devices()) < 2,
    reason="Test requires > 1 device to verify implicit sharding propagation",
)
def test_sharding_no_inside_jit():
    mlp = eqx.nn.MLP(2, 2, 2, 2, key=jr.PRNGKey(0))

    num_devices = 2
    mesh = jax.make_mesh(
        (num_devices,), ("x",), axis_types=(jax.sharding.AxisType.Auto,)
    )
    sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("x"))
    sharded_mlp = eqx.filter_shard(mlp, sharding)
    assert _is_committed(eqx.filter(sharded_mlp, eqx.is_array))
    assert _is_sharded(eqx.filter(sharded_mlp, eqx.is_array), sharding)

    @eqx.filter_jit
    def f(x):
        a, b = eqx.partition(x, eqx.is_array)
        a = jtu.tree_map(lambda x: x + 1, a)
        x = eqx.combine(a, b)
        return x

    out = f(sharded_mlp)

    assert _is_sharded(eqx.filter(out, eqx.is_array), sharding)


@pytest.mark.skipif(
    condition=len(jax.devices()) < 2,
    reason="Test requires > 1 device to verify explicit sharding propagation",
)
def test_sharding_only_inside_jit():
    # Make sharding
    num_devices = 2
    mesh = jax.make_mesh(
        (num_devices,), ("x",), axis_types=(jax.sharding.AxisType.Auto,)
    )
    sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("x"))
    # Make dummy pytree
    shape = (10 * num_devices,)
    pytree = ((jnp.zeros(shape), 4), (jnp.zeros(shape),), "something_static")
    assert not _is_sharded(eqx.filter(pytree, eqx.is_array), sharding)

    @eqx.filter_jit
    def f(x):
        a, b = eqx.partition(x, eqx.is_array)
        a = jtu.tree_map(lambda x: x + 1, a)
        x = eqx.combine(a, b)
        return eqx.filter_shard(x, sharding)

    # Evaluate jitted computation
    out = f(pytree)

    assert _is_sharded(eqx.filter(out, eqx.is_array), sharding)


def _is_sharded(pytree, sharding):
    return jax.tree.reduce(
        lambda x, y: x and y, jax.tree.map(lambda x: x.sharding == sharding, pytree)
    )


def _is_committed(pytree):
    return jax.tree.reduce(
        lambda x, y: x and y, jax.tree.map(lambda x: x.committed, pytree)
    )
