import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.sharding as jshard
import jax.tree_util as jtu


def test_sharding_no_inside_jit():
    mlp = eqx.nn.MLP(2, 2, 2, 2, key=jr.PRNGKey(0))

    num_devices = 1
    mesh = jax.make_mesh((num_devices,), ("x",))
    sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("x"))
    sharded_mlp = eqx.filter_shard(mlp, sharding)
    # Known bug, not resolved until JAX 0.7.2. Eager behavior of `eqx.filter_shard`
    # does not commit arrays to the sharding if does not change the original
    # array (i.e. it does not result in a copy): https://github.com/jax-ml/jax/issues/31199
    # assert is_committed(eqx.filter(sharded_mlp, eqx.is_array))
    # assert is_sharded(eqx.filter(sharded_mlp, eqx.is_array), sharding)

    @eqx.filter_jit
    def f(x):
        a, b = eqx.partition(x, eqx.is_array)
        a = jtu.tree_map(lambda x: x + 1, a)
        x = eqx.combine(a, b)
        return x

    _ = f(sharded_mlp)

    # assert is_sharded(eqx.filter(out, eqx.is_array), sharding)


def test_sharding_only_inside_jit():
    # Make sharding
    num_devices = 1
    mesh = jax.make_mesh((num_devices,), ("x",))
    sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("x"))
    # Make dummy pytree
    shape = (10 * num_devices,)
    pytree = ((jnp.zeros(shape), 4), (jnp.zeros(shape),), "something_static")
    # assert not _is_sharded(eqx.filter(pytree, eqx.is_array), sharding)

    @eqx.filter_jit
    def f(x):
        a, b = eqx.partition(x, eqx.is_array)
        a = jtu.tree_map(lambda x: x + 1, a)
        x = eqx.combine(a, b)
        return eqx.filter_shard(x, sharding)

    # Evaluate jitted computation
    _ = f(pytree)

    # assert is_sharded(eqx.filter(out, eqx.is_array), sharding)


def _is_sharded(pytree, sharding):
    return jax.tree.reduce(
        lambda x, y: x and y, jax.tree.map(lambda x: x.sharding == sharding, pytree)
    )


def _is_committed(pytree):
    return jax.tree.reduce(
        lambda x, y: x and y, jax.tree.map(lambda x: x.committed, pytree)
    )
