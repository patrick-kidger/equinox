import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.sharding as jshard
import jax.tree_util as jtu

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_num_cpu_devices", 2)



def test_sharding_no_inside_jit():
    is_committed = lambda pytree: jax.tree.reduce(lambda x, y: x and y, jax.tree.map(lambda x: x.committed, pytree))
    is_sharded = lambda pytree, sharding: jax.tree.reduce(lambda x, y: x and y, jax.tree.map(lambda x: eqx.tree_equal(x.sharding, sharding), pytree))

    mlp = eqx.nn.MLP(2, 2, 2, 2, key=jr.PRNGKey(0))

    num_devices = 2
    mesh = jax.make_mesh((num_devices,), ("x",))
    sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("x"))
    sharded_mlp = eqx.filter_shard(mlp, sharding)
    assert is_committed(eqx.filter(sharded_mlp, eqx.is_array))
    assert is_sharded(eqx.filter(sharded_mlp, eqx.is_array), sharding)

    @eqx.filter_jit
    def f(x):
        a, b = eqx.partition(x, eqx.is_array)
        a = jtu.tree_map(lambda x: x + 1, a)
        x = eqx.combine(a, b)
        return eqx.filter_shard(a, sharding)

    out = f(sharded_mlp)

    assert is_sharded(eqx.filter(out, eqx.is_array), sharding)

def test_sharding_only_inside_jit():
    is_sharded = lambda pytree, sharding: jax.tree.reduce(lambda x, y: x and y, jax.tree.map(lambda x: x.sharding == sharding, pytree))

    # Make sharding
    devices = jax.local_devices(backend="cpu")
    num_devices = len(devices)
    mesh = jax.make_mesh((num_devices,), ("x",))
    sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("x"))
    # Make dummy pytree
    shape = (num_devices,)
    pytree = ((jnp.zeros(shape), 4), (jnp.zeros(shape),), "something_static")
    assert not is_sharded(eqx.filter(pytree, eqx.is_array), sharding)

    @eqx.filter_jit
    def f(x):
        a, b = eqx.partition(x, eqx.is_array)
        a = jtu.tree_map(lambda x: x + 1, a)
        x = eqx.combine(a, b)
        return eqx.filter_shard(x, sharding)

    # Evaluate jitted computation
    out = f(pytree)

    assert is_sharded(eqx.filter(out, eqx.is_array), sharding)