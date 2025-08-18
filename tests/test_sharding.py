import equinox as eqx
import jax
import jax.random as jr
import jax.tree_util as jtu
from jax.sharding import Mesh, PartitionSpec

[cpu] = jax.local_devices(backend="cpu")
sharding = jax.sharding.NamedSharding(Mesh([cpu], "x"), PartitionSpec("x"))


def test_sharding():
    is_committed = lambda pytree: jax.tree.reduce(lambda x, y: x and y, jax.tree.map(lambda x: x.committed, pytree))
    is_sharded = lambda pytree, sharding: jax.tree.reduce(lambda x, y: x and y, jax.tree.map(lambda x: x.sharding == sharding, pytree))

    mlp = eqx.nn.MLP(2, 2, 2, 2, key=jr.PRNGKey(0))

    cpu_sharding = jax.sharding.SingleDeviceSharding(cpu)
    sharded_mlp = eqx.filter_shard(mlp, cpu_sharding)
    assert is_committed(eqx.filter(sharded_mlp, eqx.is_array))
    assert is_sharded(eqx.filter(sharded_mlp, eqx.is_array), cpu_sharding)

    @eqx.filter_jit
    def f(x):
        a, b = eqx.partition(x, eqx.is_array)
        a = jtu.tree_map(lambda x: x + 1, a)
        x = eqx.combine(a, b)
        return x
        # return eqx.filter_shard(x, sharding)

    _ = f(mlp)

    # assert is_sharded(eqx.filter(out, eqx.is_array), sharding)