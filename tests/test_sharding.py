import equinox as eqx
import jax
import jax.random as jr
import jax.tree_util as jtu
from jax.sharding import Mesh, PartitionSpec


[cpu] = jax.local_devices(backend="cpu")
sharding = jax.sharding.NamedSharding(Mesh([cpu], "x"), PartitionSpec("x"))


def test_sharding():
    mlp = eqx.nn.MLP(2, 2, 2, 2, key=jr.PRNGKey(0))

    eqx.filter_shard(mlp, cpu)

    @eqx.filter_jit
    def f(x):
        a, b = eqx.partition(x, eqx.is_array)
        a = jtu.tree_map(lambda x: x + 1, a)
        x = eqx.combine(a, b)
        return eqx.filter_shard(x, sharding)

    f(mlp)
