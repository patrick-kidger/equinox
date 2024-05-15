import equinox as eqx
import jax
import jax.random as jr
import jax.tree_util as jtu


[cpu] = jax.local_devices(backend="cpu")
sharding = jax.sharding.PositionalSharding([cpu])


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
