import equinox as eqx
import jax
import jax.random as jr


[cpu] = jax.local_devices(backend="cpu")
sharding = jax.sharding.PositionalSharding([cpu])


def test_sharding():
    mlp = eqx.nn.MLP(2, 2, 2, 2, key=jr.PRNGKey(0))
    eqx.filter_device_put(mlp, cpu)

    @eqx.filter_jit
    def f(x):
        return eqx.filter_with_sharding_constraint(x, sharding)

    f(mlp)
