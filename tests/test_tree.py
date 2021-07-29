import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom


def test_tree_at():
    [1, 2, {"a": jnp.array([1., 2.])}, eqx.nn.Linear(1, 2, key=jrandom.PRNGKey(0))]
