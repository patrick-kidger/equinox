import jax.numpy as jnp

import equinox as eqx


def test_clear_caches():
    @eqx.filter_jit
    def f(x):
        return x + 1

    f(jnp.array(1.0))
    eqx.clear_caches()
