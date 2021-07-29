import equinox as eqx
import jax.numpy as jnp


def test_apply_updates1():
    o = object()
    params = [o, jnp.array([2])]
    grads = [0, 1]
    new_params = eqx.apply_updates(params, grads)
    assert new_params == [o, jnp.array([3])]


def test_apply_updates2():
    o = object()
    params = [o, jnp.array([3.]), jnp.array([2.])]

    def f(p):
        return p[1] + p[2]

    grads = eqx.gradf(f, argnums=1)(params)
    new_params = eqx.apply_updates(params, grads)
    assert new_params == [o, jnp.array([4.]), jnp.array([2.])]
