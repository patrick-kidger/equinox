import jax.numpy as jnp
import pytest

import equinox as eqx


def test_apply_updates1():
    params = [jnp.array([5]), jnp.array([2])]
    grads = [-1, 1]
    new_params = eqx.apply_updates(params, grads)
    assert new_params == [jnp.array([4]), jnp.array([3])]


def test_apply_updates2():
    o = object()
    params = [o, jnp.array(3.0), jnp.array(2.0)]

    def f(p):
        return p[1] + p[2]

    grads = eqx.gradf(f, filter_fn=lambda x: x == 3)(params)
    new_params = eqx.apply_updates(params, grads)
    assert new_params == [o, jnp.array([4.0]), jnp.array([2.0])]


def test_apply_updates3():
    o = object()
    params = [o, jnp.array([2])]
    grads = [0, 1]
    with pytest.raises(TypeError):
        eqx.apply_updates(params, grads)
