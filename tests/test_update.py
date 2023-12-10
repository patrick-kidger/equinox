import equinox as eqx
import jax.numpy as jnp
import pytest


def test_apply_updates1():
    params = [object(), jnp.array([5]), jnp.array([2])]
    grads = [None, -1, 1]
    new_params = eqx.apply_updates(params, grads)
    assert new_params == [params[0], jnp.array([4]), jnp.array([3])]


def test_apply_updates2():
    o = object()
    params = [o, jnp.array([2])]
    grads = [0, 1]
    with pytest.raises(TypeError):
        eqx.apply_updates(params, grads)
