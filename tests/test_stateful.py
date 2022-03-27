import functools as ft

import jax
import jax.numpy as jnp
import pytest

import equinox as eqx


def test_basic():
    a = [jnp.array(3), jnp.array(2)]
    b = [jnp.array(4), jnp.array(5)]
    index = eqx.StateIndex()
    eqx.set_state(index, a)
    assert eqx.get_state(index, b) == a


def test_jit():
    index = eqx.StateIndex()

    @eqx.filter_jit
    def set_state(x):
        eqx.set_state(index, x)

    @eqx.filter_jit
    def get_state(x):
        return eqx.get_state(index, x)

    a = [jnp.array(3), jnp.array(2)]
    b = [jnp.array(4), jnp.array(5)]
    set_state(a)
    assert get_state(b) == a


def test_no_nonjaxarray():
    c = 0
    index = eqx.StateIndex()
    with pytest.raises(ValueError):
        eqx.set_state(index, c)

    d = object()
    index = eqx.StateIndex()
    with pytest.raises(ValueError):
        eqx.set_state(index, d)

    e = [jnp.array(2), 0]
    index = eqx.StateIndex()
    with pytest.raises(ValueError):
        eqx.set_state(index, e)


def test_no_set():
    index = eqx.StateIndex()
    a = jnp.array(2)
    with pytest.raises(KeyError):
        eqx.get_state(index, a)


def test_no_change_shape():
    index1 = eqx.StateIndex()
    index2 = eqx.StateIndex()

    @jax.jit
    def set_state1():
        eqx.set_state(index1, jnp.array(1))
        eqx.set_state(index1, jnp.array([2]))

    @jax.jit
    def set_state2():
        eqx.set_state(index2, jnp.array(1))
        eqx.set_state(index2, [jnp.array(1)])

    with pytest.raises(ValueError):
        set_state1()
    with pytest.raises(ValueError):
        set_state2()


def test_index_jittable():
    index1 = eqx.StateIndex()
    index2 = eqx.StateIndex()

    @eqx.filter_jit
    def get_state(i, x):
        return eqx.get_state(i, x)

    a = [jnp.array(3), jnp.array(2)]
    b = [jnp.array(4), jnp.array(5)]
    c = [jnp.array(6), jnp.array(9)]
    d = [jnp.array(7), jnp.array(8)]
    eqx.set_state(index1, a)
    eqx.set_state(index2, b)
    assert get_state(index1, c) == a
    assert get_state(index2, d) == b


def test_vmap():
    index = eqx.StateIndex()

    @jax.vmap
    def set_state(x):
        eqx.set_state(index, x)

    @jax.vmap
    def get_state(y):
        return eqx.get_state(index, y)

    a = jnp.array([1, 2])
    b = jnp.array([3, 4])
    set_state(a)
    assert jnp.array_equal(get_state(b), a)

    with pytest.raises(ValueError):
        eqx.get_state(index, b)


def test_multi_vmap():
    index = eqx.StateIndex()

    @jax.vmap
    @jax.vmap
    def set_state(x):
        eqx.set_state(index, x)

    @jax.vmap
    @jax.vmap
    def get_state(y):
        return eqx.get_state(index, y)

    @ft.partial(jax.vmap, in_axes=(1,))
    @ft.partial(jax.vmap, in_axes=(0,))
    def get_state_bad(y):
        return eqx.get_state(index, y)

    a = jnp.array([[1, 2]])
    b = jnp.array([[3, 4]])
    set_state(a)
    assert jnp.array_equal(get_state(b), a)

    with pytest.raises(ValueError):
        eqx.get_state(index, b)

    with pytest.raises(ValueError):
        get_state_bad(b)
