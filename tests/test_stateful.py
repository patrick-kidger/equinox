import jax
import jax.numpy as jnp
import pytest

import equinox as eqx


def test_basic():
    a = (jnp.array(3), 0, object())
    index = eqx.StateIndex()
    eqx.set_state(index, a)
    assert eqx.get_state(index) == a


def test_jit():
    index = eqx.StateIndex()

    @eqx.filter_jit
    def set_state(x):
        eqx.set_state(index, x)

    @eqx.filter_jit
    def get_state():
        return eqx.get_state(index)

    a = (jnp.array(3), 0, object())
    set_state(a)
    assert get_state() == a


def test_no_set():
    index = eqx.StateIndex()
    with pytest.raises(RuntimeError):
        eqx.get_state(index)


def test_no_change_size():
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
    def get_state(i):
        return eqx.get_state(i)

    a = jnp.array(1)
    b = object()
    eqx.set_state(index1, a)
    eqx.set_state(index2, b)
    assert get_state(index1) == a
    assert get_state(index2) == b
