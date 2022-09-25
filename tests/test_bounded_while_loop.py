import jax
import jax.numpy as jnp

import equinox.internal as eqxi

from .helpers import shaped_allclose


def test_functional_no_vmap_no_inplace():
    def cond_fun(val):
        x, step = val
        return step < 5

    def body_fun(val, _):
        x, step = val
        return (x + 0.1, step + 1)

    init_val = (jnp.array([0.3]), 0)

    val = eqxi.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=0)
    assert shaped_allclose(val[0], jnp.array([0.3])) and val[1] == 0

    val = eqxi.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=1)
    assert shaped_allclose(val[0], jnp.array([0.4])) and val[1] == 1

    val = eqxi.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=2)
    assert shaped_allclose(val[0], jnp.array([0.5])) and val[1] == 2

    val = eqxi.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=4)
    assert shaped_allclose(val[0], jnp.array([0.7])) and val[1] == 4

    val = eqxi.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=8)
    assert shaped_allclose(val[0], jnp.array([0.8])) and val[1] == 5

    val = eqxi.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=None)
    assert shaped_allclose(val[0], jnp.array([0.8])) and val[1] == 5


def test_functional_no_vmap_inplace():
    def cond_fun(val):
        x, step = val
        return step < 5

    def body_fun(val, inplace):
        x, step = val
        x = inplace(x).at[jnp.minimum(step + 1, 4)].set(x[step] + 0.1)
        step = inplace(step).at[()].set(step + 1)
        x = eqxi.HadInplaceUpdate(x)
        step = eqxi.HadInplaceUpdate(step)
        return x, step

    init_val = (jnp.array([0.3, 0.3, 0.3, 0.3, 0.3]), 0)

    val = eqxi.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=0)
    assert shaped_allclose(val[0], jnp.array([0.3, 0.3, 0.3, 0.3, 0.3])) and val[1] == 0

    val = eqxi.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=1)
    assert shaped_allclose(val[0], jnp.array([0.3, 0.4, 0.3, 0.3, 0.3])) and val[1] == 1

    val = eqxi.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=2)
    assert shaped_allclose(val[0], jnp.array([0.3, 0.4, 0.5, 0.3, 0.3])) and val[1] == 2

    val = eqxi.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=4)
    assert shaped_allclose(val[0], jnp.array([0.3, 0.4, 0.5, 0.6, 0.7])) and val[1] == 4

    val = eqxi.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=8)
    assert shaped_allclose(val[0], jnp.array([0.3, 0.4, 0.5, 0.6, 0.8])) and val[1] == 5

    val = eqxi.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=None)
    assert shaped_allclose(val[0], jnp.array([0.3, 0.4, 0.5, 0.6, 0.8])) and val[1] == 5


def test_functional_vmap_no_inplace():
    def cond_fun(val):
        x, step = val
        return step < 5

    def body_fun(val, _):
        x, step = val
        return (x + 0.1, step + 1)

    init_val = (jnp.array([[0.3], [0.4]]), jnp.array([0, 3]))

    val = jax.vmap(
        lambda v: eqxi.bounded_while_loop(cond_fun, body_fun, v, max_steps=0)
    )(init_val)
    assert shaped_allclose(val[0], jnp.array([[0.3], [0.4]])) and jnp.array_equal(
        val[1], jnp.array([0, 3])
    )

    val = jax.vmap(
        lambda v: eqxi.bounded_while_loop(cond_fun, body_fun, v, max_steps=1)
    )(init_val)
    assert shaped_allclose(val[0], jnp.array([[0.4], [0.5]])) and jnp.array_equal(
        val[1], jnp.array([1, 4])
    )

    val = jax.vmap(
        lambda v: eqxi.bounded_while_loop(cond_fun, body_fun, v, max_steps=2)
    )(init_val)
    assert shaped_allclose(val[0], jnp.array([[0.5], [0.6]])) and jnp.array_equal(
        val[1], jnp.array([2, 5])
    )

    val = jax.vmap(
        lambda v: eqxi.bounded_while_loop(cond_fun, body_fun, v, max_steps=4)
    )(init_val)
    assert shaped_allclose(val[0], jnp.array([[0.7], [0.6]])) and jnp.array_equal(
        val[1], jnp.array([4, 5])
    )

    val = jax.vmap(
        lambda v: eqxi.bounded_while_loop(cond_fun, body_fun, v, max_steps=8)
    )(init_val)
    assert shaped_allclose(val[0], jnp.array([[0.8], [0.6]])) and jnp.array_equal(
        val[1], jnp.array([5, 5])
    )

    val = jax.vmap(
        lambda v: eqxi.bounded_while_loop(cond_fun, body_fun, v, max_steps=None)
    )(init_val)
    assert shaped_allclose(val[0], jnp.array([[0.8], [0.6]])) and jnp.array_equal(
        val[1], jnp.array([5, 5])
    )


def test_functional_vmap_inplace():
    def cond_fun(val):
        x, step, max_step = val
        return step < max_step

    def body_fun(val, inplace):
        x, step, max_step = val
        x = inplace(x).at[jnp.minimum(step + 1, 4)].set(x[step] + 0.1)
        step = inplace(step).at[()].set(step + 1)
        x = eqxi.HadInplaceUpdate(x)
        step = eqxi.HadInplaceUpdate(step)
        return x, step, max_step

    init_val = (
        jnp.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4]]),
        jnp.array([0, 1]),
        jnp.array([5, 3]),
    )

    val = jax.vmap(
        lambda v: eqxi.bounded_while_loop(cond_fun, body_fun, v, max_steps=0)
    )(init_val)
    assert shaped_allclose(
        val[0], jnp.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([0, 1]))

    val = jax.vmap(
        lambda v: eqxi.bounded_while_loop(cond_fun, body_fun, v, max_steps=1)
    )(init_val)
    assert shaped_allclose(
        val[0], jnp.array([[0.3, 0.4, 0.3, 0.3, 0.3], [0.4, 0.4, 0.5, 0.4, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([1, 2]))

    val = jax.vmap(
        lambda v: eqxi.bounded_while_loop(cond_fun, body_fun, v, max_steps=2)
    )(init_val)
    assert shaped_allclose(
        val[0], jnp.array([[0.3, 0.4, 0.5, 0.3, 0.3], [0.4, 0.4, 0.5, 0.6, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([2, 3]))

    val = jax.vmap(
        lambda v: eqxi.bounded_while_loop(cond_fun, body_fun, v, max_steps=4)
    )(init_val)
    assert shaped_allclose(
        val[0], jnp.array([[0.3, 0.4, 0.5, 0.6, 0.7], [0.4, 0.4, 0.5, 0.6, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([4, 3]))

    val = jax.vmap(
        lambda v: eqxi.bounded_while_loop(cond_fun, body_fun, v, max_steps=8)
    )(init_val)
    assert shaped_allclose(
        val[0], jnp.array([[0.3, 0.4, 0.5, 0.6, 0.8], [0.4, 0.4, 0.5, 0.6, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([5, 3]))

    val = jax.vmap(
        lambda v: eqxi.bounded_while_loop(cond_fun, body_fun, v, max_steps=None)
    )(init_val)
    assert shaped_allclose(
        val[0], jnp.array([[0.3, 0.4, 0.5, 0.6, 0.8], [0.4, 0.4, 0.5, 0.6, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([5, 3]))
