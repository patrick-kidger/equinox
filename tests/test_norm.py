import equinox as eqx
import jax
import jax.flatten_util as jfu
import jax.numpy as jnp

from .helpers import tree_allclose


def _square(x):
    return x * jnp.conj(x)


def _two_norm(x):
    return jnp.sqrt(jnp.sum(_square(jfu.ravel_pytree(x)[0]))).real


def _rms_norm(x):
    return jnp.sqrt(jnp.mean(_square(jfu.ravel_pytree(x)[0]))).real


def _max_norm(x):
    return jnp.max(jnp.abs(jfu.ravel_pytree(x)[0]))


def test_nonzero():
    zero = [jnp.array(0.0), jnp.zeros((2, 2))]
    x = [jnp.array(1.0), jnp.arange(4.0).reshape(2, 2)]
    tx = [jnp.array(0.5), jnp.arange(1.0, 5.0).reshape(2, 2)]

    two = eqx.two_norm(x)
    rms = eqx.rms_norm(x)
    max = eqx.max_norm(x)
    true_two = _two_norm(x)
    true_rms = _rms_norm(x)
    true_max = _max_norm(x)
    assert jnp.allclose(two, true_two)
    assert jnp.allclose(rms, true_rms)
    assert jnp.allclose(max, true_max)

    two_jvp = jax.jvp(eqx.two_norm, (x,), (tx,))
    true_two_jvp = jax.jvp(_two_norm, (x,), (tx,))
    rms_jvp = jax.jvp(eqx.rms_norm, (x,), (tx,))
    true_rms_jvp = jax.jvp(_rms_norm, (x,), (tx,))
    max_jvp = jax.jvp(eqx.max_norm, (x,), (tx,))
    true_max_jvp = jax.jvp(_max_norm, (x,), (tx,))
    assert tree_allclose(two_jvp, true_two_jvp)
    assert tree_allclose(rms_jvp, true_rms_jvp)
    assert tree_allclose(max_jvp, true_max_jvp)

    two0_jvp = jax.jvp(eqx.two_norm, (x,), (zero,))
    rms0_jvp = jax.jvp(eqx.rms_norm, (x,), (zero,))
    max0_jvp = jax.jvp(eqx.max_norm, (x,), (zero,))
    assert tree_allclose(two0_jvp, (true_two, jnp.array(0.0)))
    assert tree_allclose(rms0_jvp, (true_rms, jnp.array(0.0)))
    assert tree_allclose(max0_jvp, (true_max, jnp.array(0.0)))


def test_zero():
    zero = [jnp.array(0.0), jnp.zeros((2, 2))]
    tx = [jnp.array(0.5), jnp.arange(1.0, 5.0).reshape(2, 2)]
    for t in (zero, tx):
        two0 = jax.jvp(eqx.two_norm, (zero,), (t,))
        rms0 = jax.jvp(eqx.rms_norm, (zero,), (t,))
        max0 = jax.jvp(eqx.max_norm, (zero,), (t,))
        true0 = (jnp.array(0.0), jnp.array(0.0))
        assert tree_allclose(two0, true0)
        assert tree_allclose(rms0, true0)
        assert tree_allclose(max0, true0)


def test_complex():
    x = jnp.array([3 + 1.2j, -0.5 + 4.9j])
    tx = jnp.array([2 - 0.3j, -0.7j])
    two = jax.jvp(eqx.two_norm, (x,), (tx,))
    true_two = jax.jvp(_two_norm, (x,), (tx,))
    rms = jax.jvp(eqx.rms_norm, (x,), (tx,))
    true_rms = jax.jvp(_rms_norm, (x,), (tx,))
    max = jax.jvp(eqx.max_norm, (x,), (tx,))
    true_max = jax.jvp(_max_norm, (x,), (tx,))
    assert two[0].imag == 0
    assert tree_allclose(two, true_two)
    assert rms[0].imag == 0
    assert tree_allclose(rms, true_rms)
    assert max[0].imag == 0
    assert tree_allclose(max, true_max)


def test_size_zero():
    zero = jnp.array(0.0)
    for x in (jnp.array([]), [jnp.array([]), jnp.array([])]):
        assert tree_allclose(eqx.two_norm(x), zero)
        assert tree_allclose(eqx.rms_norm(x), zero)
        assert tree_allclose(eqx.max_norm(x), zero)
        assert tree_allclose(jax.jvp(eqx.two_norm, (x,), (x,)), (zero, zero))
        assert tree_allclose(jax.jvp(eqx.rms_norm, (x,), (x,)), (zero, zero))
        assert tree_allclose(jax.jvp(eqx.max_norm, (x,), (x,)), (zero, zero))
