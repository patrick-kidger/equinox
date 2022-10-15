import jax
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx

from .helpers import shaped_allclose


def test_simple():
    @jax.jit
    @eqx.internal.noinline
    def addone(x):
        return x + 1

    assert shaped_allclose(addone(1), jnp.array(2))


def test_unchanged(getkey):
    mlp = eqx.nn.MLP(2, 2, 512, 2, key=getkey())
    mlp_noinline = eqx.internal.noinline(mlp)
    mlp_jit = eqx.filter_jit(mlp)
    mlp_jit_noinline = eqx.filter_jit(mlp_noinline)
    x = jr.normal(getkey(), (2,))
    o1 = mlp_jit(x)
    o2 = mlp_noinline(x)
    o3 = mlp_jit_noinline(x)
    assert shaped_allclose(o1, o2)
    assert shaped_allclose(o1, o3)


def test_vmap_unchanged(getkey):
    mlp = eqx.nn.MLP(2, 2, 512, 2, key=getkey())
    mlp_noinline = eqx.internal.noinline(mlp)
    mlp_vmap = jax.vmap(mlp)
    mlp_jit_vmap = eqx.filter_jit(mlp_vmap)
    mlp_vmap_noinline = jax.vmap(mlp_noinline)
    mlp_jit_vmap_noinline = eqx.filter_jit(mlp_vmap_noinline)
    x = jr.normal(getkey(), (5, 2))
    o1 = mlp_jit_vmap(x)
    o2 = mlp_vmap_noinline(x)
    o3 = mlp_jit_vmap_noinline(x)
    assert shaped_allclose(o1, o2, atol=1e-5, rtol=1e-5)
    assert shaped_allclose(o1, o3)


def test_jvp_unchanged(getkey):
    mlp = eqx.nn.MLP(2, 2, 512, 2, key=getkey())
    mlp_noinline = eqx.internal.noinline(mlp)
    mlp_jvp = lambda p, t: jax.jvp(mlp, (p,), (t,))
    mlp_jit_jvp = eqx.filter_jit(mlp_jvp)
    mlp_jvp_noinline = lambda p, t: jax.jvp(mlp_noinline, (p,), (t,))
    mlp_jit_jvp_noinline = eqx.filter_jit(mlp_jvp_noinline)
    x = jr.normal(getkey(), (2,))
    y = jr.normal(getkey(), (2,))
    o1 = mlp_jit_jvp(x, y)
    o2 = mlp_jvp_noinline(x, y)
    o3 = mlp_jit_jvp_noinline(x, y)
    assert shaped_allclose(o1, o2)
    assert shaped_allclose(o1, o3)


def test_grad_unchanged(getkey):
    mlp = eqx.nn.MLP(2, 2, 512, 2, key=getkey())
    mlp_noinline = eqx.internal.noinline(mlp)
    mlp_grad = jax.grad(lambda x: jnp.sum(mlp(x)))
    mlp_jit_grad = eqx.filter_jit(mlp_grad)
    mlp_grad_noinline = jax.grad(lambda x: jnp.sum(mlp_noinline(x)))
    mlp_jit_grad_noinline = eqx.filter_jit(mlp_grad_noinline)
    x = jr.normal(getkey(), (2,))
    o1 = mlp_jit_grad(x)
    o2 = mlp_grad_noinline(x)
    o3 = mlp_jit_grad_noinline(x)
    assert shaped_allclose(o1, o2)
    assert shaped_allclose(o1, o3)


def test_num_traces():
    num_traces = 0

    @jax.jit
    @eqx.internal.noinline
    def fn(x):
        nonlocal num_traces
        num_traces += 1
        return x * 2

    assert shaped_allclose(fn(1), jnp.array(2))
    assert num_traces == 2
    assert shaped_allclose(fn(2), jnp.array(4))
    assert num_traces == 2


def test_pytree_in():
    @eqx.filter_jit
    @eqx.internal.noinline
    def fn(f, x):
        return f(x[0][0])

    o1 = fn(lambda x: x + 1, [(1,)])
    o2 = fn(lambda x: x + 1, ([jnp.array(1)],))
    assert shaped_allclose(o1, jnp.array(2))
    assert shaped_allclose(o2, jnp.array(2))
