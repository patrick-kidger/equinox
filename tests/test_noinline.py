import jax
import jax.numpy as jnp
import jax.random as jr
from helpers import shaped_allclose

import equinox as eqx


def test_unchanged(getkey):
    mlp = eqx.nn.MLP(2, 2, 512, 2, key=getkey())
    mlp_noinline = eqx.experimental.noinline(mlp)
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
    mlp_noinline = eqx.experimental.noinline(mlp)
    mlp_vmap = jax.vmap(mlp)
    mlp_jit_vmap = eqx.filter_jit(mlp_vmap)
    mlp_vmap_noinline = jax.vmap(mlp_noinline)
    mlp_jit_vmap_noinline = eqx.filter_jit(mlp_vmap_noinline)
    x = jr.normal(getkey(), (5, 2))
    o1 = mlp_jit_vmap(x)
    o2 = mlp_vmap_noinline(x)
    o3 = mlp_jit_vmap_noinline(x)
    assert shaped_allclose(o1, o2)
    assert shaped_allclose(o1, o3)


def test_jvp_unchanged(getkey):
    mlp = eqx.nn.MLP(2, 2, 512, 2, key=getkey())
    mlp_noinline = eqx.experimental.noinline(mlp)
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
    mlp_noinline = eqx.experimental.noinline(mlp)
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

    def fn(x):
        nonlocal num_traces
        num_traces += 1
        return x * 2

    fn_noinline = eqx.experimental.noinline(fn)
    assert shaped_allclose(fn_noinline(1), jnp.array(2))
    assert num_traces == 2  # eval_shape + jit
    assert shaped_allclose(fn_noinline(2), jnp.array(4))
    assert num_traces == 2

    fn_jit_noinline = jax.jit(fn_noinline)
    assert shaped_allclose(fn_jit_noinline(1), jnp.array(2))
    assert num_traces == 2
    assert shaped_allclose(fn_jit_noinline(2), jnp.array(4))
    assert num_traces == 2
