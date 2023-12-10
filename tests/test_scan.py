import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest

from .helpers import tree_allclose


@pytest.mark.parametrize("kind", ("lax", "checkpointed"))
@pytest.mark.parametrize("length", (4, None))
@pytest.mark.parametrize("vjp", (True, False))
def test_scan(kind, length, vjp, getkey):
    def f(carry, x):
        carry1, carry2 = carry
        x1, x2 = x
        carry1 = carry1 + 1
        carry2 = carry2 - 1
        y1 = 2 * x1 + carry1
        y2 = 2 * x2 + carry2
        return (carry1, carry2), (y1, y2)

    init = (0, 2)
    xs = (jnp.arange(4.0), jnp.arange(8.0).reshape(4, 2))

    def lax_run(init, xs):
        final, ys = lax.scan(f, init, xs)
        return ys, final

    def run(init, xs):
        final, ys = eqxi.scan(f, init, xs, length=length, kind=kind)
        return ys, final

    if vjp:

        @jax.jit
        def vjp_lax_run(init, xs):
            lax_out, lax_vjp_fn, lax_aux = jax.vjp(lax_run, init, xs, has_aux=True)
            ct_out = jtu.tree_map(
                lambda x: jr.normal(getkey(), x.shape, x.dtype), lax_out
            )
            lax_out = (lax_out, lax_aux)
            lax_grad = lax_vjp_fn(ct_out)
            return lax_out, lax_grad, ct_out

        @jax.jit
        def vjp_run(init, xs, ct_out):
            out, vjp_fn, aux = jax.vjp(run, init, xs, has_aux=True)
            out = (out, aux)
            grad = vjp_fn(ct_out)
            return out, grad

        lax_out, lax_grad, ct_out = vjp_lax_run(init, xs)
        out, grad = vjp_run(init, xs, ct_out)
        assert tree_allclose(grad, lax_grad)
    else:
        lax_out = jax.jit(lax_run)(init, xs)
        out = jax.jit(run)(init, xs)

    assert tree_allclose(out, lax_out)
