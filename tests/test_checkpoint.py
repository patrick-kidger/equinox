from functools import partial

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu


def test_checkpoint(getkey):
    mlp = eqx.nn.MLP(1, 2, 5, 2, key=getkey())

    def fun(mlp, x, checkpoint):
        if checkpoint:
            mlp = eqx.filter_checkpoint(mlp)
        return jnp.sum(mlp(x))

    x = jnp.array([1.0])

    grad_f_check = eqx.filter_grad(partial(fun, checkpoint=True))
    grad_f_nocheck = eqx.filter_grad(partial(fun, checkpoint=False))

    for l1, l2 in zip(
        jtu.tree_leaves(grad_f_check(mlp, x)), jtu.tree_leaves(grad_f_nocheck(mlp, x))
    ):
        assert jnp.allclose(l1, l2)
