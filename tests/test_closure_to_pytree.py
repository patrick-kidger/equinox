import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax


def test_fixup_optax():
    lr = jnp.array(1e-3)
    optim = optax.chain(
        optax.adam(lr),
        optax.scale_by_schedule(optax.piecewise_constant_schedule(1, {200: 0.1})),
    )
    optim = eqxi.closure_to_pytree(optim)

    for leaf in jtu.tree_leaves(optim):
        if eqx.is_array(leaf) and leaf == -lr:
            break
    else:
        assert False

    # Check that we can still init and update as normal.
    grads = params = {"foo": jnp.array(1.0)}
    state = optim.init(params)
    with jax.numpy_dtype_promotion("standard"):
        optim.update(grads, state)

    lr = jnp.array(1e-2)
    optim2 = optax.chain(
        optax.adam(lr),
        optax.scale_by_schedule(optax.piecewise_constant_schedule(1, {200: 0.1})),
    )
    optim2 = eqxi.closure_to_pytree(optim2)

    compiling = 0

    @eqx.filter_jit
    def f(x):
        nonlocal compiling
        compiling += 1

    f(optim)
    assert compiling == 1
    f(optim2)
    assert compiling == 1


def test_closure_same_name():
    def f(flag):
        if flag:

            def g(y):
                return 1 + y

        else:

            def g(y):
                return 2 + y

        def h(y):
            return g(y)

        return h

    h1 = eqxi.closure_to_pytree(f(True))
    h2 = eqxi.closure_to_pytree(f(False))

    @eqx.filter_jit
    def run(f, y):
        return f(y)

    assert run(h1, jnp.array(1.0)) == 2
    assert run(h2, jnp.array(1.0)) == 3
