import functools as ft

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest

import equinox as eqx
import equinox.internal as eqxi

from .helpers import shaped_allclose


def _get_problem(key, *, num_steps):
    valkey, modelkey = jr.split(key)

    def cond_fun(carry):
        if num_steps is None:
            return True
        else:
            step, _ = carry
            return step < num_steps

    def make_body_fun(dynamic_mlp):
        mlp = eqx.combine(dynamic_mlp, static_mlp)

        def body_fun(carry):
            # A simple new_val = mlp(val) tends to converge to a fixed point in just a few
            # iterations, which implies zero gradient... which doesn't make for a test that
            # actually tests anything. Making things rotational like this keeps things more
            # interesting.
            step, val = carry
            (theta,) = mlp(val)
            real, imag = val
            z = real + imag * 1j
            z = z * jnp.exp(1j * theta)
            real = jnp.real(z)
            imag = jnp.imag(z)
            jax.debug.print("{}", step)
            return step + 1, jnp.stack([real, imag])

        return body_fun

    init_val = jr.normal(valkey, (2,))
    mlp = eqx.nn.MLP(2, 1, 2, 2, key=modelkey)
    dynamic_mlp, static_mlp = eqx.partition(mlp, eqx.is_array)

    return cond_fun, make_body_fun, init_val, dynamic_mlp


def _while_as_scan(cond, body, init_val, max_steps):
    def f(val, _):
        val2 = lax.cond(cond(val), body, lambda x: x, val)
        return val2, None

    final_val, _ = lax.scan(f, init_val, xs=None, length=max_steps)
    return final_val


def test_notangent_forward(getkey):
    cond_fun, make_body_fun, init_val, mlp = _get_problem(getkey(), num_steps=5)
    body_fun = make_body_fun(mlp)
    true_final_val = lax.while_loop(cond_fun, body_fun, (0, init_val))
    final_val = eqxi.checkpointed_while_loop(
        cond_fun, body_fun, (0, init_val), max_steps=None, checkpoints=1
    )
    assert shaped_allclose(final_val, true_final_val)


def test_forward(getkey):
    cond_fun, make_body_fun, init_val, mlp = _get_problem(getkey(), num_steps=5)
    body_fun = make_body_fun(mlp)
    true_final_val = lax.while_loop(cond_fun, body_fun, (0, init_val))

    @jax.jit
    def run(init_val):
        return eqxi.checkpointed_while_loop(
            cond_fun, body_fun, (0, init_val), max_steps=None, checkpoints=9
        )

    final_val, _ = jax.linearize(run, init_val)
    assert shaped_allclose(final_val, true_final_val)


@pytest.mark.parametrize(
    "checkpoints, num_steps, backward_order",
    [
        (1, 3, "0,1,2,0,1,0"),  # Stumm--Walther
        (1, 5, "0,1,2,3,4,0,1,2,3,0,1,2,0,1,0"),  # Use trivial Wang--Moin
        (2, 8, "7,0,1,2,3,4,5,6,3,4,5,3,4,3,0,1,2,1,0"),  # Use simple Wang--Moin
        # Use nontrivial Wang--Moin
        (3, 11, "10,4,5,6,7,8,9,7,8,7,4,5,6,5,4,0,1,2,3,2,1,0"),
        (9, 5, "4,3,2,1,0"),  # Fewer steps than checkpoints
        (2, 5, "3,4,3,0,1,2,1,0"),  # More steps than checkpoints; Stumm-Walther
        # All of these are from the Stumm--Walther paper, Fig 2.2.
        # Note that the figure commits an off-by-one-error. (It seems to me.)
        # The number of steps shown is one greater than it should be for the checkpoint
        # pattern shown.
        (4, 4, "3,2,1,0"),
        (4, 5, "3,4,3,2,1,0"),
        (4, 6, "5,2,3,4,3,2,1,0"),
        (4, 7, "6,5,2,3,4,3,2,0,1,0"),
        (4, 8, "7,6,5,0,1,2,3,4,3,2,1,0"),
        (4, 9, "7,8,7,6,5,0,1,2,3,4,3,2,1,0"),
        (4, 10, "9,6,7,8,7,6,5,0,1,2,3,4,3,2,1,0"),
        (4, 11, "10,9,5,6,7,8,7,6,5,0,1,2,3,4,3,2,1,0"),
        (4, 12, "10,11,10,9,5,6,7,8,7,6,5,0,1,2,3,4,3,2,1,0"),
        (4, 13, "12,9,10,11,10,9,5,6,7,8,7,6,5,0,1,2,3,4,3,2,1,0"),
        (4, 14, "12,13,12,9,10,11,10,9,5,6,7,8,7,6,5,0,1,2,3,4,3,2,1,0"),
    ],
)
@pytest.mark.parametrize("with_max_steps", [True, False])
def test_backward(
    checkpoints, num_steps, backward_order, with_max_steps, getkey, capfd
):
    if with_max_steps:
        max_steps = num_steps
        get_num_steps = None
    else:
        max_steps = None
        get_num_steps = num_steps
    cond_fun, make_body_fun, init_val, mlp = _get_problem(
        getkey(), num_steps=get_num_steps
    )

    @jax.jit
    @jax.value_and_grad
    def true_run(arg):
        init_val, mlp = arg
        body_fun = make_body_fun(mlp)
        _, true_final_val = _while_as_scan(
            cond_fun, body_fun, (0, init_val), max_steps=num_steps
        )
        return jnp.sum(true_final_val)

    @jax.jit
    @jax.value_and_grad
    def run(arg):
        init_val, mlp = arg
        body_fun = make_body_fun(mlp)
        _, final_val = eqxi.checkpointed_while_loop(
            cond_fun,
            body_fun,
            (0, init_val),
            max_steps=max_steps,
            checkpoints=checkpoints,
        )
        return jnp.sum(final_val)

    true_value, true_grad = true_run((init_val, mlp))
    capfd.readouterr()
    value, grad = run((init_val, mlp))
    text, _ = capfd.readouterr()
    true_text = "".join(f"{i}\n" for i in range(num_steps)) + backward_order.replace(
        ",", "\n"
    )
    assert shaped_allclose(value, true_value)
    assert shaped_allclose(grad, true_grad)
    assert text.strip() == true_text


def test_vmap_primal_unbatched_cond(getkey):
    cond_fun, make_body_fun, init_val, mlp = _get_problem(getkey(), num_steps=14)

    @jax.jit
    @ft.partial(jax.vmap, in_axes=((0, None),))
    @jax.value_and_grad
    def true_run(arg):
        init_val, mlp = arg
        body_fun = make_body_fun(mlp)
        _, true_final_val = _while_as_scan(
            cond_fun, body_fun, (0, init_val), max_steps=14
        )
        return jnp.sum(true_final_val)

    @jax.jit
    @ft.partial(jax.vmap, in_axes=((0, None),))
    @jax.value_and_grad
    def run(arg):
        init_val, mlp = arg
        body_fun = make_body_fun(mlp)
        _, final_val = eqxi.checkpointed_while_loop(
            cond_fun,
            body_fun,
            (0, init_val),
            max_steps=None,
            checkpoints=4,
        )
        return jnp.sum(final_val)

    init_val = jtu.tree_map(
        lambda x: jr.normal(getkey(), (3,) + x.shape, x.dtype), init_val
    )
    true_value, true_grad = true_run((init_val, mlp))
    value, grad = run((init_val, mlp))
    assert shaped_allclose(value, true_value)
    assert shaped_allclose(grad, true_grad)


def test_vmap_primal_batched_cond(getkey):
    cond_fun, make_body_fun, init_val, mlp = _get_problem(getkey(), num_steps=14)

    @jax.jit
    @ft.partial(jax.vmap, in_axes=((0, None), 0))
    @jax.value_and_grad
    def true_run(arg, init_step):
        init_val, mlp = arg
        body_fun = make_body_fun(mlp)
        _, true_final_val = _while_as_scan(
            cond_fun, body_fun, (init_step, init_val), max_steps=14
        )
        return jnp.sum(true_final_val)

    @jax.jit
    @ft.partial(jax.vmap, in_axes=((0, None), 0))
    @jax.value_and_grad
    def run(arg, init_step):
        init_val, mlp = arg
        body_fun = make_body_fun(mlp)
        _, final_val = eqxi.checkpointed_while_loop(
            cond_fun,
            body_fun,
            (init_step, init_val),
            max_steps=None,
            checkpoints=4,
        )
        return jnp.sum(final_val)

    init_step = jnp.array([0, 1, 2, 3, 5, 10])
    init_val = jtu.tree_map(
        lambda x: jr.normal(getkey(), (6,) + x.shape, x.dtype), init_val
    )
    true_value, true_grad = true_run((init_val, mlp), init_step)
    value, grad = run((init_val, mlp), init_step)
    assert shaped_allclose(value, true_value, rtol=1e-4, atol=1e-4)
    assert shaped_allclose(grad, true_grad, rtol=1e-4, atol=1e-4)


def test_vmap_cotangent(getkey):
    cond_fun, make_body_fun, init_val, mlp = _get_problem(getkey(), num_steps=14)

    @jax.jit
    @jax.jacrev
    def true_run(arg):
        init_val, mlp = arg
        body_fun = make_body_fun(mlp)
        _, true_final_val = _while_as_scan(
            cond_fun, body_fun, (0, init_val), max_steps=14
        )
        return jnp.sum(true_final_val)

    @jax.jit
    @jax.jacrev
    def run(arg):
        init_val, mlp = arg
        body_fun = make_body_fun(mlp)
        _, final_val = eqxi.checkpointed_while_loop(
            cond_fun,
            body_fun,
            (0, init_val),
            max_steps=None,
            checkpoints=4,
        )
        return jnp.sum(final_val)

    true_jac = true_run((init_val, mlp))
    jac = run((init_val, mlp))
    assert shaped_allclose(jac, true_jac, rtol=1e-4, atol=1e-4)
