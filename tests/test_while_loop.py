import functools as ft
import timeit
from typing import Optional

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.lib
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
from jaxtyping import Array

from .helpers import tree_allclose


def _get_problem(key, *, num_steps: Optional[int]):
    valkey1, valkey2, modelkey = jr.split(key, 3)

    def cond_fun(carry):
        if num_steps is None:
            return True
        else:
            step, _, _ = carry
            return step < num_steps

    def make_body_fun(dynamic_mlp):
        mlp = eqx.combine(dynamic_mlp, static_mlp)

        def body_fun(carry):
            # A simple new_val = mlp(val) tends to converge to a fixed point in just a
            # few iterations, which implies zero gradient... which doesn't make for a
            # test that actually tests anything. Making things rotational like this
            # keeps things more interesting.
            step, val1, val2 = carry
            (theta,) = mlp(val1)
            real, imag = val1
            z = real.astype(jnp.complex64) + imag.astype(jnp.complex64) * 1j
            z = z * jnp.exp(1j * theta.astype(jnp.complex64))
            real = jnp.real(z)
            imag = jnp.imag(z)
            jax.debug.print("{}", step)  # pyright: ignore
            val1 = jnp.stack([real, imag])
            val2 = val2.at[step].set(real)
            return step + 1, val1, val2

        return body_fun

    init_val1 = jr.normal(valkey1, (2,))
    init_val2 = jr.normal(valkey2, (20,))
    if num_steps is not None:
        # So that things fit in the buffer update above.
        assert num_steps < 20
    mlp = eqx.nn.MLP(2, 1, 2, 2, key=modelkey)
    dynamic_mlp, static_mlp = eqx.partition(mlp, eqx.is_array)

    return cond_fun, make_body_fun, init_val1, init_val2, dynamic_mlp


def _while_as_scan(cond, body, init_val, max_steps):
    def f(val, _):
        val2 = lax.cond(cond(val), body, lambda x: x, val)
        return val2, None

    final_val, _ = lax.scan(f, init_val, xs=None, length=max_steps)
    return final_val


@pytest.mark.parametrize("buffer", (False, True))
@pytest.mark.parametrize("kind", ("lax", "bounded", "checkpointed"))
def test_notangent_forward(buffer, kind, getkey):
    cond_fun, make_body_fun, init_val1, init_val2, mlp = _get_problem(
        getkey(), num_steps=5
    )
    body_fun = make_body_fun(mlp)
    true_final_carry = lax.while_loop(cond_fun, body_fun, (0, init_val1, init_val2))
    if buffer:
        buffer_fn = lambda i: i[2]
    else:
        buffer_fn = None
    final_carry = eqxi.while_loop(
        cond_fun,
        body_fun,
        (0, init_val1, init_val2),
        max_steps=10 if kind == "bounded" else None,
        kind=kind,
        buffers=buffer_fn,
        checkpoints=1,
    )
    assert tree_allclose(final_carry, true_final_carry, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("buffer", (False, True))
@pytest.mark.parametrize("kind", ("lax", "bounded", "checkpointed"))
def test_forward(buffer, kind, getkey):
    cond_fun, make_body_fun, init_val1, init_val2, mlp = _get_problem(
        getkey(), num_steps=5
    )
    body_fun = make_body_fun(mlp)
    true_final_carry = lax.while_loop(cond_fun, body_fun, (0, init_val1, init_val2))

    @jax.jit
    def run(init_val1, init_val2):
        if buffer:
            buffer_fn = lambda i: i[2]
        else:
            buffer_fn = None
        return eqxi.while_loop(
            cond_fun,
            body_fun,
            (0, init_val1, init_val2),
            max_steps=10 if kind == "bounded" else None,
            buffers=buffer_fn,
            kind=kind,
            checkpoints=9,
        )

    final_carry, _ = jax.linearize(run, init_val1, init_val2)
    assert tree_allclose(final_carry, true_final_carry, atol=1e-4, rtol=1e-4)


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
@pytest.mark.parametrize("with_max_steps", (True, False))
@pytest.mark.parametrize("buffer", (False, True))
def test_backward_checkpointed(
    checkpoints, num_steps, backward_order, with_max_steps, buffer, getkey, capfd
):
    if with_max_steps:
        max_steps = num_steps
        get_num_steps = None
    else:
        max_steps = None
        get_num_steps = num_steps
    cond_fun, make_body_fun, init_val1, init_val2, mlp = _get_problem(
        getkey(), num_steps=get_num_steps
    )

    @jax.jit
    @jax.value_and_grad
    def true_run(arg):
        init_val1, init_val2, mlp = arg
        body_fun = make_body_fun(mlp)
        _, true_final_val1, true_final_val2 = _while_as_scan(
            cond_fun, body_fun, (0, init_val1, init_val2), max_steps=num_steps
        )
        return jnp.sum(true_final_val1) + jnp.sum(true_final_val2)

    @jax.jit
    @jax.value_and_grad
    def run(arg):
        init_val1, init_val2, mlp = arg
        if buffer:
            buffer_fn = lambda i: i[2]
        else:
            buffer_fn = None
        body_fun = make_body_fun(mlp)
        _, final_val1, final_val2 = eqxi.while_loop(
            cond_fun,
            body_fun,
            (0, init_val1, init_val2),
            max_steps=max_steps,
            buffers=buffer_fn,
            kind="checkpointed",
            checkpoints=checkpoints,
        )
        return jnp.sum(final_val1) + jnp.sum(final_val2)

    true_value, true_grad = true_run([init_val1, init_val2, mlp])
    capfd.readouterr()
    value, grad = run([init_val1, init_val2, mlp])
    text, _ = capfd.readouterr()
    true_text = "".join(f"{i}\n" for i in range(num_steps)) + backward_order.replace(
        ",", "\n"
    )
    if buffer:
        grad[1] = grad[1].at[:num_steps].set(0)
    assert tree_allclose(value, true_value, rtol=1e-4, atol=1e-4)
    assert tree_allclose(grad, true_grad, rtol=1e-4, atol=1e-4)
    assert true_text in text.strip()


@pytest.mark.parametrize("buffer", (False, True))
def test_backward_bounded(buffer, getkey):
    cond_fun, make_body_fun, init_val1, init_val2, mlp = _get_problem(
        getkey(), num_steps=None
    )

    @jax.jit
    @jax.value_and_grad
    def true_run(arg):
        init_val1, init_val2, mlp = arg
        body_fun = make_body_fun(mlp)
        _, true_final_val1, true_final_val2 = _while_as_scan(
            cond_fun, body_fun, (0, init_val1, init_val2), max_steps=14
        )
        return jnp.sum(true_final_val1) + jnp.sum(true_final_val2)

    @jax.jit
    @jax.value_and_grad
    def run(arg):
        init_val1, init_val2, mlp = arg
        if buffer:
            buffer_fn = lambda i: i[2]
        else:
            buffer_fn = None
        body_fun = make_body_fun(mlp)
        _, final_val1, final_val2 = eqxi.while_loop(
            cond_fun,
            body_fun,
            (0, init_val1, init_val2),
            max_steps=14,
            buffers=buffer_fn,
            kind="bounded",
        )
        return jnp.sum(final_val1) + jnp.sum(final_val2)

    true_value, true_grad = true_run([init_val1, init_val2, mlp])
    value, grad = run([init_val1, init_val2, mlp])
    if buffer:
        grad[1] = grad[1].at[:14].set(0)
    assert tree_allclose(value, true_value, rtol=1e-4, atol=1e-4)
    assert tree_allclose(grad, true_grad, rtol=1e-4, atol=1e-4)


def _maybe_value_and_grad(kind):
    if kind == "lax":

        def value_and_grad(fn):
            def wrapped(*args, **kwargs):
                return fn(*args, **kwargs), None

            return wrapped

        return value_and_grad
    else:
        return jax.value_and_grad


@pytest.mark.parametrize("buffer", (False, True))
@pytest.mark.parametrize("kind", ("lax", "bounded", "checkpointed"))
def test_vmap_primal_unbatched_cond(buffer, kind, getkey):
    cond_fun, make_body_fun, init_val1, init_val2, mlp = _get_problem(
        getkey(), num_steps=14
    )

    value_and_grad = _maybe_value_and_grad(kind)

    @jax.jit
    @ft.partial(jax.vmap, in_axes=([0, 0, None],))
    @value_and_grad
    def true_run(arg):
        init_val1, init_val2, mlp = arg
        body_fun = make_body_fun(mlp)
        _, true_final_val1, true_final_val2 = _while_as_scan(
            cond_fun, body_fun, (0, init_val1, init_val2), max_steps=14
        )
        return jnp.sum(true_final_val1) + jnp.sum(true_final_val2)

    @jax.jit
    @ft.partial(jax.vmap, in_axes=([0, 0, None],))
    @value_and_grad
    def run(arg):
        init_val1, init_val2, mlp = arg
        if buffer:
            buffer_fn = lambda i: i[2]
        else:
            buffer_fn = None
        body_fun = make_body_fun(mlp)
        _, final_val1, final_val2 = eqxi.while_loop(
            cond_fun,
            body_fun,
            (0, init_val1, init_val2),
            max_steps=16 if kind == "bounded" else None,
            buffers=buffer_fn,
            kind=kind,
            checkpoints=4,
        )
        return jnp.sum(final_val1) + jnp.sum(final_val2)

    init_val1, init_val2 = jtu.tree_map(
        lambda x: jr.normal(getkey(), (3,) + x.shape, x.dtype), (init_val1, init_val2)
    )
    true_value, true_grad = true_run([init_val1, init_val2, mlp])
    value, grad = run([init_val1, init_val2, mlp])
    if buffer and kind != "lax":
        grad[1] = grad[1].at[:, :14].set(0)
    assert tree_allclose(value, true_value, atol=1e-4)
    assert tree_allclose(grad, true_grad, atol=1e-4)


@pytest.mark.parametrize("buffer", (False, True))
@pytest.mark.parametrize("kind", ("lax", "bounded", "checkpointed"))
def test_vmap_primal_batched_cond(buffer, kind, getkey):
    cond_fun, make_body_fun, init_val1, init_val2, mlp = _get_problem(
        getkey(), num_steps=14
    )

    value_and_grad = _maybe_value_and_grad(kind)

    @jax.jit
    @ft.partial(jax.vmap, in_axes=([0, 0, None], 0))
    @value_and_grad
    def true_run(arg, init_step):
        init_val1, init_val2, mlp = arg
        body_fun = make_body_fun(mlp)
        _, true_final_val1, true_final_val2 = _while_as_scan(
            cond_fun, body_fun, (init_step, init_val1, init_val2), max_steps=14
        )
        return jnp.sum(true_final_val1) + jnp.sum(true_final_val2)

    @jax.jit
    @ft.partial(jax.vmap, in_axes=([0, 0, None], 0))
    @value_and_grad
    def run(arg, init_step):
        init_val1, init_val2, mlp = arg
        if buffer:
            buffer_fn = lambda i: i[2]
        else:
            buffer_fn = None
        body_fun = make_body_fun(mlp)
        _, final_val1, final_val2 = eqxi.while_loop(
            cond_fun,
            body_fun,
            (init_step, init_val1, init_val2),
            max_steps=16 if kind == "bounded" else None,
            buffers=buffer_fn,
            kind=kind,
            checkpoints=4,
        )
        return jnp.sum(final_val1) + jnp.sum(final_val2)

    init_step = jnp.array([0, 1, 2, 3, 5, 10])
    init_val1, init_val2 = jtu.tree_map(
        lambda x: jr.normal(getkey(), (6,) + x.shape, x.dtype), (init_val1, init_val2)
    )
    true_value, true_grad = true_run([init_val1, init_val2, mlp], init_step)
    value, grad = run([init_val1, init_val2, mlp], init_step)
    if buffer and kind != "lax":
        for i, j in enumerate(init_step):
            grad[1] = grad[1].at[i, j.item() : 14].set(0)
    assert tree_allclose(value, true_value, rtol=1e-4, atol=1e-4)
    assert tree_allclose(grad, true_grad, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("buffer", (False, True))
@pytest.mark.parametrize("kind", ("bounded", "checkpointed"))
def test_vmap_cotangent(buffer, kind, getkey):
    cond_fun, make_body_fun, init_val1, init_val2, mlp = _get_problem(
        getkey(), num_steps=14
    )

    @jax.jit
    @jax.jacrev
    def true_run(arg, init_val2):
        init_val1, mlp = arg
        body_fun = make_body_fun(mlp)
        _, true_final_val1, true_final_val2 = _while_as_scan(
            cond_fun, body_fun, (0, init_val1, init_val2), max_steps=14
        )
        return true_final_val1, true_final_val2

    @jax.jit
    @jax.jacrev
    def run(arg, init_val2):
        init_val1, mlp = arg
        if buffer:
            buffer_fn = lambda i: i[2]
        else:
            buffer_fn = None
        body_fun = make_body_fun(mlp)
        _, final_val1, final_val2 = eqxi.while_loop(
            cond_fun,
            body_fun,
            (0, init_val1, init_val2),
            max_steps=16 if kind == "bounded" else None,
            buffers=buffer_fn,
            kind=kind,
            checkpoints=4,
        )
        return final_val1, final_val2

    true_jac = true_run((init_val1, mlp), init_val2)
    jac = run((init_val1, mlp), init_val2)
    assert tree_allclose(jac, true_jac, rtol=1e-4, atol=1e-4)


# This test might be superfluous?
#
# This tests that XLA correctly optimises
# select(pred, dynamic_update_slice(xs, i, x), xs)
# into
# dynamic_update_slice(xs, i, select(pred, dynamic_slice(xs, i), x)))
#
# This was a fix I contributed to XLA, that is present in jaxlib>=0.4.2.
# In practice handling scatter as well as dynamic_update_slice was a bit too hard, and
# in practice we do need scatter (.at[i].set() with vmap'd i) so we still need to
# implement a workaround in the `buffers` of `while_loop`. So maybe this
# doesn't matter?
#
# Anyway, we still test it just to be sure.
#
# This test is really just checking `lax.while_loop`, but we throw in
# `while_loop` too, because why not?
@pytest.mark.parametrize("inplace_op", ["scatter", "dynamic_update_slice"])
@pytest.mark.parametrize(
    "while_loop",
    [
        lax.while_loop,
        ft.partial(eqxi.while_loop, kind="checkpointed", checkpoints=50_000),
    ],
)
def test_speed_while(inplace_op, while_loop):
    @jax.jit
    @jax.vmap
    def f(init_step, init_xs):
        def cond(carry):
            step, xs = carry
            return step < xs.size

        def body(carry):
            step, xs = carry
            if inplace_op == "scatter":
                xs = xs.at[step].set(1)
            elif inplace_op == "dynamic_update_slice":
                xs = lax.dynamic_update_index_in_dim(xs, 1.0, step, 0)
            else:
                assert False
            return step + 1, xs

        return while_loop(cond, body, (init_step, init_xs))

    size = 100_000
    args = jnp.array([0]), jnp.zeros((1, size))
    f(*args)  # compile

    speed = timeit.timeit(lambda: f(*args), number=1)
    # Takes O(1e-3) with optimisation.
    # Takes O(10) without optimisation.
    # So we have two orders of magnitude safety margin each way, so the test shouldn't
    # be flaky.
    assert speed < 0.1


# This tests the possible failure mode of "the buffer doesn't do anything".
# This test takes O(1e-1) seconds with buffer.
# This test takes O(10) seconds without buffer.
# This speed improvement is precisely the reason that buffer exists.
@pytest.mark.parametrize("read", (False, True))
def test_speed_buffer_while(read):
    @jax.jit
    @jax.vmap
    def f(init_step, init_xs):
        def cond(carry):
            step, xs = carry
            return step < xs.size

        def body(carry):
            step, xs = carry
            if read:
                i = jr.randint(jr.PRNGKey(step), shape=(), minval=0, maxval=step)
                xs = xs.at[step].set(2 * jnp.tanh(xs[i]))
            else:
                xs = xs.at[step].set(1)
            return step + 1, xs

        def loop(init_xs):
            return eqxi.while_loop(
                cond,
                body,
                (init_step, init_xs),
                buffers=lambda i: i[1],
                kind="checkpointed",
                checkpoints=50_000,
            )

        # Linearize so that we save residuals
        return jax.linearize(loop, init_xs)

    size = 100_000
    # nontrivial batch size is important to ensure that the `.at[].set()` is really a
    # scatter, and that XLA doesn't optimise it into a dynamic_update_slice. (Which
    # can be switched with `select` in the compiler.)
    args = jnp.array([0, 1]), jnp.zeros((2, size))
    f(*args)  # compile

    speed = timeit.timeit(lambda: f(*args), number=1)
    assert speed < 1


@pytest.mark.skipif(
    jax.lib.__version__ == "0.4.16",  # pyright: ignore
    reason="jaxlib bug; see https://github.com/google/jax/pull/17724",
)
# This isn't testing any particular failure mode: just that things generally work.
def test_speed_grad_checkpointed_while(getkey):
    mlp = eqx.nn.MLP(2, 1, 2, 2, key=getkey())
    checkpoints = 10_000

    @jax.jit
    @jax.vmap
    @jax.grad
    def f(init_val, init_step):
        def cond(carry):
            step, _ = carry
            return step < 2 * checkpoints

        def body(carry):
            step, val = carry
            (theta,) = mlp(val)
            real, imag = val
            z = real.astype(jnp.complex64) + imag.astype(jnp.complex64) * 1j
            z = z * jnp.exp(1j * theta.astype(jnp.complex64))
            real = jnp.real(z)
            imag = jnp.imag(z)
            return step + 1, jnp.stack([real, imag])

        _, final_xs = eqxi.while_loop(
            cond,
            body,
            (init_step, init_val),
            kind="checkpointed",
            checkpoints=checkpoints,
        )
        return jnp.sum(final_xs)

    init_step = jnp.array([0, 10])
    init_val = jr.normal(getkey(), (2, 2))

    f(init_val, init_step)  # compile
    speed = timeit.timeit(lambda: f(init_val, init_step), number=1)
    # Should take ~0.01 seconds
    assert speed < 0.5


# This is deliberately meant to emulate the pattern of saving used in
# `diffrax.diffeqsolve(..., saveat=SaveAt(ts=...))`.
@pytest.mark.parametrize("read", (False, True))
def test_nested_loops(read, getkey):
    @ft.partial(jax.jit, static_argnums=5)
    @ft.partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, None))
    def run(step, vals, ts, final_step, cotangents, true):
        val1, val2, val3, val4 = vals
        value, vjp_fn = jax.vjp(
            lambda _val1: outer_loop(
                step, (_val1, val2, val3, val4), ts, true, final_step
            ),
            val1,
        )
        cotangents = vjp_fn(cotangents)
        return value, cotangents

    def outer_loop(step, vals, ts, true, final_step):
        def cond(carry):
            step, _ = carry
            return step < final_step

        def body(carry):
            step, (val1, val2, val3, val4) = carry
            mul = 1 + 0.05 * jnp.sin(105 * val1 + 1)
            val1 = val1 * mul
            return inner_loop(step, (val1, val2, val3, val4), ts, true)

        def buffers(carry):
            _, (_, val2, val3, _) = carry
            return val2, val3

        if true:
            while_loop = ft.partial(_while_as_scan, max_steps=50)
        else:
            while_loop = ft.partial(
                eqxi.while_loop, max_steps=50, buffers=buffers, kind="checkpointed"
            )
        _, out = while_loop(cond, body, (step, vals))
        return out

    def inner_loop(step, vals, ts, true):
        ts_done = jnp.floor(ts[step] + 1)

        def cond(carry):
            step, _ = carry
            return ts[step] < ts_done

        def body(carry):
            step, (val1, val2, val3, val4) = carry
            mul = 1 + 0.05 * jnp.sin(100 * val1 + 3)
            val1 = val1 * mul
            if read:
                i2 = jr.randint(jr.PRNGKey(step), shape=(), minval=0, maxval=step)
                i3 = jr.randint(jr.PRNGKey(step + 100), shape=(), minval=0, maxval=step)
                i4 = jr.randint(jr.PRNGKey(step + 200), shape=(), minval=0, maxval=step)
                setval2 = val1 + val2[i2]
                setval3 = 0.8 * val1 + val2[i3] + val3[i3]
                setval4 = val1 + jnp.tanh(val2[i2] * val4[i4])
                setval2 = jnp.where(step == 0, 2.0, setval2)
                setval3 = jnp.where(step == 0, 3.0, setval3)
                setval4 = jnp.where(step == 0, 4.0, setval4)
            else:
                setval2 = val1
                setval3 = val1
                setval4 = val1
            val2 = val2.at[step].set(setval2)
            val3 = val3.at[step].set(setval3)
            val4 = val4.at[step].set(setval4)
            return step + 1, (val1, val2, val3, val4)

        def buffers(carry):
            _, (_, _, val3, val4) = carry
            return val3, val4

        if true:
            while_loop = ft.partial(_while_as_scan, max_steps=10)
        else:
            while_loop = ft.partial(
                eqxi.while_loop, max_steps=10, buffers=buffers, kind="checkpointed"
            )
        return while_loop(cond, body, (step, vals))

    step = jnp.array([0, 5])
    val1 = jr.uniform(getkey(), shape=(2,), minval=0.1, maxval=0.7)
    val2 = val3 = val4 = jnp.zeros((2, 47))
    ts = jnp.stack([jnp.linspace(0, 19, 47), jnp.linspace(0, 13, 47)])
    final_step = jnp.array([46, 43])
    cotangents = (
        jr.normal(getkey(), (2,)),
        jr.normal(getkey(), (2, 47)),
        jr.normal(getkey(), (2, 47)),
        jr.normal(getkey(), (2, 47)),
    )

    value, grads = run(
        step, (val1, val2, val3, val4), ts, final_step, cotangents, False
    )
    true_value, true_grads = run(
        step, (val1, val2, val3, val4), ts, final_step, cotangents, True
    )

    assert tree_allclose(value, true_value)
    assert tree_allclose(grads, true_grads, rtol=1e-4, atol=1e-5)


def test_zero_buffer():
    def cond_fun(carry):
        step, val = carry
        return step < 5

    def body_fun(carry):
        step, val = carry
        return step + 1, val

    init_step = 0
    init_val = jnp.zeros((5, 0))
    init_carry = (init_step, init_val)

    def run(init_carry):
        return eqxi.while_loop(
            cond_fun, body_fun, init_carry, kind="checkpointed", max_steps=5
        )

    jax.linearize(run, init_carry)


def test_symbolic_zero(capfd):
    def cond_fun(carry):
        return True

    def body_fun(carry):
        carry0, carry1, carry2, carry3, carry4, carry5 = carry
        return carry2, carry3, lax.stop_gradient(carry1), carry3, carry5, carry4

    @jax.grad
    def run(init_carry):
        init_carry = init_carry + (5, jnp.array(6))
        outs = eqxi.while_loop(
            cond_fun, body_fun, init_carry, kind="checkpointed", max_steps=5
        )
        return outs[0]

    capfd.readouterr()
    run((1.0, 2.0, jnp.array(3.0), jnp.array(4.0)))
    text, _ = capfd.readouterr()
    assert (
        "symbolic_zero_gradient "
        "(True, True, True, (False, True, False, True, True, True))"
    ) in text


def test_disable_jit():
    def cond_fun(carry):
        return True

    def body_fun(carry):
        return 5

    with jax.disable_jit():
        eqxi.while_loop(cond_fun, body_fun, 3, max_steps=3, kind="checkpointed")


def test_buffer_index():
    def cond_fun(carry):
        return True

    def body_fun(carry):
        return carry.at[..., 1:].set(0)

    def buffers(carry):
        return carry

    init = jnp.zeros((3, 5))
    eqxi.while_loop(
        cond_fun, body_fun, init, kind="checkpointed", buffers=buffers, max_steps=2
    )


# This test includes complexities like buffers, exact ararys, and `None`, just to be
# sure we handle these complexities here too.
def test_nondifferentiable_body1():
    def cond_fun(carry):
        return True

    def body_fun(carry):
        step, x, y, z, _ = carry
        y2 = eqxi.nondifferentiable(y)
        return step + 1, x + y2, y + 1, z.at[step].set(y), None

    @eqx.filter_jit
    @jax.value_and_grad
    def run(x__z, y_in, true):
        x_in, z_in = x__z
        init = (0, x_in, y_in, z_in, None)
        if true:
            out = _while_as_scan(cond_fun, body_fun, init, max_steps=3)
        else:
            out = eqxi.while_loop(
                cond_fun, body_fun, init, max_steps=3, kind="checkpointed"
            )
        _, x_out, y_out, z_out, none = out
        assert none is None
        return x_out + y_out + jnp.sum(z_out)

    x_in = jnp.array(1.2)
    y_in = jnp.array(0.7)
    z_in = jnp.array([-5.0, -5.0, -5.0])
    true = run((x_in, z_in), y_in, true=True)
    outs = run((x_in, z_in), y_in, true=False)
    assert tree_allclose(true, outs)


def test_nondifferentiable_body2(capfd):
    def cond_fun(carry):
        return True

    # This function is set up so that (x, y, z) require multiple passes through to
    # propagate which values are perturbed.
    # This function is set up so that w has a cotangent that should be dropped.
    def body_fun(carry):
        x, y, z, w = carry
        w = eqxi.nondifferentiable(w)
        return x + 1, x + y, y + z, w * 2

    @jax.jit
    @jax.grad
    def run(x, y, z, w):
        x, y, z, w = eqxi.while_loop(
            cond_fun, body_fun, (x, y, z, w), max_steps=3, kind="checkpointed"
        )
        return y + w

    capfd.readouterr()
    run(1.0, 1.0, 1.0, 1.0)
    text, _ = capfd.readouterr()
    assert "perturb_val (False, False, False, (True, True, True, False))" in text
    assert (
        "symbolic_zero_gradient (True, True, True, (False, False, True, True))" in text
    )


def test_body_fun_grads(capfd):
    def cond_fun(carry):
        return True

    @eqx.filter_jit
    @jax.grad
    def run(x__y, true):
        x, y = x__y
        # `init` and `body_fun` are deliberately chosen so that `carry[0]` requires a
        # gradient solely for the purpose of propagating that gradient back into `x`.
        # (And in particular, not for propagating it back into `init`.)
        # Thus this test is checking that we get gradients with respect to `body_fun`
        # correctly.
        init = (1.0, y)

        def body_fun(carry):
            a, b = carry
            return a * x, b + 1

        if true:
            final = _while_as_scan(cond_fun, body_fun, init, max_steps=3)
        else:
            final = eqxi.while_loop(
                cond_fun, body_fun, init, max_steps=3, kind="checkpointed"
            )
        return sum(final)

    x__y = (jnp.array(1.0), jnp.array(1.0))

    capfd.readouterr()
    outs = run(x__y, true=False)
    text, _ = capfd.readouterr()
    assert "perturb_val (False, False, False, (True, True))" in text
    assert "symbolic_zero_gradient (True, True, True, (False, False))" in text

    true = run(x__y, true=True)
    assert tree_allclose(true, outs)


def test_trivial_vjp(capfd):
    def cond_fun(carry):
        return True

    def body_fun(carry):
        return carry

    @jax.jit
    @jax.grad
    def run(x):
        a, b = eqxi.while_loop(
            cond_fun, body_fun, (x, 0.0), max_steps=3, kind="checkpointed"
        )
        return b

    capfd.readouterr()
    assert run(1.0) == 0
    text, _ = capfd.readouterr()
    assert "perturb_val (False, False, False, (True, False))" in text
    assert "symbolic_zero_gradient" not in text


def test_buffer_at_set():
    array = jnp.array([0])
    assert eqx.tree_equal(eqxi.buffer_at_set(array, 0, 1), jnp.array([1]))
    assert eqx.tree_equal(eqxi.buffer_at_set(array, 0, 1, pred=False), jnp.array([0]))

    @jax.jit
    def f(pred):
        return eqxi.buffer_at_set(array, 0, 1, pred=pred)

    assert eqx.tree_equal(f(True), jnp.array([1]))
    assert eqx.tree_equal(f(False), jnp.array([0]))

    @jax.jit
    def g(pred):
        def cond(_):
            return True

        def body(carry):
            assert not eqx.is_array(carry)
            return eqxi.buffer_at_set(carry, 0, 1, pred=pred)

        return eqxi.while_loop(
            cond, body, array, max_steps=1, kind="checkpointed", buffers=lambda x: x
        )

    assert eqx.tree_equal(g(True), jnp.array([1]))
    assert eqx.tree_equal(g(False), jnp.array([0]))


def test_unperturbed_output():
    class Carry(eqx.Module):
        is_none: None
        is_bool: Array
        is_int: Array
        is_unperturbed_float: Array
        is_perturbed_float1: Array
        is_perturbed_float2: Array
        is_perturbed_buffer: Array
        is_unperturbed_buffer: Array

    init_carry = Carry(
        None,
        jnp.array(True),
        jnp.array(1),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(2.0),
        jnp.array([3.0]),
        jnp.array([4.0]),
    )

    def cond(_):
        return True

    def body(carry: Carry):
        perturbed_buffer = carry.is_perturbed_buffer.at[0].set(
            carry.is_perturbed_float2
        )
        unperturbed_buffer = carry.is_unperturbed_buffer.at[0].set(
            carry.is_unperturbed_float
        )
        # 1->2
        return Carry(
            carry.is_none,
            carry.is_bool,
            carry.is_int,
            carry.is_unperturbed_float,
            carry.is_perturbed_float2,
            carry.is_perturbed_float2,
            perturbed_buffer,
            unperturbed_buffer,
        )

    # The `eqxi.nondifferentiable` serve as our test assertions.
    def run(init_carry):
        init_carry = eqx.tree_at(
            lambda c: (
                c.is_unperturbed_float,
                c.is_unperturbed_buffer,
                c.is_perturbed_float1,
            ),
            init_carry,
            replace_fn=lax.stop_gradient,
        )

        final_carry = eqxi.while_loop(
            cond,
            body,
            init_carry,
            max_steps=1,
            buffers=lambda x: x.is_perturbed_buffer,
            kind="checkpointed",
        )

        unperturbed = eqxi.nondifferentiable(
            (
                final_carry.is_none,
                final_carry.is_bool,
                final_carry.is_int,
                final_carry.is_unperturbed_float,
                final_carry.is_unperturbed_buffer,
            )
        )
        with pytest.raises(RuntimeError, match="kaboom!"):
            eqxi.nondifferentiable(final_carry.is_perturbed_float1, msg="kaboom!")
        with pytest.raises(RuntimeError, match="kaboom!"):
            eqxi.nondifferentiable(final_carry.is_perturbed_float2, msg="kaboom!")
        with pytest.raises(RuntimeError, match="kaboom!"):
            eqxi.nondifferentiable(final_carry.is_perturbed_buffer, msg="kaboom!")
        with jax.numpy_dtype_promotion("standard"):
            out = sum(jtu.tree_leaves((unperturbed, final_carry)))
        return jnp.reshape(out, ())

    jax.linearize(run, init_carry)
    eqx.filter_grad(run)(init_carry)
