import re

import equinox as eqx
import jax
import jax.numpy as jnp


@eqx.filter_jit
def _run_impl(
    val0: jax.Array,
    meter: eqx.AbstractProgressMeter,
    length: jax.Array,
) -> jax.Array:
    def cond_fn(carry):
        _, _, i = carry
        return i < length

    def body_fn(carry):
        val, state, i = carry
        state = meter.step(state, i / length)
        return val + 1, state, i + 1

    state0 = meter.init()
    val1, state1, _ = jax.lax.while_loop(cond_fn, body_fn, (val0, state0, 0))
    meter.close(state1)
    return val1


def _run(meter: eqx.AbstractProgressMeter, length: jax.Array):
    _run_impl(jnp.array(1), meter, length)


def test_tqdm_progress_meter(capfd):
    solves = [
        lambda: _run(eqx.TqdmProgressMeter(refresh_steps=4), jnp.array(10)),
        lambda: jax.vmap(lambda l: _run(eqx.TqdmProgressMeter(refresh_steps=4), l))(
            jnp.array([6, 10])
        ),
    ]
    for solve in solves:
        capfd.readouterr()
        solve()
        jax.effects_barrier()
        captured = capfd.readouterr()
        err = captured.err.strip()
        assert re.match("0.00%|[ ]+|", err.split("\r", 1)[0])
        assert re.match("100.00%|█+|", err.rsplit("\r", 1)[1])
        assert captured.err.count("\r") == 5
        assert captured.err.count("\n") == 1


def test_text_progress_meter(capfd):
    solves = [
        lambda: _run(eqx.TextProgressMeter(), jnp.array(10)),
        lambda: jax.vmap(lambda l: _run(eqx.TextProgressMeter(), l))(
            jnp.array([10, 8])
        ),
    ]
    for solve in solves:
        capfd.readouterr()
        solve()
        jax.effects_barrier()
        captured = capfd.readouterr()
        out = captured.out.strip()
        assert (
            out
            == "0.00%\n10.00%\n20.00%\n30.00%\n40.00%\n50.00%\n60.00%\n70.00%\n80.00%"
            "\n90.00%\n100.00%"
        )
