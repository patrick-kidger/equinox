import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import pytest


def _make_runner(progress_meter, num_steps):
    """Drive `progress_meter` through `num_steps` synthetic steps, sweeping
    progress linearly from `1/num_steps` to `1`. Returns a callable so the
    caller can wrap with `jit`/`vmap`/`grad` as needed.
    """

    def run(x):
        state = progress_meter.init()

        def body(state, i):
            progress = (i + 1) / num_steps
            return progress_meter.step(state, progress), None

        state, _ = jax.lax.scan(body, state, jnp.arange(num_steps))
        progress_meter.close(state)
        return x

    return run


def test_no_progress_meter(capfd):
    capfd.readouterr()
    run = _make_runner(eqxi.NoProgressMeter(), 5)
    run(1.0)
    jax.effects_barrier()
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_text_progress_meter(capfd):
    expected = "%\n".join(f"{x:.2f}" for x in jnp.linspace(0, 100, num=11)) + "%\n"

    run = _make_runner(eqxi.TextProgressMeter(minimum_increase=0.0999), 10)

    capfd.readouterr()
    run(1.0)
    jax.effects_barrier()
    assert capfd.readouterr().out == expected

    capfd.readouterr()
    jax.jit(run)(1.0)
    jax.effects_barrier()
    assert capfd.readouterr().out == expected

    capfd.readouterr()
    jax.vmap(run)(jnp.arange(3.0))
    jax.effects_barrier()
    assert capfd.readouterr().out == expected

    capfd.readouterr()
    jax.jit(jax.vmap(run))(jnp.arange(3.0))
    jax.effects_barrier()
    assert capfd.readouterr().out == expected


def test_tqdm_progress_meter(capfd):
    pytest.importorskip("tqdm")

    run = _make_runner(eqxi.TqdmProgressMeter(refresh_steps=5), 50)

    capfd.readouterr()
    jax.jit(run)(1.0)
    jax.effects_barrier()
    err = capfd.readouterr().err
    assert err.count("\r") >= 1
    final_frame = err.rsplit("\r", 1)[-1]
    assert "100.00%" in final_frame
    assert "█" in final_frame


def test_tqdm_progress_meter_vmap(capfd):
    pytest.importorskip("tqdm")

    run = _make_runner(eqxi.TqdmProgressMeter(refresh_steps=5), 50)

    capfd.readouterr()
    jax.jit(jax.vmap(run))(jnp.arange(3.0))
    jax.effects_barrier()
    err = capfd.readouterr().err
    assert err.count("\r") >= 1
    assert "100.00%" in err.rsplit("\r", 1)[-1]


def _grad_smoke(progress_meter: eqxi.AbstractProgressMeter, capfd):
    num_steps = 10

    def fwd(p):
        state = progress_meter.init()

        def body(state, i):
            progress = (i + 1) / num_steps
            return progress_meter.step(state, progress), None

        state, _ = jax.lax.scan(body, state, jnp.arange(num_steps))
        progress_meter.close(state)
        return p * p

    capfd.readouterr()
    g = jax.grad(fwd)(jnp.array(1.0))
    jax.effects_barrier()
    assert g == 2.0

    if isinstance(progress_meter, eqxi.TextProgressMeter):
        out = capfd.readouterr().out
        expected = "%\n".join(f"{x:.2f}" for x in jnp.linspace(0, 100, num=11)) + "%\n"
        assert out == expected

    capfd.readouterr()
    jax.jit(jax.grad(fwd))(jnp.array(1.0))
    jax.effects_barrier()


def test_grad_text_progress_meter(capfd):
    _grad_smoke(eqxi.TextProgressMeter(minimum_increase=0.0999), capfd)


def test_grad_tqdm_progress_meter(capfd):
    pytest.importorskip("tqdm")
    _grad_smoke(eqxi.TqdmProgressMeter(refresh_steps=5), capfd)
