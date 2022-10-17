"""Benchmarks the effect of `equinox.internal.noinline` and
`equinox.internal.bounded_while_loop`.
"""
import functools as ft
import sys
import timeit

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx


# inline vs no-inline are done through separate executions of the script, to
# be sure that there's no jaxpr caching making the second compilation faster.
_, inline, checkpoint, grad = sys.argv
if inline == "inline":
    noinline = False
elif inline == "noinline":
    noinline = True
else:
    raise ValueError
if checkpoint == "none":
    adjoint = dfx.DirectAdjoint()
elif checkpoint == "recursive":
    adjoint = dfx.RecursiveCheckpointAdjoint()
else:
    raise ValueError
if grad == "grad":
    grad_decorator = jax.grad
elif grad == "nograd":
    grad_decorator = lambda x: x
else:
    raise ValueError


def _weight(in_, out, key):
    return [[w_ij for w_ij in w_i] for w_i in jr.normal(key, (out, in_))]


class VectorField(eqx.Module):
    weights: list

    def __init__(self, in_, out, width, depth, *, key):
        keys = jr.split(key, depth + 1)
        self.weights = [_weight(in_, width, keys[0])]
        for i in range(1, depth):
            self.weights.append(_weight(width, width, keys[i]))
        self.weights.append(_weight(width, out, keys[depth]))

    def __call__(self, t, y, args):
        # Inefficient computation graph to make a toy example more expensive.
        y = eqx.internal.announce_jaxpr(y, "vf")
        y = [y_i for y_i in y]
        for w in self.weights:
            y = [sum(w_ij * y_j for w_ij, y_j in zip(w_i, y)) for w_i in w]
        return jnp.stack(y)


vf = VectorField(1, 1, 16, 2, key=jr.PRNGKey(0))
if noinline:
    vf = eqx.internal.noinline(vf)
term = dfx.ODETerm(vf)
solver = dfx.Dopri8(scan_stages=False)
stepsize_controller = dfx.PIDController(rtol=1e-3, atol=1e-6)
t0 = 0
t1 = 1
dt0 = 0.01


@jax.jit
@grad_decorator
def solve(y0):
    y0 = eqx.internal.announce_jaxpr(y0, name="y0")
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
        max_steps=16**2,
    )
    return jnp.sum(sol.ys)


solve_ = ft.partial(solve, jnp.array([1.0]))
print("Compile+run time", timeit.timeit(solve_, number=1))
print("Run time", timeit.timeit(solve_, number=1))
