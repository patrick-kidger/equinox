from typing import Union

import jax.numpy as jnp
import pytest
from helpers import shaped_allclose

import equinox as eqx


@pytest.mark.parametrize("call", [False, True])
@pytest.mark.parametrize("outer", [False, True])
def test_methods(call, outer):
    num_traces = 0

    class M(eqx.Module):
        increment: Union[int, jnp.ndarray]

        if call:

            def __call__(self, x):
                nonlocal num_traces
                num_traces += 1
                return x + self.increment

            if not outer:
                __call__ = eqx.filter_pmap(__call__)
        else:

            def method(self, x):
                nonlocal num_traces
                num_traces += 1
                return x + self.increment

            if not outer:
                method = eqx.filter_pmap(method)

    y = jnp.ndarray([1, 2])

    def run(_m):
        if call:
            if outer:
                return eqx.filter_pmap(_m)(y)
            else:
                return _m(y)
        else:
            if outer:
                return eqx.filter_pmap(_m.method)(y)
            else:
                return _m.method(y)

    m = M(1)
    assert shaped_allclose(run(m), jnp.array([2, 3]))
    assert shaped_allclose(run(m), jnp.array([2, 3]))
    assert num_traces == 1
    n = M(2)
    assert shaped_allclose(run(n), jnp.array([3, 4]))
    assert shaped_allclose(run(n), jnp.array([3, 4]))
    assert num_traces == 2
    o = M(jnp.ndarray([5, 6]))
    p = M(jnp.ndarray([6, 7]))
    assert shaped_allclose(run(o), jnp.array([6, 8]))
    assert shaped_allclose(run(p), jnp.array([7, 9]))
    assert num_traces == 3
