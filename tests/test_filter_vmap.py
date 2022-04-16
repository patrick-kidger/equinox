from typing import Union

import jax.numpy as jnp
import pytest

import equinox as eqx


@pytest.mark.parametrize("call", [False, True])
@pytest.mark.parametrize("outer", [False, True])
def test_methods(call, outer):
    class M(eqx.Module):
        increment: Union[int, jnp.ndarray]

        if call:

            def __call__(self, x):
                return x + self.increment

            if not outer:
                __call__ = eqx.filter_vmap(__call__)
        else:

            def method(self, x):
                return x + self.increment

            if not outer:
                method = eqx.filter_vmap(method)

    m = M(5)
    y = jnp.ndarray(1.0)

    if call:
        if outer:
            assert eqx.filter_vmap(m)(y) == 1
        else:
            assert m(y) == 1
    else:
        if outer:
            assert eqx.filter_vmap(m.method)(y) == 1
        else:
            assert m.method(y) == 1
