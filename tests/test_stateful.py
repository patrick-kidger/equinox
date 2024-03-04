import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest


def test_delete_init_state():
    model = eqx.nn.BatchNorm(3, "batch")
    eqx.nn.State(model)
    model2 = eqx.nn.delete_init_state(model)

    eqx.nn.State(model)
    with pytest.raises(ValueError):
        eqx.nn.State(model2)

    leaves = [x for x in jtu.tree_leaves(model) if eqx.is_array(x)]
    leaves2 = [x for x in jtu.tree_leaves(model2) if eqx.is_array(x)]
    assert len(leaves) == len(leaves2) + 3


def test_double_state():
    # From https://github.com/patrick-kidger/equinox/issues/450#issuecomment-1714501666

    class Counter(eqx.Module):
        index: eqx.nn.StateIndex

        def __init__(self):
            init_state = jnp.array(0)
            self.index = eqx.nn.StateIndex(init_state)

        def __call__(self, x, state):
            value = state.get(self.index)
            new_x = x + value
            new_state = state.set(self.index, value + 1)
            return new_x, new_state

    class Model(eqx.Module):
        linear: eqx.nn.Linear
        counter: Counter
        v_counter: Counter

        def __init__(self, key):
            # Not-stateful layer
            self.linear = eqx.nn.Linear(2, 2, key=key)
            # Stateful layer.
            self.counter = Counter()
            # Vmap'd stateful layer. (Whose initial state will include a batch
            # dimension.)
            self.v_counter = eqx.filter_vmap(Counter, axis_size=2)()

        def __call__(self, x, state):
            assert x.shape == (2,)
            x = self.linear(x)
            x, state = self.counter(x, state)
            substate = state.substate(self.v_counter)
            x, substate = eqx.filter_vmap(self.v_counter)(x, substate)
            state = state.update(substate)
            return x, state

    key = jr.PRNGKey(0)
    model, state = eqx.nn.make_with_state(Model)(key)
    x = jnp.array([5.0, -1.0])
    model(x, state)

    @jax.jit
    def make_state(key):
        _, state = eqx.nn.make_with_state(Model)(key)
        return state

    new_state = make_state(jr.PRNGKey(1))
    model(x, new_state)
