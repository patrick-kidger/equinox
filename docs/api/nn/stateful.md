# Stateful operations

These are the tools that underlie stateful operations, like [`equinox.nn.BatchNorm`][] or [`equinox.nn.SpectralNorm`][]. These are fairly unusual layers, so most users will not need this part of the API.

!!! Example

    The [stateful example](../../examples/stateful.ipynb) is a good reference for the typical workflow for stateful layers.

---

::: equinox.nn.make_with_state

## Extra features

Let's explain how this works under the hood. First of all, all stateful layers (`BatchNorm` etc.) include an "index". This is basically just a unique hashable value (used later as a dictionary key), and an initial value for the state:

::: equinox.nn.StateIndex
    selection:
        members:
            - __init__

---

This `State` object that's being passed around is essentially just a dictionary, mapping from `StateIndex`s to PyTrees-of-arrays. Correspondingly this has `.get` and `.set` methods to read and write values to it.

::: equinox.nn.State
    selection:
        members:
            - get
            - set
            - substate
            - update

## Custom stateful layers

Let's use [`equinox.nn.StateIndex`][] to create a custom stateful layer.

```python
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

class Counter(eqx.Module):
    index: eqx.nn.StateIndex

    def __init__(self):
        init_state = jnp.array(0)
        self.index = eqx.nn.StateIndex(init_state)

    def __call__(self, x: Array, state: eqx.nn.State) -> tuple[Array, eqx.nn.State]:
        value = state.get(self.index)
        new_x = x + value
        new_state = state.set(self.index, value + 1)
        return new_x, new_state

counter, state = eqx.nn.make_with_state(Counter)()
x = jnp.array(2.3)

num_calls = state.get(counter.index)
print(f"Called {num_calls} times.")  # 0

_, state = counter(x, state)
num_calls = state.get(counter.index)
print(f"Called {num_calls} times.")  # 1

_, state = counter(x, state)
num_calls = state.get(counter.index)
print(f"Called {num_calls} times.")  # 2
```

## Vmap'd stateful layers

This is an advanced thing to do! Here we'll build on [the ensembling guide](../../../tricks/#ensembling), and see how how we can create vmap'd stateful layers.

This follows on from the previous example, in which we define `Counter`.
```python
import jax.random as jr

class Model(eqx.Module):
    linear: eqx.nn.Linear
    counter: Counter
    v_counter: Counter

    def __init__(self, key):
        # Not-stateful layer
        self.linear = eqx.nn.Linear(2, 2, key=key)
        # Stateful layer.
        self.counter = Counter()
        # Vmap'd stateful layer. (Whose initial state will include a batch dimension.)
        self.v_counter = eqx.filter_vmap(Counter, axis_size=2)()

    def __call__(self, x: Array, state: eqx.nn.State) -> tuple[Array, eqx.nn.State]:
        # This bit happens as normal.
        assert x.shape == (2,)
        x = self.linear(x)
        x, state = self.counter(x, state)

        # For the vmap, we have to restrict our state to just those states we want to
        # vmap, and then update the overall state again afterwards.
        #
        # After all, the state for `self.counter` isn't expecting to be batched, so we
        # have to remove that.
        substate = state.substate(self.v_counter)
        x, substate = eqx.filter_vmap(self.v_counter)(x, substate)
        state = state.update(substate)

        return x, state

key = jr.PRNGKey(0)
model, state = eqx.nn.make_with_state(Model)(key)
x = jnp.array([5.0, -1.0])
model(x, state)
```
