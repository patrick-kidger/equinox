# FAQ

## Optax is throwing an error.

Probably you're writing code that looks like
```python
optim = optax.adam(learning_rate)
optim.init(model)
```
and getting an error that looks like
```
TypeError: zeros_like requires ndarray or scalar arguments, got <class 'jax._src.custom_derivatives.custom_jvp'> at position 0.
```

This can be fixed by doing
```python
optim.init(eqx.filter(model, eqx.is_array))
```
which after a little thought should make sense: Optax can only optimise JAX arrays. It's not meaningful to ask Optax to optimise whichever other arbitrary Python objects may be a part of your model. (e.g. the activation function of an `eqx.nn.MLP`).

## A module saved in two places has become two independent copies.

Probably you're doing something like
```python
class Module(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(...):
        shared_linear = eqx.nn.Linear(...)
        self.linear1 = shared_linear
        self.linear2 = shared_linear
```
in which the same object is saved multiple times in the model.

Don't do this!

After making some gradient updates you'll find that `self.linear1` and `self.linear2` are now different.

Recall that in Equinox, models are PyTrees. Meanwhile, JAX treats all PyTrees as *trees*: that is, the same object does not appear more in the tree than once. (If it did, then it would be a *directed acyclic graph* instead.) If JAX ever encounters the same object multiple times then it will unwittingly make independent copies of the object whenever it transforms the overall PyTree.

The resolution is simple: just don't store the same object in multiple places in the PyTree.

## How do I input higher-order tensors (e.g. with batch dimensions) into my model?

Use [`jax.vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax.vmap). This maps arbitrary JAX operations -- including any Equinox module -- over additional dimensions (such as batch dimensions).

For example if `x` is an array/tensor of shape `(batch_size, input_size)`, then the following PyTorch code:

```python
import torch
linear = torch.nn.Linear(input_size, output_size)

y = linear(x)
```

is equivalent to the following Equinox code:
```python
import jax
import equinox as eqx
key = jax.random.PRNGKey(seed=0)
linear = eqx.nn.Linear(input_size, output_size, key=key)

y = jax.vmap(linear)(x)
```

## TypeError: not a valid JAX type.

You might be getting an error like
```
TypeError: Argument '<function ...>' of type <class 'function'> is not a valid JAX type.
```
Example:
```python3
import jax
import equinox as eqx

def loss_fn(model, x, y):
    return (model(x) - y) ** 2

model = eqx.nn.Lambda(lambda x: x)

try:
    jax.jit(loss_fn)(model, 0, 0) # error
except TypeError as e:
    print(e)

eqx.filter_jit(loss_fn)(model, 0, 0) # ok
```

Instead of [`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html), use [`equinox.filter_jit`](https://docs.kidger.site/equinox/api/filtering/filtered-transformations/#equinox.filter_jit). Likewise for [other transformations](https://docs.kidger.site/equinox/api/filtering/filtered-transformations/).
