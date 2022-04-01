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
which after a little thought should make sense: Optax can only optimise JAX arrays. It's not meaningful to ask Optax to optimsie whichever other arbitrary Python objects may be a part of your model. (e.g. the activation function of an `eqx.nn.MLP`).

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
in which the same object is saved multiple times in the model. After making some gradient updates you'll find that `self.linear1` and `self.linear2` are now different.

Recall that in Equinox, models are PyTrees. Meanwhile, JAX treats all PyTrees as *trees*: that is, the same object does not appear more in the tree than once. (If it did, then it would be a *directed acyclic graph* instead.) If JAX ever encounters the same object multiple times then it will unwittingly make independent copies of the object whenever it transforms the overall PyTree.

The resolution is simple: just don't store the same object in multiple places in the PyTree.

## I cannot feed higher-order tensors (with batch dimensions or others) into my layers 

It is quite common for deep learning frameworks like PyTorch to support higher-order tensors as inputs.
For example, if `x` is a vector of shape `(..., d_in)`, the following PyTorch code

```python
import torch
linear = torch.nn.Linear(d_in, d_out)
y = linear(x)
```

is legitimate and produces a vector of shape `(..., d_out)` where the left-most dimensions of the input are preserved.
In contrast, running the similar Equinox code

```python
import jax
import equinox as eqx
key = jax.random.PRNGKey(seed=0)
linear = eqx.nn.Linear(d_in, d_out, key=key)
# will fail with a TypeError
y = linear(x)
```

will fail. 
In PyTorch, this code is possible due to implicit use of broadcasting. 
Equinox, on the other hand, is a bit stricter here.
Luckily, the functional style of JAX makes it straight-forward to add support for higher-order tensors
by transforming the linear layer call with [`jax.vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax.vmap).
Making use of `vmap`, the rewritten code reads

```python
import jax
import equinox as eqx
key = jax.random.PRNGKey(seed=0)
linear = eqx.nn.Linear(d_in, d_out, key=key)
y = jax.vmap(linear, in_axes=0, out_axes=0)(x)
```
