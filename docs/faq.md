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
optim.init(eqx.filter(model, eqx.is_inexact_array))
```
which after a little thought should make sense: Optax can only optimise floating-point JAX arrays. It's not meaningful to ask Optax to optimise whichever other arbitrary Python objects may be a part of your model. (e.g. the activation function of an `eqx.nn.MLP`).

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

You can check for whether you have duplicate nodes by using the [`equinox.tree_check`][] function.

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

## My model is slow to train!

Most autodifferentiable programs will have a "numerical bit" (e.g. a training step for your model) and a "normal programming bit" (e.g. saving models to disk).

JAX makes this difference explicit. All the numerical work should go inside a single big JIT region, within which all numerical operations are compiled. For example:

```python    
@eqx.filter_jit
def make_step(model, x, y):
    # Inside JIT region
    grads = compute_loss(model, x, y)
    model = stochastic_gradient_descent(model, grads)
    return model

@eqx.filter_grad
def compute_loss(model, x, y):
    # Still inside JIT region
    ...
    
def stochastic_gradient_descent(model, grads):
    # Also inside JIT region
    ...

for step, (x, y) in zip(range(number_of_steps), dataloader):
    model = make_step(model, x, y)
    # Outside JIT region
```

A common mistake would be to put `jax.jit`/`eqx.filter_jit` on the `compute_loss` function instead of the overall `make_step` function. This would mean doing numerical work (`stochastic_gradient_descent`) outside of JIT. That would run, but would be unnecessarily slow.

See [the RNN example](https://docs.kidger.site/equinox/examples/train_rnn/) as an example of good practice. The whole `make_step` function is JIT compiled in one go.

## My model is slow to compile!

95% of the time, it's because you've done something like this:
```python
@jax.jit
def f(x):
    for i in range(100):
        x = my_complicated_function(x)
    return x
```
When JAX traces through this, it can't see the `for` loop. (`jax.jit` replaces the `x` argument with a tracer object that records everything that happens to it -- and this effectively unrolls the loop.) As a result you'll get 100 independent copies of `my_complicated_function`, which all get compiled separately.

In this case, a `jax.lax.scan` is probably what you want. Likewise it's usually also preferable to rewrite even simple stuff like
```python
x2 = f(x1)
x3 = f(x2)
```
as a little length-2 scan.

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
    return ((model(x) - y) ** 2).mean()

model = eqx.nn.Lambda(lambda x: x)
model = eqx.nn.MLP(2, 2, 2, 2, key=jax.random.PRNGKey(0))

x = jax.numpy.arange(2)
y = x * x

try:
    jax.jit(loss_fn)(model, x, y) # error
except TypeError as e:
    print(e)

eqx.filter_jit(loss_fn)(model, x, y) # ok
```

This error happens because a model, when treated as a PyTree, may have leaves that are not JAX types (such as functions). It only makes sense to trace arrays. Filtering is used to handle this.

Instead of [`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html), use [`equinox.filter_jit`][]. Likewise for [other transformations](https://docs.kidger.site/equinox/api/filtering/transformations).

## How do I mark an array as being non-trainable? (Like PyTorch's buffers?)

This can be done by using `jax.lax.stop_gradient`:
```python
class Model(eqx.Module):
    buffer: Array
    param: Array

    def __call__(self, x):
        return self.param * x + jax.lax.stop_gradient(self.buffer)
```
