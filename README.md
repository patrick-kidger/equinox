<h1 align='center'>Equinox</h1>
<h2 align='center'>Filtered JIT/grad transformations in JAX => neural networks</h2>

Equinox brings more power to your JAX workflow.
- Filtered JIT and grad;
- Specifying models as callable PyTrees;
- Integrates smoothly with JAX: Equinox is a library, not a framework.

### Installation

```
pip install git+https://github.com/patrick-kidger/equinox.git
```
Requires JAX 0.2.18+.

### Filtered JIT and filtered grad

Equinox offers two main functions: `jitf` and `gradf`. These are thin wrappers around `jax.jit` and `jax.grad`. Instead of specifying e.g. `jit(static_argnums=...)` or `grad(argnums=...)`, you instead specify `jitf(filter_fn=..., filter_tree=...)` or `gradf(filter_fn=..., filter_tree=...)`. These look at the input PyTrees and specify True/False whether to JIT/differentiate each leaf of each PyTree. This gives a finer resolution than arguments.

This simple approach offers a powerful fine-grained way to control JIT and autodifferentiation. For example:
- Annotate some PyTree of parameters of a model as being frozen, and only train the others.
- Build a complex model as a PyTree, mixing JAX arrays with boolean flags with arbitrary Python objects -- and then simply specify that on the forward pass, all JAX arrays should be JIT traced whilst all boolean flags and Python objects should be static arguments.

### Modules

Equinox offers a familiar class-based syntax to build *callable PyTrees*. Because they're callable, they can be used to represent the forward pass of a model. Because they're PyTrees, they work directly with all of `jax.jit`, `jax.grad`, `jitf` and `gradf` in exactly the way you expect.

As such it is both functional programming *and* a familiar PyTorch-inspired `Module` syntax for building models.

*(Unlike for example Haiku's class-to-functional transform, or any additional complexity to learn like Objax's custom collections of parameters. Equinox's `Module` implementation is only 60 lines long!)*

### Integrates smoothly with JAX

It was a key design goal that Equinox be a library and not a framework. It introduces zero new abstractions. `jitf` and `gradf` are just thing wrappers around `jax.jit` and `jax.grad`, whilst its Modules are just a neat syntax to construct PyTrees.

## Examples

- [`train_mlp.py`](./examples/train_mlp.py) gives a short example that introduces `jitf` and `gradf`. These will be used to filter out the parameters of an MLP and train them.
- [`frozen_layer.py`](./examples/frozen_layer.py) demonstrates how this approach really shines: some of the parameters will be trained, some of them will be frozen, but *all* of them will be efficiently JIT-traced.
- [`build_model.py`](./examples/build_model.py) constructs an MLP from scratch using Modules. We use a class-based syntax that is *simultaneously* functional programming. And so it works directly with JIT/grad without any complexities or edge cases.

As a quick example:
```python
import equinox as eqx, functools as ft, jax.numpy as jnp, jax.random as jrandom, typing

class LinearOrIdentity(eqx.Module):
    weight: typing.Any  # we want to differentiate and JIT-trace this
    flag: bool  # we want to JIT-static this

    def __init__(self, in_features, out_features, flag, key):
        self.weight = jrandom.normal(key, (out_features, in_features))
        self.flag = flag

    def __call__(self, x):
        if self.flag:
            return x
        return self.weight @ x

# Differentiate and trace every floating-point array. Everything else is static/undifferentiated.
# `filter_fn` is just a boolean function that determine whether to jit/grad each leaf of the PyTree.
@ft.partial(eqx.jitf, filter_fn=eqx.is_inexact_array)
@ft.partial(eqx.gradf, filter_fn=eqx.is_inexact_array)
def loss(model, x, y):
    pred_y = jax.vmap(model)(x)
    return jnp.mean((y - pred_y) ** 2)

key = jrandom.PRNGKey(0)
model = LinearOrIdentity(2, 3, flag=True, key=key)  # is a PyTree with elements `weight` and `flag`.
x, y = ... # get data
loss(model, x, y)
```

## API
