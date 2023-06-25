# Getting started

Equinox is a JAX library for parameterised functions (e.g. neural networks) offering:

- a PyTorch-like API...
- ...that's fully compatible with *native* JAX transformations...
- ...with no new concepts you have to learn.

If you're completely new to JAX, then start with this [CNN on MNIST example](https://docs.kidger.site/equinox/examples/mnist/).  
If you're already familiar with JAX, then the main idea is to represent parameterised functions (such as neural networks) as PyTrees, so that they can pass across JIT/grad/etc. boundaries smoothly.

The elegance of Equinox is its selling point in a world that already has [Haiku](https://github.com/deepmind/dm-haiku), [Flax](https://github.com/google/flax) and so on.

_In other words, why should you care? Because Equinox is really simple to learn, and really simple to use._

## Installation

```bash
pip install equinox
```

Requires Python 3.9+ and JAX 0.4.11+.

## Quick example

Models are defined using PyTorch-like syntax:

```python
import equinox as eqx
import jax

class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))

    def __call__(self, x):
        return self.weight @ x + self.bias
```

and may be used alongside normal JAX operations:

```python
@jax.jit
@jax.grad
def loss_fn(model, x, y):
    pred_y = jax.vmap(model)(x)
    return jax.numpy.mean((y - pred_y) ** 2)

batch_size, in_size, out_size = 32, 2, 3
model = Linear(in_size, out_size, key=jax.random.PRNGKey(0))
x = jax.numpy.zeros((batch_size, in_size))
y = jax.numpy.zeros((batch_size, out_size))
grads = loss_fn(model, x, y)
```

There's no magic behind the scenes. All `eqx.Module` does is register your class as a PyTree. From that point onwards, JAX already knows how to work with PyTrees.

## Next steps

If this quick start has got you interested, then have a read of [All of Equinox](./all-of-equinox.md), which introduces you to basically everything in Equinox.

## Citation

--8<-- ".citation.md"

## See also: other libraries in the JAX ecosystem

[Optax](https://github.com/deepmind/optax): first-order gradient (SGD, Adam, ...) optimisers.

[Diffrax](https://github.com/patrick-kidger/diffrax): numerical differential equation solvers.

[Lineax](https://github.com/google/lineax): linear solvers and linear least squares.

[jaxtyping](https://github.com/google/jaxtyping): type annotations for shape/dtype of arrays.

[Eqxvision](https://github.com/paganpasta/eqxvision): computer vision models.

[sympy2jax](https://github.com/google/sympy2jax): SymPy<->JAX conversion; train symbolic expressions via gradient descent.
