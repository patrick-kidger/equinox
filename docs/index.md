# Getting started

Equinox is a JAX library based around a simple idea: **represent parameterised functions (such as neural networks) as PyTrees.**

In doing so:

- We get a PyTorch-like API...
- ...that's fully compatible with *native* JAX transformations...
- ...with no new concepts you have to learn. (It's all just PyTrees.)

The elegance of Equinox is its selling point in a world that already has [Haiku](https://github.com/deepmind/dm-haiku), [Flax](https://github.com/google/flax) and so on.

_(In other words, why should you care? Because Equinox is really simple to learn, and really simple to use.)_

## Installation

```bash
pip install equinox
```

Requires Python 3.7+ and JAX 0.2.18+.

## Quick example

Models are defined using PyTorch-like syntax:

```python
import equinox as eqx
import jax

class Linear(eqx.Module):
    weight: jax.numpy.ndarray
    bias: jax.numpy.ndarray

    def __init__(self, in_size, out_size, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))

    def __call__(self, x):
        return self.weight @ x + self.bias
```

and fully compatible with normal JAX operations:

```python
@jax.jit
@jax.grad
def loss_fn(model, x, y):
    pred_y = jax.vmap(model)(y)
    return jnp.mean((y - pred_y) ** 2)

batch_size, in_size, out_size = 32, 2, 3
model = Linear(in_size, out_size, key=jrandom.PRNGKey(0))
x = jnp.zeros((batch_size, in_size))
y = jnp.zeros((batch_size, out_size))
grads = loss_fn(model, x, y)
```

Finally, there's no magic behind the scenes. All `eqx.Module` does is register your class as a PyTree. From that point onwards, JAX already knows how to work with PyTrees.

## Next steps

If this quick start has got you interested, then have a read of [All of Equinox](./all-of-equinox.md), which introduces you to basically everything in Equinox. (Doesn't take very long! Equinox is simple because everything is a PyTree.)

## Citation

--8<-- ".citation.md"
