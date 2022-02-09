<h1 align='center'>Equinox</h1>

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

## Documentation

Available at [https://docs.kidger.site/equinox](https://docs.kidger.site/equinox).

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
    pred_y = jax.vmap(model)(x)
    return jax.numpy.mean((y - pred_y) ** 2)

batch_size, in_size, out_size = 32, 2, 3
model = Linear(in_size, out_size, key=jax.random.PRNGKey(0))
x = jax.numpy.zeros((batch_size, in_size))
y = jax.numpy.zeros((batch_size, out_size))
grads = loss_fn(model, x, y)
```

Finally, there's no magic behind the scenes. All `eqx.Module` does is register your class as a PyTree. From that point onwards, JAX already knows how to work with PyTrees.

## Citation

If you found this library to be useful in academic work, then please cite: ([arXiv link](https://arxiv.org/abs/2111.00254))

```bibtex
@article{kidger2021equinox,
    author={Patrick Kidger and Cristian Garcia},
    title={{E}quinox: neural networks in {JAX} via callable {P}y{T}rees and filtered transformations},
    year={2021},
    journal={Differentiable Programming workshop at Neural Information Processing Systems 2021}
}
```

(Also consider starring the project on GitHub.)
