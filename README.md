<h1 align='center'>Equinox</h1>

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

Requires Python 3.9+ and JAX 0.4.13+.

## Documentation

Available at [https://docs.kidger.site/equinox](https://docs.kidger.site/equinox).

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


## See also: other libraries in the JAX ecosystem

[Optax](https://github.com/deepmind/optax): first-order gradient (SGD, Adam, ...) optimisers.

[Diffrax](https://github.com/patrick-kidger/diffrax): numerical differential equation solvers.

[Lineax](https://github.com/google/lineax): linear solvers and linear least squares.

[jaxtyping](https://github.com/google/jaxtyping): type annotations for shape/dtype of arrays.

[Eqxvision](https://github.com/paganpasta/eqxvision): computer vision models.

[sympy2jax](https://github.com/google/sympy2jax): SymPy<->JAX conversion; train symbolic expressions via gradient descent.
