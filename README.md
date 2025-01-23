<h1 align='center'>Equinox</h1>

Equinox is your one-stop [JAX](https://github.com/google/jax) library, for everything you need that isn't already in core JAX:

- neural networks (or more generally any model), with easy-to-use PyTorch-like syntax;
- filtered APIs for transformations;
- useful PyTree manipulation routines;
- advanced features like runtime errors;

and best of all, Equinox isn't a framework: everything you write in Equinox is compatible with anything else in JAX or the ecosystem.

If you're completely new to JAX, then start with this [CNN on MNIST example](https://docs.kidger.site/equinox/examples/mnist/).

_Coming from [Flax](https://github.com/google/flax) or [Haiku](https://github.com/deepmind/haiku)? The main difference is that Equinox (a) offers a lot of advanced features not found in these libraries, like PyTree manipulation or runtime errors; (b) has a simpler way of building models: they're just PyTrees, so they can pass across JIT/grad/etc. boundaries smoothly._

## Installation

Requires Python 3.10+.

```bash
pip install equinox
```

Equinox is also available through a community-supported build on [conda-forge](https://github.com/conda-forge/equinox-feedstock).

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

and are fully compatible with normal JAX operations:

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

**Always useful**  
[jaxtyping](https://github.com/patrick-kidger/jaxtyping): type annotations for shape/dtype of arrays.  

**Deep learning**  
[Optax](https://github.com/deepmind/optax): first-order gradient (SGD, Adam, ...) optimisers.  
[Orbax](https://github.com/google/orbax): checkpointing (async/multi-host/multi-device).  
[Levanter](https://github.com/stanford-crfm/levanter): scalable+reliable training of foundation models (e.g. LLMs).  
[paramax](https://github.com/danielward27/paramax): parameterizations and constraints for PyTrees.

**Scientific computing**  
[Diffrax](https://github.com/patrick-kidger/diffrax): numerical differential equation solvers.  
[Optimistix](https://github.com/patrick-kidger/optimistix): root finding, minimisation, fixed points, and least squares.  
[Lineax](https://github.com/patrick-kidger/lineax): linear solvers.  
[BlackJAX](https://github.com/blackjax-devs/blackjax): probabilistic+Bayesian sampling.  
[sympy2jax](https://github.com/patrick-kidger/sympy2jax): SymPy<->JAX conversion; train symbolic expressions via gradient descent.  
[PySR](https://github.com/milesCranmer/PySR): symbolic regression. (Non-JAX honourable mention!)  

**Awesome JAX**  
[Awesome JAX](https://github.com/n2cholas/awesome-jax): a longer list of other JAX projects.  
