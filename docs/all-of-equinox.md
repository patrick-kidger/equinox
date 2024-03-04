# All of Equinox

Equinox is a small and easy to understand library. So as the title suggests, this page tells you essentially everything you need to know to use Equinox.

## 1. Models as PyTrees

!!! info "What's a PyTree?"

    [PyTrees](https://jax.readthedocs.io/en/latest/pytrees.html) are what JAX calls nested collections of tuples, lists, and dicts. (And any custom-registered PyTree nodes.) The "leaves" of the tree can be anything at all: JAX/NumPy arrays, floats, functions, etc. Most JAX operations will accept either (a) arbitrary PyTrees; (b) PyTrees with just JAX/NumPy arrays as the leaves; (c) PyTrees without any JAX/NumPy arrays as the leaves.

As we saw on the [Getting Started](./index.md) page, Equinox offers the ability to represents models as PyTrees. This is one of Equinox's main features.

Once we've done so, we'll be able to JIT/grad/etc. with respect to the model. For example, using a few built-in layers by way of demonstration, here's a small neural network:

```python
import equinox as eqx
import jax

class NeuralNetwork(eqx.Module):
    layers: list
    extra_bias: jax.Array

    def __init__(self, key):
        key1, key2, key3 = jax.random.split(key, 3)
        # These contain trainable parameters.
        self.layers = [eqx.nn.Linear(2, 8, key=key1),
                       eqx.nn.Linear(8, 8, key=key2),
                       eqx.nn.Linear(8, 2, key=key3)]
        # This is also a trainable parameter.
        self.extra_bias = jax.numpy.ones(2)

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x) + self.extra_bias

@jax.jit  # compile this function to make it run fast.
@jax.grad  # differentiate all floating-point arrays in `model`.
def loss(model, x, y):
    pred_y = jax.vmap(model)(x)  # vectorise the model over a batch of data
    return jax.numpy.mean((y - pred_y) ** 2)  # L2 loss

x_key, y_key, model_key = jax.random.split(jax.random.PRNGKey(0), 3)
# Example data
x = jax.random.normal(x_key, (100, 2))
y = jax.random.normal(y_key, (100, 2))
model = NeuralNetwork(model_key)
# Compute gradients
grads = loss(model, x, y)
# Perform gradient descent
learning_rate = 0.1
new_model = jax.tree_util.tree_map(lambda m, g: m - learning_rate * g, model, grads)
```

In this example, `model = NeuralNetwork(...)` is the overall PyTree. Nested within that is `model.layers` and `model.extra_bias`. The former is also a PyTree, containing three `eqx.nn.Linear` layers at `model.layers[0]`, `model.layers[1]`, and `model.layers[2]`. Each of these are also PyTrees, containing their weights and biases, e.g. `model.layers[0].weight`.

## 2. Filtering

In the previous example, all of the leaves were JAX arrays. This made things simple, because `jax.jit` and `jax.grad`-decorated functions require that all of their inputs are PyTrees of arrays.

Equinox goes further, and supports using arbitrary Python objects for its leaves. For example, we might like to make our activation function part of the PyTree (rather than just hardcoding it as above). The activation function will just be some arbitrary Python function, and this isn't an array. Another common example is having a `bool`-ean flag in your model, which specifies whether to enable some extra piece of behaviour.

To support this, then Equinox offers *filtering*, as follows.

**Create a model**

Start off by creating a model just like normal, now with some arbitrary Python objects as part of its PyTree structure. In this case, we have `jax.nn.relu`, which is a Python function.

```python
import equinox as eqx
import functools as ft
import jax

class NeuralNetwork2(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2 = jax.random.split(key)
        self.layers = [eqx.nn.Linear(2, 8, key=key1),
                       jax.nn.relu,
                       eqx.nn.Linear(8, 2, key=key2)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

x_key, y_key, model_key = jax.random.split(jax.random.PRNGKey(0), 3)
x, y = jax.random.normal(x_key, (100, 2)), jax.random.normal(y_key, (100, 2))
model = NeuralNetwork2(model_key)
```

**Option 1: use `eqx.{partition,combine}`**

```python
@ft.partial(jax.jit, static_argnums=1)  # `static` must be a PyTree of non-arrays.
@jax.grad  # differentiates with respect to `params`, as it is the first argument
def loss(params, static, x, y):
    model = eqx.combine(params, static)
    pred_y = jax.vmap(model)(x)
    return jax.numpy.mean((y - pred_y) ** 2)

params, static = eqx.partition(model, eqx.is_array)
loss(params, static, x, y)
```

Here, we split our model PyTree into two pieces. `params` and `static` are both instances of `NeuralNetwork2`. `params` keeps just the leaves that are arrays; `static` keeps everything else. Then `combine` merges the two PyTrees back together after crossing the `jax.jit` and `jax.grad` API boundaries.

The choice of `eqx.is_array` is a *filter function*: a boolean function specifying whether each leaf should go into `params` or into `static`. In this case very simply `eqx.is_array(x)` returns `True` for JAX and NumPy arrays, and `False` for everything else.

**Option 2: use filtered transformations**

```python
@eqx.filter_jit
@eqx.filter_grad
def loss(model, x, y):
    pred_y = jax.vmap(model)(x)
    return jax.numpy.mean((y - pred_y) ** 2)

loss(model, x, y)
```

As a convenience, `eqx.filter_jit` and `eqx.filter_grad` wrap filtering and transformation together. It turns out to be really common to only need to filter around JAX transformations.

If your models only use JAX arrays, then `eqx.filter_{jit,grad,...}` will do exactly the same as `jax.{jit,grad,...}`. So if you just want to keep things simple, it is safe to just always use `eqx.filter_{jit,grad,...}`.

Both approaches are equally valid. Some people prefer the shorter syntax of the filtered transformations. Some people prefer to explicitly see the `jax.{jit,grad,...}` operations directly.

## 3. PyTree manipulation routines.

Equinox clearly places a heavy focus on PyTrees! As such, it's quite common to need to perform operations on PyTrees. Whilst many common operations are already provided by JAX (for example, `jax.tree_util.tree_map` will apply an operation to every leaf of a PyTree), Equinox additionally offers some extra features. For example, `eqx.tree_at` mutates a particular leaf or leaves of a PyTree.

## 4. Advanced goodies.

Finally, Equinox offers a number of more advanced goodies, like serialisation, debugging tools, and runtime errors. We won't discuss them here, but check out the API reference on the left.

## 5. Summary

**Equinox integrates smoothly with JAX**

Equinox introduces a powerful yet straightforward way to build neural networks, without introducing lots of new notions or tieing you into a framework. Indeed Equinox is a *library*, not a *framework* -- this means that anything you write in Equinox is fully compatible with anything else in the JAX ecosystem.

Equinox is all just regular JAX: PyTrees and transformations. Together, these two pieces allow us to specify complex models in JAX-friendly ways.

**API reference**

- For building models: [`equinox.Module`][].
- Prebuilt neural network layers: [`equinox.nn.Linear`][], [`equinox.nn.Conv2d`][], etc.
- Filtered transformations: [`equinox.filter_jit`][] etc.
- Tools for PyTree manipulation: [`equinox.partition`][], etc.
- Advanced goodies: serialisation, debugging tools, runtime errors, etc. 

See the API reference on the left.

**Next steps**

And that's it! That's pretty much everything you need to know about Equinox. Everything you've seen so far should be enough to get started with using the library. Also see the [Train RNN](./examples/train_rnn.ipynb) example for a fully worked example.
