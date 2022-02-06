# All of Equinox

Equinox is a very small and easy to understand library. (Because it uses JAX abstractions like PyTrees, it doesn't need any real complexity.)

So as the title suggests, this page tells you essentially everything you need to know to use Equinox.

## Parameterised functions as PyTrees

As we saw on the [Getting Started](./index.md) page, Equinox represents parameterised functions as PyTrees.

!!! example

    A neural network is a function parameterised by its weights, biases, etc.

    (But you can use Equinox to represent any kind of parameterised function.)

And now you can JIT/grad/etc. with respect to your model. For example, using a few built-in layers by way of demonstration:

```python
import equinox as eqx
import jax

class MyModule(eqx.Module):
    layers: list
    bias: jax.numpy.ndarray
    
    def __init__(self, key):
        key1, key2, key3 = jax.random.split(key, 3)
        self.layers = [eqx.nn.Linear(2, 8, key=key1),
                       eqx.nn.Linear(8, 8, key=key2),
                       eqx.nn.Linear(8, 2, key=key3)]
        self.bias = jax.numpy.ones(2)

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x) + self.bias

@jax.jit
@jax.grad
def loss(model, x, y):
    pred_y = jax.vmap(model)(x)
    return jax.numpy.mean((y - pred_y) ** 2)

x_key, y_key, model_key = jax.random.split(jax.random.PRNGKey(0), 3)
x = jax.random.normal(x_key, (100, 2))
y = jax.random.normal(y_key, (100, 2))
model = MyModule(model_key)
grads = loss(model, x, y)
learning_rate = 0.1
model = jax.tree_map(lambda m, g: m - learning_rate * g, model, grads)
```

## Filtering

In the previous example, all of the model attributes were `Module`s and JAX arrays.

But Equinox supports using arbitrary Python objects too! If you choose to include those in your models, then Equinox offers the tools to handle them appropriately around `jax.jit` and `jax.grad`. (Which themselves only really work with JAX arrays.)

!!! example

    You might have a `bool`-ean flag in your model-as-a-PyTree, specifying whether to enable some extra piece of behaviour. You might want to treat that as a `static_argnum` to `jax.jit`.

!!! example

    The activation function in a multi-layer perceptron isn't a JAX array -- it's an arbitrary Python function -- but you may reasonably wish to make that part of your model's parameterisation. (Rather than just baking in a particular choice into your forward pass.)

If you want to do this, then Equinox offers *filtering*, like follows.

```python
import equinox as eqx
import functools as ft
import jax

# Start off by creating a model just like normal, but with some arbitrary Python
# objects as part of its parameterisation. In this case, we have `jax.nn.relu`, which
# is a Python function.
class AnotherModule(eqx.Module):
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
model = AnotherModule(model_key)

# Option 1: manually filter out anything that isn't JIT/grad-able.

@ft.partial(jax.jit, static_argnums=1)
@jax.grad
def loss(params, static, x, y):
    model = eqx.combine(params, static)
    pred_y = jax.vmap(model)(x)
    return jax.numpy.mean((y - pred_y) ** 2)

params, static = eqx.partition(model, eqx.is_array)
loss(params, static, x, y)

# Option 2: use filtered transformations, which automates the above process for you.

@eqx.filter_jit
@eqx.filter_grad
def loss(model, x, y):
    pred_y = jax.vmap(model)(x)
    return jax.numpy.mean((y - pred_y) ** 2)

loss(model, x, y)
```

Here, `params` and `static` are both instances of `AnotherModule`: `params` keeps just the leaves that are JAX arrays; `static` keeps everything else. Then `combine` merges the two PyTrees back together after crossing the `jax.jit` and `jax.grad` API boundaries.

The choice of `eqx.is_array` is a *filter function*: a boolean function specifying whether each leaf should go into `params` or into `static`. In this case very simply `eqx.is_array(x) == isinstance(x, jax.numpy.ndarray)`.

As a convenience, `eqx.filter_jit` and `eqx.filter_grad` just wrap filtering and transformation together. It turns out to be really common to only need to filter around JAX transformations.

## Integrates smoothly with JAX

Equinox introduces a powerful yet straightforward way to build neural networks, without introducing lots of new notions or tieing you into a framework.

Equinox is all just regular JAX -- PyTrees and transformations. Together, these two pieces allow us to specify complex models in JAX-friendly ways.

## Next steps

And that's it! That's pretty much everything you need to know about Equinox. Everything you've seen so far should be enough to get started with using the library. Also see the [Train RNN](./examples/train_rnn.ipynb) example for a fully worked example.

## Summary

Equinox includes four main things:

- For building models: `equinox.Module`.
- Prebuilt neural network layers: `equinox.nn.Linear`, `equinox.nn.Conv2d`, etc.
- Filtering, and filtered transformations: `equinox.filter`, `equinox.filter_jit` etc.
- Some utilities to help manipulate PyTrees: `equinox.tree_at` etc.

See also the API reference on the left.

!!! faq "FAQ"

    One common question that gets asked: a lot of other libraries introduce custom `library.jit` etc. operations, specifically to work with `library.Module`. What makes the filtered transformations of Equinox different?

    The answer is that filters are tools that apply to *any PyTree*; and models just happen to be PyTrees. There is no special coupling between a filtered transformation and `eqx.Module`. (Not to mention that the purpose of filtered transformations -- being able to include non-JAX-arrays in your model's parameters -- is itself unusual/impossible in many other libraries.)
