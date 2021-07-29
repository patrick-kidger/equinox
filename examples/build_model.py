#############
#
# In this example, we look at how to use Equinox's concept of Modules.
#
# Now I know what you're thinking! Every deep learning library out there has a class called Module. There's
# haiku.Module, flax.linen.Module, objax.Module etc.
#
# And each time you have to sit down and read the documentation and understand what "Module" means for each library.
# A lot of these quite heavyweight, and come with a lot of baggage - introducing extra notions of variables,
# parameters, module names, etc. For example there's `haiku.transform` or `objax.VarCollection`, that you need to
# understand to use these libraries.
#
# Equinox aims to be different; to be simple. (If you want look up the source code for `equinox.Module` -- it's only
# about 50 lines long.) It's JAX like you're used to, and doesn't introduce any extra concepts at the same time.
#
#############
#
# Now that we've finished complaining about the competition ;) let's see how it works.
#
# It's very simple: (subclasses of) Modules are PyTrees. This means you can stack them and work with them like in
# normal JAX. Any attributes of the Module are also part of the same PyTree. (For example, the parameters of the
# network as jnp.arrays, or a boolean flag dictating how the model should behave, or even arbitrary Python objects
# like a function e.g. specifying the activation function.)
#
# But because they're classes, we can also define a __call__ method. That is, we can associate a function with our
# PyTree of data.
# (In fact we can define any method we like; none of them are special-cased. __call__ means that we can do `model()`
# rather than `model.method()`, but do whatever you prefer. If you want multiple functions associated with your data
# then go ahead and define multiple methods.)
#
# This means that there is *no distinction between model and parameters*. You just have a single thing -- a model --
# which is both a PyTree of data and some functions parameterised by that data. And the PyTree of data can include
# both jnp.arrays and arbitrary Python objects, no special-casing between them.
#
#############
#
# The trick we have up our sleeve that makes this so useful, of course, is `equinox.gradf` and `equinox.jitf`. This
# allows us to grad and JIT (and vmap; `jax.vmap` is usually enough already) the forward passes of our models without
# any headaches about the complicated PyTrees we're passing around as models.
#
# In this example, we'll demonstrate how to use `equinox.Module` to create simple MLP.

import equinox as eqx
import functools as ft
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from typing import Any, List


# First pass at creating a Linear layer. `Linear1` will be a PyTree node, with `weight` and `bias` as its children.
# A type annotation (the `Any`) is needed for these to be recognised; just use `typing.Any` if you want.
class Linear1(eqx.Module):
    weight: Any
    bias: Any

    def __call__(self, x):
        return self.weight @ x + self.bias


# We do need to initialise this. If we wanted to we could create a free function to do it:
def linear(in_features, out_features, key):
    wkey, bkey = jrandom.split(key)
    weight = jrandom.normal(key, (out_features, in_features))
    bias = jrandom.normal(key, (out_features,))
    return Linear1(weight, bias)  # uses the default __init__ for Linear1.


# Alternatively we can  use a custom __init__:
class Linear2(eqx.Module):
    weight: Any
    bias: Any

    def __init__(self, in_features, out_features, key):
        super().__init__()  # Not actually necessary here, but good practice.
        wkey, bkey = jrandom.split(key)
        weight = jrandom.normal(key, (out_features, in_features))
        bias = jrandom.normal(key, (out_features,))
        self.weight = weight
        self.bias = bias
        # An error will be thrown if you forget to set either `weight` or `bias` (or if you to try to set
        # anything else).

    def __call__(self, x):
        return self.weight @ x + self.bias


# And now we can compose these into a small MLP:
class MLP(eqx.Module):
    layers: List[Linear2]
    activation: callable

    def __init__(self, in_size, out_size, width_size, depth, key, activation=jnn.relu):
        super().__init__()  # Once again not necessary but good Python practice
        keys = jrandom.split(key, depth + 1)
        self.layers = []
        self.layers.append(Linear2(in_size, width_size, keys[0]))
        for i in range(depth - 1):
            self.layers.append(Linear2(width_size, width_size, keys[i + 1]))
        self.layers.append(Linear2(width_size, out_size, keys[-1]))
        self.activation = activation

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        return self.layers[-1](x)


# Which we can now use:
def main():
    key = jrandom.PRNGKey(5678)
    model_key, data_key = jrandom.split(key, 2)
    model = MLP(in_size=2, out_size=2, width_size=8, depth=2, key=model_key)
    data = jrandom.normal(data_key, (2,))
    model(data)  # Calls __call__

    # Because `model` is a PyTree we can use it with normal JAX: vmap, grad, jit etc.
    # The equinox.jitf and equinox.gradf utilities can also be helpful to filter on what you do and don't want to
    # include.

    @ft.partial(eqx.jitf, filter_fn=eqx.is_inexact_array)
    def example_jit(model, data):
        model(data)

    @ft.partial(eqx.gradf, filter_fn=eqx.is_inexact_array)
    def example_grad(model, data):
        return jnp.sum(model(data))  # return a scalar

    @ft.partial(jax.vmap, in_axes=(None, 0))
    def example_vmap(model, data):
        return model(data)

    # (Note that eqx.jitf(model, ...), jax.jit(model, ...), eqx.gradf(model, ...), etc. would be wrong.
    #  The function argument to these operations must always be a pure function.)

    example_jit(model, data)
    example_grad(model, data)
    example_vmap(model, jnp.stack([data, data]))


if __name__ == "__main__":
    main()
