#############
#
# In this example, we look at Modules.
#
# Now I know what you're thinking! Every deep learning library out there has a class called Module. There's
# haiku.Module, flax.linen.Module, objax.Module etc.
#
# And each time you have to sit down and read the documentation and understand what "Module" means for each library.
# A lot of these also have custom notions of variables, transforms, scopes, etc. For example there's `haiku.transform`
# or `objax.VarCollection`.
#
# In constrast, Equinox introduces no new abstractions. An Equinox Module is just a nice way to create a PyTree,
# really. If you want, look up the source code for `equinox.Module` -- it's only about 70 lines long.
#
#############
#
# Now that we've finished complaining about the competition ;) let's see how it works.
#
# It's very simple: `Module`, and its subclasses, are PyTrees. Any attributes of the Module are also part of the
# same PyTree. (Your whole model is just one big PyTree.) This means you can use the model in the normal way in
# JAX, with vmap/grad/jit etc.
#
# These attributes can be JAX arrays, but they can also be arbitrary Python objects. Crucially: because of the
# filtering provided `equinox.jitf` and `equinox.gradf`, it's possible to elegantly select just those elements of the
# PyTree that you'd like any given transformation to interact with. This can mean excluding anything that isn't a JAX
# array. If you want it can also be used to freeze a layer, and not compute gradients with respect to some parameters.
#
# Now because a `Module` is also a class, we can define methods on it. The `self` parameter -- which is a PyTree,
# remember! -- means that this is just a function that takes PyTrees as inputs, like any other function. No method
# is special cased. If you want you can group several related functions under different methods. If you just want
# to define a single forward pass, then the __call__ method is a convenient choice.
#
#############
#
# In this example, we'll demonstrate how to use `equinox.Module` to create a simple MLP.

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
    model = MLP(in_size=2, out_size=3, width_size=8, depth=2, key=model_key)
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
