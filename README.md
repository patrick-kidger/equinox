<h1 align='center'>Equinox</h1>
<h2 align='center'>Callable PyTrees and filtered JIT/grad transformations<br>=> neural networks in JAX</h2>

Equinox brings more power to your JAX workflow.
- Filtered JIT and grad;
- Specifying models as callable PyTrees;
- Integrates smoothly with JAX: Equinox is a library, not a framework.

### In brief

Equinox synergises two main ideas: *callable PyTrees* and *filtered transformations*.

**Callable PyTrees**<br>
Most neural networks in JAX represent the model's parameters as a PyTree. In a similar vein, Equinox provides `Module` as a class that is also a PyTree. (JAX allows you to turn classes into PyTrees.) Its subtrees are parameters and other modules, ... and arbitary Python objects; more on that later. As the class is a PyTree, then the `self` parameter to its method is now a PyTree. And so each method is now a function acting on PyTrees.

In this way the class (with its methods) are a *callable PyTree*. It has data in the PyTree structure, and can define functions (like a forward pass through a model) via its methods. Thus, we have models that (a) fit the JAX functional programming paradigm, **and** (b) use a familiar class-based syntax for building models. Simple, right?

Well: models can be complicated. We can have the parameters of the model, sure ... but we can also have boolean flags indicating special behaviour, or arbitrary Python objects doing special things, or maybe even some JAX arrays that aren't parameters at all. This mean we want to JIT and autodifferentiate with respect to only *part* of the `self` argument -- usually those arrays representing parameters -- recall that `self` is a PyTree with parameters (and everything else) inside it.

This problem is probably the reason that this "simple apprach" isn't the beginning and end of every JAX-based neural network library.

**Filtered transformations**<br>
Now for the real trick. Enter *filtered transformations*: `jitf` and `gradf`. These are thin wrappers around `jax.jit` and `jax.grad`, that unpack PyTree arguments, examine every leaf with a filter function that specifies what should be JIT'd or autodifferentiated, and then passes them on to `jax.jit` and `jax.grad`.

This gives a powerful fine-grained way to control JIT and autodifferentiation. Build a complex model as a PyTree parameterised by anything you like. Then during its forward pass, simply filter which pieces are static/traced in the JIT compiler, or which pieces are differentiable/nondifferentiable in the autodifferentiation. For example you can statically compile with respect to boolean flags or arbitrary Python objects embedded inside the model Pytree; or you can mark some parameters of a model as being nondifferentiable and therefore frozen.

**Integrates smoothly with JAX**<br>
Equinox is a library and not a framework. It integrates directly with existing JAX. Its callable PyTree abstraction is just a convenient way to create PyTrees and functions together, but you can do it any other way as well. Its filtered transformations are general-purpose tools you can use on any JAX function you like.

### Installation

```
pip install git+https://github.com/patrick-kidger/equinox.git
```
Requires JAX 0.2.18+.

## Examples

- [`train_mlp.py`](./examples/train_mlp.py) gives a short example that introduces `jitf` and `gradf`. These will be used to select the parameters of an MLP and train them.
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

### Filtered transformations

```python
equinox.jitf(fun, *, filter_fn=None, filter_tree=None, **kwargs)
```
Wraps `jax.jit`.

- `fun` is a pure function to JIT compile.
- `filter_fn` is a callable `Any -> bool`. It will be called on every leaf of every PyTree that is inputted to `fun`. If it returns `True`, the leaf will be traced. It returns `False`, the leaf with be treated as static.
- `filter_tree` is a tree, or tuple of trees, of the same length as the number of inputs. (Or if `static_argnums` is passed, the number of inputs not already marked static via `static_argnums`.) It must have the exact same tree structure as the inputs. Every leaf must be either `True` or `False`. Each leaf of `filter_tree` is matched up against the corresponding input: if it is `True` the leaf will be traced; it it is `False` the leaf will be treated as static.
- `**kwargs` are the usual other arguments to `jax.jit`, like `static_argnums`. In particular, a leaf will be marked static if either (a) it is filtered as being so, *or* (b) it is part of a PyTree that is marked through `static_argnums`.

Note that `filter_tree` may be easily created using the `equinox.tree_at` function.

```python
equinox.gradf(fun, *, filter_fn=None, filter_tree=None, **kwargs)
```
Wraps `jax.grad`.

- `fun` is a pure function to JIT compile.
- `filter_fn` is a callable `Any -> bool`. It will be called on every leaf of every PyTree that is marked as potentially requiring gradient via `argnums`. If it returns True, the leaf will be differentiated. If it returns False, the leaf will not be differentiated.
- `filter_tree` is a tree, or tuple of trees, of the same length as the number of inputs marked as potentially requiring gradient via `argnums`. It must have the exact same tree structure as the inputs. Every leaf must be either `True` or `False`. Each leaf of `filter_tree` is matched up against the corresponding input: if it is `True` the leaf will be differentiated; if it is `False` the leaf will not be differentiated.
- `**kwargs` are the usual other argments to `jax.grad`, like `argnums`. In particular, a leaf will only be differentiated if (a) it is filtered as being so, *and* (b) it is part of a PyTree that is marked through `argnums`.

```python
equinox.value_and_grad_f(fun, *, filter_fn=None, filter_tree=None, **kwargs)
```
Wraps `jax.value_and_grad`. Arguments are as `equinox.gradf`.

### Filters

Any function `Any -> bool` can be used as a filter. We provide some convenient common defaults.

```python
equinox.is_inexact_array(element)
```
returns `True` if `element` is a floating point JAX array (but not a NumPy array).

```python
equinox.is_array_like(element)
```
returns `True` if `element` can be interpreted as a JAX array. (i.e. does `jax.numpy.array` throw an exception or not.)

### Module

```python
equinox.Module
```
Base class; create your model by inheriting from this.

Specify all its attributes at the class level (identical to [dataclasses](https://docs.python.org/3/library/dataclasses.html)). This defines its children in the PyTree.

```python
class MyModule(Module):
    weight: typing.Any
    bias: typing.Any
    submodule: Module
```

A default `__init__` method is provided, which just fills in these attributes with the argments passed: `MyModule(weight, bias, submodule)` or `MyModule(weight=weight, bias=bias, submodule=submodule)`. Alternatively you can provide an `__init__` method yourself. (For example to specify dimension sizes instead of raw weights.) By the end of `__init__`, every attribute must have been assigned.

```python
class AnotherModule(Module):
    weight: Any

    def __init__(self, input_size, output_size, key):
        self.weight = jax.random.normal(key, (output_size, input_size))
```

After initialisation then attributes cannot be modified.

It is typical to also create some methods on the class. As `self` will be an input parameter -- treated as a PyTree -- then these methods will get access to the attributes of the instance. Defining `__call__` gives an easy way to define a forward pass for a model:

```python
class LinearWithoutBias(Module):
    weight: Any

    def __call__(self, x):
        return self.weight @ x
```

If defining a method `meth`, then take care not to ever do `instance = MyModule(...); jax.jit(instance.meth)(...)`. (Or similarly with `jax.grad`, `equinox.jitf` etc.) As `instance.meth` implicitly has `self` parameter already passed in, then it is not a pure function. Instead do either `jax.jit(MyModule.meth)(instance, ...)` or
```python
@jax.jit
def func(instance, args):
    instance.meth(args)
    # Use this pattern to do instance(args) if you defined `__call__` instead of `meth`.
```

### Tree utilities

```python
equinox.tree_at(where, pytree, replace=_sentinel, replace_fn=_sentinel)
```
Modifies an existing tree, and returns the modified tree. (Like `.at` for "in place modifications" in JAX.)

- `where` is a callable `PyTree -> Leaf` or `PyTree -> Tuple[Leaf, ...]`. It should consume a PyTree of the same shape as `pytree`, and return the leaf or leaves that should be replaced. For example `where=lambda p: p[-1].linear.weight`.
- `pytree` is the existing PyTree to modify.
- `replace` should either be a single element, or a tuple of the same length as returned by `where`. This specifies the replacements to make at the locations specified by `where`. Mutually exclusive with `replace_fn`.
- `replace_fn` should be a function `Leaf -> Any`. It will be called on every leaf specified by `where` as being replaced; the return value from `replace_fn` will be used in its place. Mutually exclusive with `replace`.

For example this can be used to specify the weights of a model to train or not train:
```
trainable = jax.tree_map(lambda _: False, model)
trainable = equinox.tree_at(lambda p: p[-1].linear.weight, model, replace=True)
equinox.gradf(..., filter_tree=trainable)
```

```python
equinox.tree_equal(*pytrees)
```
Returns `True` if all PyTrees in the list are equal. All arrays must have the same shape, dtype, and values. JAX arrays and NumPy arrays are not considered equal.

### Neural network library

Equinox includes a small neural network library, mostly as a tech demo for how the rest of the library can be used. Its API is modelled after PyTorch.

```python
equinox.nn.Linear(in_features, out_features, bias=True, *, key)(input)
equinox.nn.Identity(*args, **kwargs)(input)  # args and kwargs are ignored
equinox.nn.Dropout(p=0.5, deterministic=False)(input, *, key, deterministic=None)
equinox.nn.GRUCell(input_size, hidden_size, bias=True, *, key)(input, hidden)
equinox.nn.LSTMCell(input_size, hidden_size, bias=True, *, key)(input, hidden)
equinox.nn.Sequential(layers)(input, *, key=None)
equinox.nn.MLP(in_size, out_size, width_size, depth,
               activation=jax.nn.relu, final_activation=lambda x: x, *, key)(input)
```
These all behave in the way you expect. The `key` arguments are used to generate the random initial weights, or on the forward pass for stochastic layers like `Dropout`.

The `Dropout(deterministic=...)(deterministic=...)` options determines whether to have the layer act as the identity function, as is commonly done with dropout during inference time. The call-time `deterministic` takes precendence if it passed; otherwise the init-time `deterministic` is used. (Note that because models are PyTrees, you can modify the `deterministic` flag using `equinox.tree_at`, e.g. during inference if it's easier than using the call-time flag.)

The `MLP(final_activation=...)` option determines any final activation function to apply after the last layer. (In some cases it is desirable for this to be different to the activation used in the main part of the network.)
