<h1 align='center'>Equinox</h1>
<h2 align='center'>Callable PyTrees and filtered JIT/grad transformations<br>=> neural networks in JAX</h2>

Equinox brings more power to your model building in [JAX](https://github.com/google/jax).<br>
- *Filtered tranformations*: `jitf` and `gradf`;
- Specifying models as *callable PyTrees*;
- Integrates smoothly with JAX: Equinox is a library, not a framework.

This is half tech-demo, half neural network library.

### Equinox in brief

The design goals for Equinox were that it must be:
- as minimal as possible, and use no behind-the-scenes magic;
- of functional style, working with existing JAX operations like `jax.jit` without difficulty;
- offer a PyTorch-like class-based syntax to build models.

Equinox is a library, not a framework. It is not a wrapper around JAX, like many neural network libraries are. In particular it introduces no new concepts around how to manage state, and allows you to use not just arbitrary PyTrees but arbitrary Python objects as well. Its models operate with day-to-day JAX (including `jit`, `grad` etc.) without any surprises.

Its elegance is its selling point in a world that already has [Haiku](https://github.com/deepmind/dm-haiku), [Flax](https://github.com/google/flax) etc.

Quick example:

```python
import equinox as eqx
import functools as ft, jax, jax.numpy as jnp, jax.random as jrandom, typing

# PyTorch-like way to define models. `Module` is a PyTree node. The instance of our model will be a PyTree.
# The PyTree structure can hold arbitrary Python objects -- you're not restricted to just JAX arrays.
class LinearOrIdentity(eqx.Module):
    weight: typing.Any  # we want to differentiate and JIT-trace this
    flag: bool          # we want to JIT-static this

    def __init__(self, in_features, out_features, flag, key):
        self.weight = jrandom.normal(key, (out_features, in_features))
        self.flag = flag

    def __call__(self, x):
        if self.flag:
            return x
        return self.weight @ x

# We pass in a `model` argument to the loss. Remember that `model` is a PyTree. We now use *filtered
# transformations* to unpack it as a PyTree, and select just the leaves we want to JIT/differentiate.
# (In this simple case all floating-point JAX arrays -- i.e. `weight` but not `flag`.)
# Equinox is JAX-friendly. If you want to differentiate everything, just use `jax.jit` and `jax.grad`.
@ft.partial(eqx.jitf, filter_fn=eqx.is_inexact_array)
@ft.partial(eqx.gradf, filter_fn=eqx.is_inexact_array)
def loss(model, x, y):
    pred_y = jax.vmap(model)(x)
    return jnp.mean((y - pred_y) ** 2)

modelkey, xkey, ykey = jrandom.split(jrandom.PRNGKey(0), 3)
model = LinearOrIdentity(2, 3, flag=False, key=modelkey)
x, y = jrandom.normal(xkey, (100, 2)), jrandom.normal(ykey, (100, 3))
grads = loss(model, x, y)
```

This quick example exposes you to the two main concepts in Equinox: *callable PyTrees* and *filtered transformations*. Together, they're very powerful.

**Callable PyTrees**<br>
This is just some methods attached to a PyTree. In this case it's the `__call__` method of a `Module` subclass.

The PyTree structure holds the data (parameters, submodules, boolean flags, arbitrary Python objects etc.), and the methods define operations parameterised by that data -- for example, the forward pass through a model.

This means that the `__call__` method in the above example is a *pure function acting on PyTrees* (`self` and `x`) -- the ideal scenario for JAX.

Callable PyTrees aren't anything special -- the build-in Python methods on lists and dictionaries are other examples of callable PyTrees.

**Filtered transformations**<br>
The one issue with putting everything about a model into a single PyTree is that this might not contain just trainable parameters. The above example includes a boolean `flag`, for example. In general we might have arbitrary Python objects, or perhaps JAX arrays that are *not* trainable parameters.

Enter *filtered transformations*. These are `jitf` and `gradf`, which are very thin wrappers around `jax.jit` and `jax.grad`. Instead of specifying `argnums` to JIT/differentiate, we instead pass a *filter* that determines which *PyTree leaves* -- not just whole arguments -- to JIT/differentiate.

These aren't "a way to make JIT/grad work with model states" like many libraries have. They are general operations on PyTrees, and nothing about `Module` is special-cased. (And if you don't want to filter out anything at all, then don't: use `jax.jit` and `jax.grad` directly and they'll work just fine.)

This gives a powerful fine-grained way control JIT and autodifferentiation.

**Integrates smoothly with JAX**<br>
Together, these two pieces allow us to specify complex models in JAX-friendly ways.

There's no behind-the-scenes magic here. There's no framework, no class-to-functional transforms, and no complicated collections of variables and states being passed around.

Equinox is its design principles; it is not just another neural network framework.

### Installation

```
pip install git+https://github.com/patrick-kidger/equinox.git
```
Requires Python 3.7+ and JAX 0.2.18+.

## Examples

- [`train_mlp.py`](./examples/train_mlp.py) gives a short example that introduces `jitf` and `gradf`. These will be used to select the parameters of an MLP and train them.
- [`frozen_layer.py`](./examples/frozen_layer.py) demonstrates how this approach really shines: some of the parameters will be trained, some of them will be frozen, but *all* of them will be efficiently JIT-traced.
- [`build_model.py`](./examples/build_model.py) constructs an MLP from scratch using `Module`s. We can produce models using a familiar class-based syntax, that are also functional and integrate directly with JAX's JIT/autograd.

## API

### Full API list
```python
# Filtered transformations       # Filters
equinox.jitf                     equinox.is_inexact_array
equinox.gradf                    equinox.is_array_like
equinox.value_and_grad_f
                                 # Neural networks
# Module                         equinox.nn.Linear
equinox.Module                   equinox.nn.Identity
                                 equinox.nn.Dropout
# Utilities                      equinox.nn.GRUCell
equinox.apply_updates            equinox.nn.LSTMCell
equinox.tree_at                  equinox.nn.Sequential
equinox.tree_equal               equinox.nn.MLP
```

### Filtered transformations

```python
equinox.jitf(fun, *, filter_fn=None, filter_tree=None, **kwargs)
```
Wraps `jax.jit`.

- `fun` is a pure function to JIT compile.
- `filter_fn` is a callable `Any -> bool`. It will be called on every leaf of every PyTree that is inputted to `fun`. If it returns `True`, the leaf will be traced. It returns `False`, the leaf with be treated as static. Mutually exclusive with `filter_tree`.
- `filter_tree` is a tree, or tuple of trees, of the same length as the number of inputs. (Or if `static_argnums` is passed, the number of inputs not already marked static via `static_argnums`.) It must have the exact same tree structure as the inputs. Every leaf must be either `True` or `False`. Each leaf of `filter_tree` is matched up against the corresponding input: if it is `True` the leaf will be traced; it it is `False` the leaf will be treated as static. Mutually exclusive with `filter_tree`.
- `**kwargs` are the usual other arguments to `jax.jit`, like `static_argnums`. In particular, a leaf will be marked static if either (a) it is filtered as being so, *or* (b) it is part of a PyTree that is marked through `static_argnums`.

Precisely one of `filter_fn` or `filter_tree` must be passed.<br>
See also `equinox.is_array_like` as usually a good choice of `filter_fn`: this will trace everything that can possible be traced, with everything else static.<br>
See also `equinox.tree_at` for an easy way to create the `filter_tree` argument.

```python
equinox.gradf(fun, *, filter_fn=None, filter_tree=None, **kwargs)
```
Wraps `jax.grad`.

- `fun` is a pure function to JIT compile.
- `filter_fn` is a callable `Any -> bool`. It will be called on every leaf of every PyTree that is marked as potentially requiring gradient via `argnums`. If it returns `True`, the leaf will be differentiated. If it returns `False`, the leaf will not be differentiated. Mutually exclusive with `filter_tree`.
- `filter_tree` is a tree, or tuple of trees, of the same length as the number of inputs marked as potentially requiring gradient via `argnums`. It must have the exact same tree structure as the inputs. Every leaf must be either `True` or `False`. Each leaf of `filter_tree` is matched up against the corresponding input: if it is `True` the leaf will be differentiated; if it is `False` the leaf will not be differentiated. Mutually exclusive with `filter_fn`.
- `**kwargs` are the usual other argments to `jax.grad`, like `argnums`. In particular, a leaf will only be differentiated if (a) it is filtered as being so, *and* (b) it is part of a PyTree that is marked through `argnums`.

Precisely one of `filter_fn` or `filter_tree` must be passed.<br>
See also `equinox.is_inexact_array` as usually a good choice of `filter_fn`: this will differentiate all floating-point arrays.<br>
See also `equinox.tree_at` for an easy way to create the `filter_tree` argument.

Note that as the returned gradients must have the same structure as the inputs, then all nondifferentiable components of the input PyTrees will have gradient `0`. If the nondifferentiable component is an arbitrary Python object that does not support addition, then doing a simple `jax.tree_map(lambda m, g: m - lr * g, model, grad)` may fail. As such Equinox provides `equinox.apply_updates` as a simple convenience: it will only apply the update if the gradient is nonzero. See below.

```python
equinox.value_and_grad_f(fun, *, filter_fn=None, filter_tree=None, **kwargs)
```
Wraps `jax.value_and_grad`. Arguments are as `equinox.gradf`.

### Filters

Any function `Any -> bool` can be used as a filter. We provide some convenient common choices.

```python
equinox.is_inexact_array(element)
```
Returns `True` if `element` is a floating point JAX array (but not a NumPy array).

```python
equinox.is_array_like(element)
```
Returns `True` if `element` can be interpreted as a JAX array. (i.e. does `jax.numpy.array` throw an exception or not.)

### Module

```python
equinox.Module
```
Base class; create your model by inheriting from this.

Specify all its attributes at the class level (identical to [dataclasses](https://docs.python.org/3/library/dataclasses.html)). This defines its children in the PyTree.

```python
class MyModule(equinox.Module):
    weight: typing.Any
    bias: typing.Any
    submodule: Module
```

A default `__init__` method is provided, which just fills in these attributes with the argments passed: `MyModule(weight, bias, submodule)` or `MyModule(weight=weight, bias=bias, submodule=submodule)`. Alternatively you can provide an `__init__` method yourself. (For example to specify dimension sizes instead of raw weights.) By the end of `__init__`, every attribute must have been assigned.

```python
class AnotherModule(equinox.Module):
    weight: Any

    def __init__(self, input_size, output_size, key):
        self.weight = jax.random.normal(key, (output_size, input_size))
```

After initialisation then attributes cannot be modified.

It is typical to also create some methods on the class. As `self` will be an input parameter -- treated as a PyTree -- then these methods will get access to the attributes of the instance. Defining `__call__` gives an easy way to define a forward pass for a model:

```python
class LinearWithoutBias(equinox.Module):
    weight: Any

    def __call__(self, x):
        return self.weight @ x
```

If defining a method `meth`, then take care not to write `instance = MyModule(...); jax.jit(instance.meth)(...)`. (Or similarly with `jax.grad`, `equinox.jitf` etc.) This is because `instance.meth` is not a pure function as it already has the `self` parameter passed implicitly. Instead do either `jax.jit(MyModule.meth)(instance, ...)` or
```python
@jax.jit
def func(instance, args):
    instance.meth(args)
    # Also use this pattern with instance(args) if you defined `__call__` instead of `meth`.
```

### Utilities

```python
equinox.apply_updates(model, updates)
```
Performs a training update to a model.
- `model` must be a PyTree;
- `updates` must be a PyTree with the same structure.
It essentially performs `jax.tree_map(lambda m, u: m + u, modelm updates)`. However anywhere `updates` is zero then no update is made at all, so as to handle nondifferentiable parts of `model` that may not have addition defined (e.g. activation functions).

To produce `updates`, it is typical to take the gradients from the loss function, and then adjust them according to any standard optimiser; for example [Optax](https://github.com/deepmind/optax) provides `optax.sgd` or `optax.adam`.

```python
equinox.tree_at(where, pytree, replace=_sentinel, replace_fn=_sentinel)
```
Modifies an existing tree, and returns the modified tree. (Like `.at` for "in place modifications" of JAX arrays.)

- `where` is a callable `PyTree -> Leaf` or `PyTree -> Tuple[Leaf, ...]`. It should consume a PyTree of the same shape as `pytree`, and return the leaf or leaves that should be replaced. For example `where=lambda p: p[-1].linear.weight`.
- `pytree` is the existing PyTree to modify.
- `replace` should either be a single element, or a tuple of the same length as returned by `where`. This specifies the replacements to make at the locations specified by `where`. Mutually exclusive with `replace_fn`.
- `replace_fn` should be a function `Leaf -> Any`. It will be called on every leaf replaced using `where`. The return value from `replace_fn` will be used in its place. Mutually exclusive with `replace`.

For example this can be used to specify the weights of a model to train or not train:
```python
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
equinox.nn.Dropout(p=0.5, deterministic=False)(input, *, key=None, deterministic=None)
equinox.nn.GRUCell(input_size, hidden_size, bias=True, *, key)(input, hidden)
equinox.nn.LSTMCell(input_size, hidden_size, bias=True, *, key)(input, hidden)
equinox.nn.Sequential(layers)(input, *, key=None)
equinox.nn.MLP(in_size, out_size, width_size, depth,
               activation=jax.nn.relu, final_activation=lambda x: x, *, key)(input)
```
These all behave in the way you expect. The `key` arguments are used to generate the random initial weights, or to generate randomness on the forward pass of stochastic layers like `Dropout`.

The `Dropout(deterministic=...)(deterministic=...)` options determines whether to have the layer act as the identity function, as is commonly done with dropout during inference time. The call-time `deterministic` takes precendence if it passed; otherwise the init-time `deterministic` is used. (Note that because models are PyTrees, you can modify the init-time `deterministic` flag using `equinox.tree_at`. This is perfectly fine, and might be handy if it's easier than using the call-time flag.)

The `MLP(final_activation=...)` option determines any final activation function to apply after the last layer. (In some cases it is desirable for this to be different to the activation used in the main part of the network.)
