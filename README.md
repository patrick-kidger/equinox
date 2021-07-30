<h1 align='center'>Equinox</h1>
<h2 align='center'>Callable PyTrees and filtered JIT/grad transformations<br>=> neural networks in JAX</h2>

Equinox brings more power to your [JAX](https://github.com/google/jax) model-building.
- Filtered tranformations: `jitf` and `gradf`;
- Specifying models as callable PyTrees;
- Integrates smoothly with JAX: Equinox is a library, not a framework.

### Equinox in brief

We're going to discuss how to design a neural network libary from scratch. Our conditions are going to be that it must be:
- (a) as minimal as possible;
- (b) of functional style, working with existing JAX operations like `jax.jit` without difficulty,
- (c) whilst *also* using a PyTorch-like class-based syntax to build models.

Equinox accomplishes these goals by synergising two main ideas: *callable PyTrees* and *filtered transformations*. In isolation neither of them mean much. But put together, they're very powerful.

**Callable PyTrees**<br>
Most neural network libraries in JAX represent a model's parameters as a PyTree. This is a sensible thing to do. Now, we're going to let this PyTree use [dataclasses](https://docs.python.org/3/library/dataclasses.html) as PyTree nodes. Dataclasses are a popular way to specify a typed data structure, for example the kind of parameters each network layer uses.

Now for the first trick. As dataclasses are classes as well as PyTrees, we can *also* define methods on them. As the `self` parameter to such a method is of a dataclass-that-is-a-PyTree, then *each of these methods is now a pure function acting on PyTrees*. This is exactly what JAX uses! Data is held in the PyTree structure, and each method defines operations parameterised by that data -- for example, the forward pass through a model.

We have obtained a *callable PyTree*, which (a) fits the JAX functional programming paradigm, **and** (b) can be used to offer a familiar (PyTorch-like) class-based syntax for building models.

There's one small problem. We have the parameters of the model held in the PyTree structure, sure ... but we might also need to store boolean flags indicating special behaviour, or some JAX arrays that aren't parameters, or even arbitrary Python objects capable of doing anything at all. We need to be able to JIT/autodifferentiate only *part* of our PyTree structure. (The part of the model holding the parameters.) JAX doesn't offer anything capable of doing this at all.

**Filtered transformations**<br>
Now for the second trick. Enter *filtered transformations*: `jitf` and `gradf`. These are thin wrappers around `jax.jit` and `jax.grad`, that
- unpacks PyTree arguments;
- examines every leaf with a user-specified filter to determine what should be JIT'd or autodifferentiated;
- passes the results to `jax.jit` or `jax.grad`;
- and finally interprets the answer back into a PyTree.

Filtered transformations give a fine-grained way to control JIT and autodifferentiation. Build a complex model as a PyTree parameterised by anything you like, arbitrary Python objects included. Then during its forward pass, filter which pieces are static/traced in the JIT compiler, or which pieces are differentiable/nondifferentiable in the autodifferentiation.

- For example, you could statically compile with respect to boolean flags or arbitrary Python objects embedded inside the model Pytree, whilst still JIT-tracing with respect to the parameters and model inputs.
- As another example, you could mark some parameters of a model as being nondifferentiable, and therefore frozen.

**Integrates smoothly with JAX**<br>
Together, these two pieces allow us to specify complex models in JAX-friendly ways.

Best of all, there's no magic here. No frameworks, no class-to-functional transforms, no complicated collections of variables and states being passed around.

As far as JAX is concerned, there is nothing special about the dataclasses we use as PyTree nodes. (And they will integrate just fine with any larger PyTree you put them in.) Meanwhile `jitf` and `gradf` are just general purpose-functions. 

Equinox is its design principles; it is not just another neural network framework.

### Installation

```
pip install git+https://github.com/patrick-kidger/equinox.git
```
Requires JAX 0.2.18+.

## Examples

- [`train_mlp.py`](./examples/train_mlp.py) gives a short example that introduces `jitf` and `gradf`. These will be used to select the parameters of an MLP and train them.
- [`frozen_layer.py`](./examples/frozen_layer.py) demonstrates how this approach really shines: some of the parameters will be trained, some of them will be frozen, but *all* of them will be efficiently JIT-traced.
- [`build_model.py`](./examples/build_model.py) constructs an MLP from scratch using `Module`s. We can produce models using a familiar class-based syntax, that are also functional and integrate directly with JAX's JIT/autograd.

As a quick example:
```python
import equinox as eqx, functools as ft, jax, jax.numpy as jnp, jax.random as jrandom, typing

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

# Differentiate and trace every floating-point array. Everything else is static/undifferentiated.
# `filter_fn` is just a boolean function specifying whether to jit/grad each leaf of the PyTree.
@ft.partial(eqx.jitf, filter_fn=eqx.is_inexact_array)
@ft.partial(eqx.gradf, filter_fn=eqx.is_inexact_array)
def loss(model, x, y):
    pred_y = jax.vmap(model)(x)
    return jnp.mean((y - pred_y) ** 2)

modelkey, xkey, ykey = jrandom.split(jrandom.PRNGKey(0), 3)
model = LinearOrIdentity(2, 3, flag=False, key=modelkey)  # is a PyTree with elements `weight` and `flag`.
x, y = jrandom.normal(xkey, (100, 2)), jrandom.normal(ykey, (100, 3))  # batch of data
grads = loss(model, x, y)
```

## API

### Filtered transformations

```python
equinox.jitf(fun, *, filter_fn=None, filter_tree=None, **kwargs)
```
Wraps `jax.jit`.

- `fun` is a pure function to JIT compile.
- `filter_fn` is a callable `Any -> bool`. It will be called on every leaf of every PyTree that is inputted to `fun`. If it returns `True`, the leaf will be traced. It returns `False`, the leaf with be treated as static. Mutually exclusive with `filter_tree`.
- `filter_tree` is a tree, or tuple of trees, of the same length as the number of inputs. (Or if `static_argnums` is passed, the number of inputs not already marked static via `static_argnums`.) It must have the exact same tree structure as the inputs. Every leaf must be either `True` or `False`. Each leaf of `filter_tree` is matched up against the corresponding input: if it is `True` the leaf will be traced; it it is `False` the leaf will be treated as static. Mutually exclusive with `filter_tree`.
- `**kwargs` are the usual other arguments to `jax.jit`, like `static_argnums`. In particular, a leaf will be marked static if either (a) it is filtered as being so, *or* (b) it is part of a PyTree that is marked through `static_argnums`.

At least one of `filter_fn` or `filter_tree` must be passed. Also see the `equinox.tree_at` function below for an easy way to create the `filter_tree` argument.

```python
equinox.gradf(fun, *, filter_fn=None, filter_tree=None, **kwargs)
```
Wraps `jax.grad`.

- `fun` is a pure function to JIT compile.
- `filter_fn` is a callable `Any -> bool`. It will be called on every leaf of every PyTree that is marked as potentially requiring gradient via `argnums`. If it returns `True`, the leaf will be differentiated. If it returns `False`, the leaf will not be differentiated. Mutually exclusive with `filter_tree`.
- `filter_tree` is a tree, or tuple of trees, of the same length as the number of inputs marked as potentially requiring gradient via `argnums`. It must have the exact same tree structure as the inputs. Every leaf must be either `True` or `False`. Each leaf of `filter_tree` is matched up against the corresponding input: if it is `True` the leaf will be differentiated; if it is `False` the leaf will not be differentiated. Mutually exclusive with `filter_fn`.
- `**kwargs` are the usual other argments to `jax.grad`, like `argnums`. In particular, a leaf will only be differentiated if (a) it is filtered as being so, *and* (b) it is part of a PyTree that is marked through `argnums`.

At least one of `filter_fn` or `filter_tree` must be passed. Also see the `equinox.tree_at` function below for an easy way to create the `filter_tree` argument.

Note that as the returned gradients must have the same structure as the inputs, then all nondifferentiable components of the input PyTrees will have gradient `0`. If the nondifferentiable component is an arbitrary Python object then doing a simple `jax.tree_map(lambda m, g: m - lr * g, model, grad)` may fail. As such Equinox provides `equinox.apply_updates` as a convenience: it will only tree to apply the update if the gradient is nonzero. See below.

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
