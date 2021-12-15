<h1 align='center'>Equinox</h1>
<h2 align='center'>Callable PyTrees and filtered JIT/grad transformations<br>=> neural networks in JAX</h2>

Equinox brings more power to your model building in [JAX](https://github.com/google/jax).<br>
Represent *parameterised functions as data* and use *filtered transformations* for powerful fine-grained control of the model-building process. Equinox demonstrates how to use a PyTorch-like class-based API without compromising on JAX-like functional programming.

Equinox is half tech-demo, half neural network library, and comes with no behind-the-scenes magic, guaranteed.

The elegance of Equinox is its selling point in a world that already has [Haiku](https://github.com/deepmind/dm-haiku), [Flax](https://github.com/google/flax) and so on.


## Quick start
### Installation

```
pip install equinox
```
Requires Python 3.7+ and JAX 0.2.18+.

### Parameterised functions as data

Equinox represents *parameterised functions as [PyTrees](https://jax.readthedocs.io/en/latest/pytrees.html)*. (For example a neural network is a function parameterised by its weights, biases, etc.) Now you can JIT/grad/etc. a higher-order function (like a loss function) with respect to a parameterised function as its input (like a model).

Previous libraries have introduced a lot of extra complexity to make this work. e.g. custom notions of parameter groups, class-to-functional transformations, or specially-wrapped `library.jit` or `library.grad` to be compatible with JAX's JIT/grad/etc.

In contrast, Equinox makes it elegant:

```python
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

class MyModule(eqx.Module):
    # Specify the module's attributes;
    layers: list
    bias: jnp.ndarray
    
    # And how to initialise them;
    def __init__(self, key):
        key1, key2 = jrandom.split(key)
        self.layers = [eqx.nn.Linear(2, 8, key=key1),
                       eqx.nn.Linear(8, 2, key=key2)]
        self.bias = jnp.ones(2)

    # And the forward pass of the model.
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jnn.relu(layer(x))
        return self.layers[-1](x) + self.bias

@jax.jit
@jax.grad
def loss(model, x, y):
    pred_y = jax.vmap(model)(x)
    return jnp.mean((y - pred_y) ** 2)

x_key, y_key, model_key = jrandom.split(jrandom.PRNGKey(0), 3)
x, y = jrandom.normal(x_key, (100, 2)), jrandom.normal(y_key, (100, 2))
model = MyModule(model_key)
grads = loss(model, x, y)
```

And there's no magic there! All `eqx.Module` really does is register your class with JAX as a PyTree node. (In fact the source code for `eqx.Module` is only about 100 lines long.)

### Filtering

In the previous example, all of the model attributes were Modules and JAX arrays.

Arbitrary Python objects are fine too! We just need to handle them appropriately around `jax.jit` and `jax.grad`.

```python
import equinox as eqx
import functools as ft
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

class AnotherModule(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2 = jrandom.split(key)
        # Model now has `jnn.relu` -- a Python function -- as part of its PyTree.
        self.layers = [eqx.nn.Linear(2, 8, key=key1),
                       jnn.relu,
                       eqx.nn.Linear(8, 2, key=key2)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

x_key, y_key, model_key = jrandom.split(jrandom.PRNGKey(0), 3)
x, y = jrandom.normal(x_key, (100, 2)), jrandom.normal(y_key, (100, 2))
model = AnotherModule(model_key)

# Option 1: explicitly filter out anything that isn't JIT/grad-able.

@ft.partial(jax.jit, static_argnums=1)
@jax.grad
def loss(params, static, x, y):
    model = eqx.combine(params, static)
    pred_y = jax.vmap(model)(x)
    return jnp.mean((y - pred_y) ** 2)

params, static = eqx.partition(model, eqx.is_array)
loss(params, static, x, y)

# Option 2: use filtered transformations, which automates the above process for you.
# (Can be handy if you want to JIT/grad with respect to different things!)

@eqx.filter_jit
@eqx.filter_grad
def loss(model, x, y):
    pred_y = jax.vmap(model)(x)
    return jnp.mean((y - pred_y) ** 2)

loss(model, x, y)
```

Here, `params` and `static` are actually both instances of `AnotherModule`. `params` keeps just the attributes that are JAX arrays, and `static` keeps everything else. Then `combine` just merges the two PyTrees back together afterwards.

### Integrates smoothly with JAX

And that's it! That's pretty much all of Equinox.

Equinox introduces a powerful yet straightforward way to build neural networks, without introducing lots of new notions or tieing you into a framework.

Equinox is all just regular JAX -- PyTrees and transformations. Together, these two pieces allow us to specify complex models in JAX-friendly ways.

## Examples

- [`build_model.py`](./examples/build_model.py) builds an MLP from scratch, demonstrating the easy parameterised-functions-as-data approach that Equinox introduces. We'll then pass it into higher-order functions like JIT and grad. Overall we produce models using a familiar class-based syntax, that are also functional and integrate directly with JAX's JIT/autograd.

- [`filtered_transformations.py`](./examples/filtered_transformations.py) introduces `equinox.filter_jit` and `equinox.filter_grad`. These will be used to select the parameters of an MLP and train them.
 
- [`frozen_layer.py`](./examples/frozen_layer.py) demonstrates how this approach really shines: some of the parameters will be trained, some of them will be frozen, but *all* of them will be efficiently JIT-traced.

- [`train_rnn.py`](./examples/train_rnn.py) trains an RNN on a toy clockwise/anticlockwise spiral classification problem.

- [`modules_to_initapply.py`](./examples/modules_to_initapply.py) demonstrates how to use Equinox in an init/apply-style way, which some JAX libraries have been built around. (e.g. Stax)

## Citation

If you find Equinox useful in academic work then please consider a citation:
```bibtex
@article{kidger2021equinox,
    author={Patrick Kidger and Cristian Garcia},
    title={{E}quinox: neural networks in {JAX} via callable {P}y{T}rees and filtered transformations},
    year={2021},
    journal={Differentiable Programming workshop at Neural Information Processing Systems 2021}
}
```

## API

### Full API list

```python
# Module                         # Neural networks
equinox.Module                   equinox.nn.Linear
                                 equinox.nn.Identity
# Filtering/combining            equinox.nn.Dropout
equinox.filter                   equinox.nn.GRUCell
equinox.partition                equinox.nn.LSTMCell
equinox.combine                  equinox.nn.Sequential
                                 equinox.nn.MLP
# Filtered transformations       
equinox.filter_jit               equinox.nn.Conv1d
equinox.filter_grad              equinox.nn.Conv2d
equinox.filter_value_and_grad    equinox.nn.Conv3d
                                 
# Filters                        # Utilities
equinox.is_array                 equinox.apply_updates
equinox.is_array_like            equinox.static_field
equinox.is_inexact_array         equinox.tree_at
equinox.is_inexact_array_like    equinox.tree_equal
```

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

In this case a default `__init__` method is provided, which just fills in these attributes with the argments passed: `MyModule(weight, bias, submodule)`. Alternatively you can provide an `__init__` method yourself. (For example to specify dimension sizes instead of raw weights.) By the end of `__init__`, every attribute must have been assigned.

```python
class AnotherModule(equinox.Module):
    weight: Any

    def __init__(self, input_size, output_size, key):
        self.weight = jax.random.normal(key, (output_size, input_size))
```

After initialisation then attributes cannot be modified: models are immutable as per functional programming. (Parameter updates are made by creating a new model, not by mutating parameters in-place; see for example [`train_rnn.py`](./examples/train_rnn.py).)

It is typical to also create some methods on the class. As `self` will be an input parameter -- treated as a PyTree -- then these methods will get access to the attributes of the instance. Defining `__call__` gives an easy way to define a forward pass for a model (although any method can be used, and no methods are special-cased):

```python
class LinearWithoutBias(equinox.Module):
    weight: Any

    def __call__(self, x):
        return self.weight @ x
```

If defining a method `meth`, then take care not to write `instance = MyModule(...); jax.jit(instance.meth)(...)`. (Or similarly with `jax.grad`, `equinox.filter_jit` etc.) This is because `instance.meth` is not a pure function as it already has the `self` parameter passed implicitly. Instead do
```python
@jax.jit
def func(instance, args):
    instance.meth(args)
    # Also use this pattern with instance(args) if you defined `__call__` instead of `meth`.
```

### Filtering/combining

Filtering can be used to organise the contents of PyTrees.

```python
equinox.filter(pytree, filter_spec, inverse=False, replace=None)
```
Filters out the leaves of a PyTree not satisfying a condition. Those not satisfying the condition are replaced with `replace`.

- `pytree` is any PyTree
- `filter_spec` is a PyTree whose structure should be a prefix of the structure of `pytree`. Each of its leaves should either be:
  - `True`, in which case the leaf or subtree is kept;
  - `False`, in which case the leaf or subtree is replaced with `replace`;
  - a callable `Leaf -> bool`, in which case this is evaluted on the leaf or mapped over the subtree, and the leaf kept or replaced as appropriate.
- `inverse` switches the truthy/falsey behaviour: falsey results are kept and truthy results are replaced.
- `replace` is what to replace any falsey leaves with. Defaults to `None`.

Returns a PyTree of the same structure as `pytree`.

An important special case is something like `equinox.filter(pytree, equinox.is_array)`. Then `equinox.is_array` is evaluted on all of `pytree`'s leaves, and each leaf then kept or replaced.

See also `equinox.combine` to reconstitute the PyTree again.

```python
equinox.partition(pytree, filter_spec, replace=None)
```
Equivalent to `filter(...), filter(..., inverse=True)`, but slightly more efficient.

```python
equinox.combine(*pytrees)
```

Every element of `pytrees` must be a PyTree of the same structure. The return value is also a PyTree of the same structure. Each leaf will be the first non-`None` leaf found in the corresponding leaves of `pytrees`, as they are iterated over. The intention is that this be used to undo a call to `equinox.filter` or `equinox.partition`.

### Filtered transformations

It's very common to need to filter just to handle JAX transformations. Equinox provides the following convenience wrappers.<br>
They're not designed to handle every edge case -- they're just a way to streamline the common cases. Use separate `equinox.filter`+`jax.jit` etc. if you need finer control.

```python
equinox.filter_jit(fun, *, filter_spec=is_array, **kwargs)
```
Wraps `jax.jit`.

- `fun` is a pure function to JIT compile.
- `filter_spec` is a PyTree whose structure should be a prefix of the structure of the inputs to `fun`. Each of its leaves should either be `True`, `False`, or a callable `Leaf -> bool`. It behaves exactly as the `filter_spec` argument to `equinox.filter`. Truthy values will be traced; falsey values will be held static. Specifically, if calling `fun(*args, **kwargs)`, then `filter_spec` must have a structure which is a prefix for `(args, kwargs)`.
- `**kwargs` are any other arguments to `jax.jit`.

An important special case is to pass a function as `filter_spec`, which will be applied to every leaf of every input. For example, `equinox.filter_jit(fun, equinox.is_array)`.

See also `equinox.is_array`, which is the default choice of `filter_spec`. This will trace every JAX array, and make every other argument static.

```python
equinox.filter_grad(fun, *, filter_spec=is_inexact_array, **kwargs)
```
Wraps `jax.grad`.

- `fun` is a pure function to differentiate.
- `filter_spec` is a PyTree whose structure should be a prefix of the structure of the **first** input to `fun`. Each of its leaves should either be `True`, `False`, or a callable `Leaf -> bool`. It behaves exactly as the `filter_spec` argument to `equinox.filter`. Truthy values will be differentiated; falsey values will not. Specifically, if calling `fun(x, *args, **kwargs)`, then `filter_spec` must have a structure which is a prefix for the structure of `x`.
- `**kwargs` are any other arguments to `jax.grad`.

An important special case is to pass a function as `filter_spec`, which will be applied to every leaf of the first input. For example, `equinox.filter_grad(fun, equinox.is_inexact_array)`.<br>

See also `equinox.is_inexact_array`, which is the default choice of `filter_spec`. This will differentiate all floating-point JAX arrays.

Note that as the returned gradients must have the same structure as the inputs, then all nondifferentiable components of the input PyTree will have gradient `None`. See `equinox.apply_updates` for a convenience to only apply non-`None` updates.

```python
equinox.filter_value_and_grad(fun, *, filter_spec=is_inexact_array, **kwargs)
```
Wraps `jax.value_and_grad`. Arguments are as `equinox.filter_grad`.

### Filters

Any function `Any -> bool` can be used as a filter. We provide some convenient common choices.

```python
equinox.is_array(element)
```
Returns `True` if `element` is a JAX array (not but a NumPy array).

```python
equinox.is_array_like(element)
```
Returns `True` if `element` is a JAX array, NumPy array, or a Python float/int/bool/complex.

```python
equinox.is_inexact_array(element)
```
Returns `True` if `element` is a floating point JAX array (but not a NumPy array).

```python
equinox.is_inexact_array_like(element)
```
Returns `True` if `element` is a floating point JAX array, floating point NumPy array, or a Python float or complex.

### Utilities

```python
equinox.apply_updates(model, updates)
```
Performs a training update to a model.
- `model` must be a PyTree;
- `updates` must be a PyTree with the same structure.

It essentially performs `jax.tree_map(lambda m, u: m + u, model, updates)` (or `optax.apply_upates(model, updates)`). However anywhere `updates` is `None` then no update is made at all, so as to handle nondifferentiable parts of `model`.

The returned value is the updated model. (`model` is not mutated in place, as is usual in JAX and functional programming.)

To produce `updates`, it is typical to take the gradients from the loss function, and then adjust them according to any standard optimiser; for example [Optax](https://github.com/deepmind/optax) provides `optax.sgd` or `optax.adam`.

```python
equinox.static_field(**kwargs)
```
This is a relatively advanced feature. Use it to mark one of the fields of a `Module` as being "static": that is, never differentiated, and always a `static_argnum` to JIT. Best used only if you control whatever will be assigned to that field. For example `equinox.nn.MLP` does *not* use this for its activation function, as in principle a learnt activation function could be passed.

Example:
```python
class MyModule(equinox.Module):
    value: list = equinox.static_field()
```

If any `**kwargs` are passed, then they will be forwarded on to `dataclasses.field`. (Recall that Equinox uses dataclasses as its modules, so general `dataclasses` behaviour should work as normal.)

```python
equinox.tree_at(where, pytree, replace=_sentinel, replace_fn=_sentinel)
```
Modifies an existing tree, and returns the modified tree. (Like `.at` for "in place modifications" of JAX arrays.)

- `where` is a callable `PyTree -> Leaf` or `PyTree -> Tuple[Leaf, ...]`. It should consume a PyTree of the same shape as `pytree`, and return the leaf or leaves that should be replaced. For example `where=lambda mlp: mlp.layers[-1].linear.weight`.
- `pytree` is the existing PyTree to modify.
- `replace` should either be a single element, or a tuple of the same length as returned by `where`. This specifies the replacements to make at the locations specified by `where`. Mutually exclusive with `replace_fn`.
- `replace_fn` should be a function `Leaf -> Any`. It will be called on every leaf replaced using `where`. The return value from `replace_fn` will be used in its place. Mutually exclusive with `replace`.

For example this can be used to help specify the weights of a model to train or not train:
```python
trainable = jax.tree_map(lambda _: False, model)
trainable = equinox.tree_at(lambda mlp: mlp.layers[-1].linear.weight, model, replace=True)
equinox.filter_grad(..., filter_spec=trainable)
```

```python
equinox.tree_equal(*pytrees)
```
Returns `True` if all PyTrees in the list are equal. All arrays must have the same shape, dtype, and values. JAX arrays and NumPy arrays are not considered equal.

### Neural network library

Equinox includes a small neural network library, mostly as a tech demo for how the rest of the library can be used. Its API is broadly modelled after PyTorch.

```python
equinox.nn.Linear(in_features, out_features, use_bias=True, *, key)(input)
equinox.nn.Identity(*args, **kwargs)(input)  # args and kwargs are ignored
equinox.nn.Dropout(p=0.5, deterministic=False)(input, *, key=None, deterministic=None)
equinox.nn.GRUCell(input_size, hidden_size, use_bias=True, *, key)(input, hidden)
equinox.nn.LSTMCell(input_size, hidden_size, use_bias=True, *, key)(input, hidden)
equinox.nn.Sequential(layers)(input, *, key=None)
equinox.nn.MLP(in_size, out_size, width_size, depth,
               activation=jax.nn.relu, final_activation=lambda x: x, *, key)(input)
```
These all behave in the way you expect. The `key` arguments are used to generate the random initial weights, or to generate randomness on the forward pass of stochastic layers like `Dropout`.

The `Dropout(deterministic=...)(deterministic=...)` options determines whether to have the layer act as the identity function, as is commonly done with dropout during inference time. The call-time `deterministic` takes precendence if it passed; otherwise the init-time `deterministic` is used. (Note that because models are PyTrees, you can modify the init-time `deterministic` flag using `equinox.tree_at`. This is perfectly fine, and might be handy if it's easier than using the call-time flag.)

The `MLP(final_activation=...)` option determines any final activation function to apply after the last layer. (In some cases it is desirable for this to be different to the activation used in the main part of the network.)
