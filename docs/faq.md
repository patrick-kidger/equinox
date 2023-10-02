# FAQ

## Optax throwing a `TypeError`.

Probably you're writing code that looks like
```python
optim = optax.adam(learning_rate)
optim.init(model)
```
and getting an error that looks like
```
TypeError: zeros_like requires ndarray or scalar arguments, got <class 'jax._src.custom_derivatives.custom_jvp'> at position 0.
```

This can be fixed by doing
```python
optim.init(eqx.filter(model, eqx.is_inexact_array))
```
which after a little thought should make sense: Optax can only optimise floating-point JAX arrays. It's not meaningful to ask Optax to optimise whichever other arbitrary Python objects may be a part of your model. (e.g. the activation function of an `eqx.nn.MLP`).

## How are batch dimensions handled?

All layers in `equinox.nn` are defined to operate on single batch elements, not a whole batch.

To act on a batch, use [`jax.vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax.vmap). This maps arbitrary JAX operations -- including any Equinox module -- over additional dimensions.

For example if `x` is an array/tensor of shape `(batch_size, input_size)`, then the following PyTorch code:

```python
import torch
linear = torch.nn.Linear(input_size, output_size)

y = linear(x)
```

is equivalent to the following Equinox code:
```python
import jax
import equinox as eqx
key = jax.random.PRNGKey(seed=0)
linear = eqx.nn.Linear(input_size, output_size, key=key)

y = jax.vmap(linear)(x)
```

## How to share a layer across two parts of a model?

Use [`equinox.nn.Shared`][] to tie together multiple nodes (layers, weights, ...) in a PyTree.

In particular, *don't* do something like this:
```python
# Buggy code!
class Module(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(...):
        shared_linear = eqx.nn.Linear(...)
        self.linear1 = shared_linear
        self.linear2 = shared_linear
```
as this is used to accomplish something different: this creates two separate layers, that are initialised with the same values for their parameters. After making some gradient updates, you'll find that `self.linear1` and `self.linear2` are now different.

The reason for this is that in Equinox+JAX, models are Py*Trees*, not DAGs. (Directed acyclic graphs.) JAX follows a functional-programming-like style, in which the *identity* of an object (whether tha be a layer, a weight, or whatever) doesn't matter. Only its *value* matters. (This is known as referential transparency.)

 See also the [`equinox.tree_check`][] function, which can be ran on a model to check if you have duplicate nodes.

## My model is slow...

#### ...to train.

Make sure you have JIT covering all JAX operations.

Most autodifferentiable programs will have a "numerical bit" (e.g. a training step for your model) and a "normal programming bit" (e.g. saving models to disk). JAX makes this difference explicit. All the numerical work should go inside a single big JIT region, within which all numerical operations are compiled.

See [the RNN example](https://docs.kidger.site/equinox/examples/train_rnn/) as an example of good practice. The whole `make_step` function is JIT compiled in one go.

Common mistakes are to put `jax.jit`/`eqx.filter_jit` on just your loss function, and leave out either (a) computing gradients or (b) applying updates with `eqx.apply_updates`.

#### ...to compile.

95% of the time, it's because you've done something like this:
```python
@jax.jit
def f(x):
    for i in range(100):
        x = my_complicated_function(x)
    return x
```
When JAX traces through this, it can't see the `for` loop. (`jax.jit` replaces the `x` argument with a tracer object that records everything that happens to it -- and this effectively unrolls the loop.) As a result you'll get 100 independent copies of `my_complicated_function`, which all get compiled separately.

In this case, a `jax.lax.scan` is probably what you want. Likewise it's usually also preferable to rewrite even simple stuff like
```python
x2 = f(x1)
x3 = f(x2)
```
as a little length-2 scan.

## TypeError: not a valid JAX type.

You might be getting an error like
```
TypeError: Argument '<function ...>' of type <class 'function'> is not a valid JAX type.
```
Example:
```python3
import jax
import equinox as eqx

def loss_fn(model, x, y):
    return ((model(x) - y) ** 2).mean()

model = eqx.nn.Lambda(lambda x: x)
model = eqx.nn.MLP(2, 2, 2, 2, key=jax.random.PRNGKey(0))

x = jax.numpy.arange(2)
y = x * x

try:
    jax.jit(loss_fn)(model, x, y) # error
except TypeError as e:
    print(e)

eqx.filter_jit(loss_fn)(model, x, y) # ok
```

This error happens because a model, when treated as a PyTree, may have leaves that are not JAX types (such as functions). It only makes sense to trace arrays. Filtering is used to handle this.

Instead of [`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html), use [`equinox.filter_jit`][]. Likewise for [other transformations](https://docs.kidger.site/equinox/api/filtering/transformations).

## How to mark arrays as non-trainable? (Like PyTorch's buffers?)

This can be done by using `jax.lax.stop_gradient`:
```python
class Model(eqx.Module):
    buffer: Array
    param: Array

    def __call__(self, x):
        return self.param * x + jax.lax.stop_gradient(self.buffer)
```

## I think my function is being recompiled each time it is run.

You can check each time your function is compiled by adding a print statement:
```python
@eqx.filter_jit
def your_function(x, y, z):
    print("Compiling!")
    ...  # rest of your code here
```
JAX calls your function each time it needs to compile it. Afterwards, it never actually calls it -- indeed it doesn't use Python at all! Instead, it uses its compiled copy of your function, which only performs array operations. Thus, a print statement is an easy way to check each time JAX is compiling your function.

A function will be recompiled every time the shape or dtype of its arrays changes, or if any of its static (non-array) inputs change (as measured by `__eq__`).

If you want to check which argument is causing an undesired recompilation, then this can be done by checking each argument in turn:
```python
@eqx.filter_jit
def check_arg(arg, name):
    print(f"Argument {name} is triggering a compile.")


for step, (x, y, z) in enumerate(...):  # e.g. a training loop
    print(f"Step is {step}")
    check_arg(x, "x")
    check_arg(y, "y")
    check_arg(z, "z")
    your_function(x, y, z)
```
for which you'll often see output like
```
Step is 0
Argument x is triggering a compile.
Argument y is triggering a compile.
Argument z is triggering a compile.
Step is 1
Argument y is triggering a compile.
Step is 2
Argument y is triggering a compile.
...
```
On the very first step, none of the arguments have been seen before, so they all trigger a compile. On later steps, just the problematic argument will trigger a recompilation of `check_arg` -- this will be the one that is triggering a recompilation of `your_function` as well!

## How does Equinox compare to...?

#### ...PyTorch?

JAX+Equinox is usually faster than PyTorch (a stronger JIT compiler), and more featureful (e.g. supporting jit-of-vmap, forward-mode autolinearisation, and autoparallelism).

For those doing scientific computing or scientific ML, then JAX+Equinox also has a much stronger ecosystem. For example, PyTorch no longer has a library for solving differential equations (torchdiffeq is unmaintained). Meanwhile, JAX has [Diffrax](https://github.com/patrick-kidger/diffrax).

Both JAX+Equinox and PyTorch are roughly equally easy to use. PyTorch tends to be a easier for new users (e.g. it's closer to being "Python as normal", and there's less functional programming), whilst JAX+Equinox generally supports advanced use-cases more cleanly (e.g. PyTorch has multiple JIT compilers each with their own quirks -- `torch.{fx, jit.script, jit.trace, compile, _dynamo, ...}` -- whilst JAX+Equinox just has the one).

PyTorch is older, and enjoys broader adoption -- it's generally easier to find developers for PyTorch, or off-the-shelf model implementations using it.

#### ...Keras?

These are two very different libraries, with very different target audiences. Keras is great for plug-and-play building of models -- it's often compared to using Lego. This makes it a convenient framework for standing up neural networks quickly. Equinox is much lower level: it tries to support more general use-cases (e.g. its downstream scientific ecosystem), but usually requires greater proficiency with numerical computing / software development / machine learning.

#### ...Flax?

- Flax introduces multiple new abstractions (`flax.linen.Module`, `flax.linen.Variable`, `Module.setup` vs `flax.linen.compact`, `flax.struct.dataclass`, etc.). Equinox tries to avoid adding new abstractions to core JAX; everything is always just a PyTree.
- Flax is a DSL: it is generally incompatible with non-Flax code, and requires using wrapped `flax.linen.{vmap, scan, ...}` rather than the native `jax.{vmap, ...}`. In contrast, Equinox allows you to use native JAX operations and aims to be compatible with arbitrary JAX code.
- Bound methods of `eqx.Module` are just PyTrees. In Flax this isn't the case -- passing around bound methods will either result in errors or recompilations, depending what you do. Likewise, `eqx.Module` handles inheritance correctly, including propagating metadata like docstrings. The equivalent `flax.struct.dataclass` silently misbehaves. Overall Equinox seems to have fewer footguns.
- Equinox offers several advanced features (like [runtime errors](../api/errors/) or [PyTree manipulation](../api/manipulation/#equinox.tree_at)) not found in other libraries.

See also the [Equinox paper](https://arxiv.org/abs/2111.00254).

#### ...Julia?

The Julia ecosystem has [historically been buggy](https://kidger.site/thoughts/jax-vs-julia/).

At time of writing, Julia does not yet have a robust autodifferentiation system. For example, it has multiple competing implementations -- both Diffractor.jl and ForwardDiff.jl for forward-mode autodifferentiation, and all of Tracker.jl, Zygote.jl, Enzyme.jl, ReverseDiff.jl for reverse-mode autodifferentiation. It does not yet support higher-order autodifferentiation robustly. In contrast, JAX+Equinox use a single strong autodifferentiation system.

However, note that JAX+Equinox don't try to offer a completely general programming model: they are optimised for arrays and linear algebra. (Essentially, the sorts of things you use NumPy for.) They're not designed for e.g. a branch-and-bound combinatorial optimisation algorithm, and for these purposes Julia will be superior.

Julia is often a small amount faster on microbenchmarks on CPUs. JAX+Equinox supports running on TPUs, whilst Julia generally does not.

**You're obviously biased! Are the above comparisons fair?**

Seriously, we think they're fair! Nonetheless all of the above approaches have their adherents, so it seems like all of these approaches are doing something right. So if you're already happily using one of them for your current project... then keep using them. (Don't rewrite things for no reason.) But conversely, we'd invite you to try Equinox for your next project. :)

For what it's worth, if you have the time to learn (e.g. you're a grad student), then we'd strongly recommend trying all of the above. All of these libraries have made substantial innovations, and have all made substantially moved the numerical computing space forward. Equinox deliberately takes inspiration from them!
