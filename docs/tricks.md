# Tricks (ensembles, surgery, custom initialisations, ...)

Here are a few advanced tricks.

## Performing model surgery

This can be done using [`equinox.tree_at`][]. For example, here's how to replace the final layer in a [`equinox.nn.MLP`][].

```python
mlp = eqx.nn.MLP(...)
new_final_layer = eqx.nn.Linear(...)
where = lambda m: m.layers[-1]
new_mlp = eqx.tree_at(where, mlp, new_final_layer)
```

This is a nice example of the simplicity of Equinox's model-as-PyTree approach.

## Custom parameter initialisation

You might want to initialise your parameters in some nonstandard way.

This can be done as a special case of model surgery.

For example, here is how to change the `weight` of a linear layer:
```python
linear = eqx.nn.Linear(...)
new_weight = jax.random.normal(...)
where = lambda l: l.weight
new_linear = eqx.tree_at(where, linear, new_weight)
```

And here is how to replace the `weight` of every linear layer in some arbitrary model. In this example we draw samples from a truncated normal distribution.
```python
def trunc_init(weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
  out, in_ = weight.shape
  stddev = math.sqrt(1 / in_)
  return stddev * jax.random.truncated_normal(key, shape=(out, in_), lower=-2, upper=2)

def init_linear_weight(model, init_fn, key):
  is_linear = lambda x: isinstance(x, eqx.nn.Linear)
  get_weights = lambda m: [x.weight
                           for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                           if is_linear(x)]
  weights = get_weights(model)
  new_weights = [init_fn(weight, subkey)
                 for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
  new_model = eqx.tree_at(get_weights, model, new_weights)
  return new_model

model = ... # any PyTree
key = jax.random.PRNGKey(...)
new_model = init_linear_weight(model, trunc_init, key)
```

The above code probably seems a bit complicated! But it also makes it possible to build large, complicated models *without* needing to thread through additional `layer_2_weight_init=...` arguments everywhere.

## Custom per-parameter behaviour

For example, making a weight matrix be symmetric (similar to [`torch.nn.utils.parameterize`](https://pytorch.org/tutorials/intermediate/parametrizations.html)), or for a particular parameter not to receive gradient updates.

This can be done by wrapping the parameter in a custom module that gives the desired behaviour, and then applying this behaviour after you've crossed your JIT and grad API boundaries. Once again we'll use `eqx.tree_at` to perform the relevant model surgery.

For example, here is how to make every linear layer have a symmetric weight matrix:
```python
# Library code

class Symmetric(eqx.Module):
  matrix: jax.Array

  def get(self):
    return 0.5 * (self.matrix + self.matrix.T)

def is_symmetric(x):
    return isinstance(x, Symmetric)

def maybe_symmetric(x):
    if is_symmetric(x):
        return x.get()
    else:
        return x  # leave everything else unchanged

def resolve_symmetric(model):
    return jax.tree_util.tree_map(maybe_symmetric, model, is_leaf=is_symmetric)

# User code

model = ...   # any PyTree
is_linear = lambda x: isinstance(x, eqx.nn.Linear)
get_weights = lambda m: [x.weight
                         for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                         if is_linear(x)]
symmetric_model = eqx.tree_at(get_weights, model, replace_fn=Symmetric)

@eqx.filter_grad
def loss_fn(model, x, y):
  model = resolve_symmetric(model)
  pred_y = jax.vmap(model)(x)
  return jnp.sum((y - pred_y)**2)

grads = loss_fn(symmetric_model, ...)
```

## Ensembling

Model ensembles can be produced by `vmap`'ing model initialisation. For example, here's how to create an ensemble of eight MLPs:

```python
key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 8)

# Create an ensemble of models
@eqx.filter_vmap
def make_ensemble(key):
    return eqx.nn.MLP(2, 2, 2, 2, key=key)

mlp_ensemble = make_ensemble(keys)

# Evaluate each member of the ensemble on the same data
@eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
def evaluate_ensemble(model, x):
    return model(x)

evaluate_ensemble(mlp_ensemble, jax.random.normal(key, (2,)))

# Evaluate each member of the ensemble on different data
@eqx.filter_vmap
def evaluate_per_ensemble(model, x):
    return model(x)

evaluate_per_ensemble(mlp_ensemble, jax.random.normal(key, (8, 2)))
```

Here, `make_ensemble` works because [`equinox.nn.MLP`][] is a PyTree, and so it is a valid output from a `filter_vmap`. This PyTree includes some JAX arrays (the weights and biases) and some non-JAX-arrays (e.g. activation functions). `filter_vmap` will vectorise the JAX arrays (with separate weights for each member of the ensemble) whilst leaving the non-JAX-arrays alone.

Note that as the weights in `mlp_ensemble` now have a leading batch dimension -- that the weights of `eqx.nn.MLP` instances do not typically have -- then it cannot be called directly. It must instead be passed back into a vectorised region to be called.

## Low-overhead training loops

Quite a common pattern is to have a training loop that looks like this:
```python
@eqx.filter_jit
def make_step(model, opt_state, x, y):
    ...
    return update_model, update_opt_state

model = ...
opt_state = ...

for batch_x, batch_y in dataloader(...):
    model, opt_state = make_step(model, opt_state, batch_x, batch_y)
```

(See the [Train RNN](https://docs.kidger.site/equinox/examples/train_rnn/) example for an example.)

Here, the PyTree structure of `model` and `opt_state` is flattened and unflattened when entering and exiting the JIT region, which incurs some small amount of overhead. This isn't actually necessary: we can "cancel out" the flattening done when entering the JIT region with unflattening done when exiting the JIT region.

This can be done by rewriting the above loop as:
```python
@eqx.filter_jit
def make_step(flat_model, flat_opt_state, x, y):
    model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
    opt_state = jax.tree_util.tree_unflatten(treedef_opt_state, flat_opt_state)
    ...
    flat_update_model = jax.tree_util.tree_leaves(update_model)
    flat_update_opt_state = jax.tree_util.tree_leaves(update_opt_state)
    return flat_update_model, flat_update_opt_state

model = ...
opt_state = ...
flat_model, treedef_model = jax.tree_util.tree_flatten(model)
flat_opt_state, treedef_opt_state = jax.tree_util.tree_flatten(opt_state)

for batch_x, batch_y in dataloader(...):
    flat_model, flat_opt_state = make_step(flat_model, flat_opt_state, batch_x, batch_y)
model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
```

This trick can be useful if the numerical computations inside your JIT region are very fast (and the overhead dominates), or if you are using models which are slow to flatten or unflatten (e.g. some external libraries).

## Improve compilation speed with scan-over-layers

It is relatively common to have multiple layers that all have the same PyTree structure (same input size, same output size, weights all of the same size, etc.), and which are all called sequentially. For example, the linear layers in an MLP, or the attention layers in a Transformer.

In this case, it is possible to improve compilation speed, by using `vmap` to create multiple layers all at the same time, and then iterating over the extra dimension using `jax.lax.scan`. See the following toy example, which chains ten MLPs together one-after-the-other.

```python
import equinox as eqx
import jax.lax as lax
import jax.random as jr

class TenMLPs(eqx.Module):
    mlps: eqx.nn.MLP
  
    def __init__(self, size, key):
        keys = jr.split(key, 10)
        make_mlp = lambda k: eqx.nn.MLP(size, size, size, 1, key=k)
        self.mlps = eqx.filter_vmap(make_mlp)(keys)
        # Store a single `eqx.nn.MLP` object, with each of its parameters (JAX
        # arrays) having an extra length-10 dimension at the start.
        #
        # This works because `make_mlp` is just a function returning a PyTree,
        # and `eqx.filter_vmap` operates on functions between PyTrees.
        #
        # `self.mlps` should not be called directly: it should only be
        # used inside a `vmap` or `scan`, to first unpeel the extra dimension.
  
    def __call__(self, x):
        dynamic_mlps, static_mlps = eqx.partition(self.mlps, eqx.is_array)
        # `dynamic_mlps` has all the parameters, with their extra length-10
        # dimension.
        # `static_mlps` has all the non-parameters, e.g. activation functions,
        # which are shared across all layers.
    
        def f(_x, _dynamic_mlp):
            mlp = eqx.combine(_dynamic_mlp, static_mlps)
            return mlp(x), None
    
        out, _ = lax.scan(f, x, dynamic_mlps)
        return out

size = 3
key = jr.PRNGKey(0)
model_key, data_key = jr.split(key, 2)
model = TenMLPs(size, model_key)
x = jr.normal(data_key, (size,))
model(x)
```

Writing out each layer manually (`y = self.mlp1(x); z = self.mlp2(y); ...`) means that JAX needs to compile each MLP separately. However the compiler knows that every iteration of a `jax.lax.scan` has the same structure, so this trick allows us to compile our models faster.
