<h1 align='center'>Equinox</h1>
<h2 align='center'>Filtered JIT/grad transformations in JAX => neural networks</h2>

Equinox brings more power to your JAX workflow.

### Filtered JIT and filtered grad

Equinox offers two main functions: `jitf` and `gradf`. These are thin wrappers around `jax.jit` and `jax.grad`. Instead of specifying e.g. `jit(static_argnums=...)` or `gradf(argnums=...)` manually, you instead specify a *filter function*, that looks at the input PyTrees and specifies True/False whether to JIT that argument (i.e. static vs trace it), or whether to differentiate with respect to each *leaf of each PyTree*.

This offers a powerful, fine-grained way to control JIT and autodifferentiation. For example:
- Annotate some layers of the model as being frozen, and only train the others.
- Build a complex model as a PyTree, mixing JAX arrays with boolean flags with arbitrary Python objects -- and then specify that the forward pass should be JIT traced with respect to all the arrays, and automatically have all the boolean flags and Python objects be treated as static arguments.

The key trick in both cases is that `jax.jit` and `jax.grad` can only specify whole *arguments* to differentiate or leave static -- whilst `jitf` and `gradf` can understand the components of each PyTree comprising these arguments.

### Modules

Augmenting this approach (but entirely optional), Equinox offers a straightforward syntax to specify PyTrees as classes. As classes can also have methods, then the PyTree structure can be used to encode parameters whilst the methods encode the forward pass of a model.

Equinox includes a tiny example `nn` library (implementing just a few things like Linear, MLP, Sequential) demonstrating how to use Modules to build your own models.

### Examples

TODO
