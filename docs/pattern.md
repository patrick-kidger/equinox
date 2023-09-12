# The abstract/final design pattern

The following is a very useful design pattern. It's not mandatory, but it comes very strongly recommended, as it's designed to produce very readable code. This is also the pattern used throughout the Equinox ecosystem -- [Lineax](https://github.com/google/lineax), [Diffrax](https://github.com/patrick-kidger/diffrax) etc. -- and as such Equinox offers a lot of tools to make this approach feel particularly powerful.

!!! Tip "The abstract/final design pattern"

    Due to `eqx.Module`, we tend to create a lot of classes. We're going to enforce the restriction that every class be precisely one of:  

    (a) abstract -- that is, it can be subclassed, but not instantiated;  
    (b) final -- that is, it can be instantiated, but not subclassed.

    Moreover, abstract classes shouldn't define `__init__` methods, nor should they define attributes (other than those marked with [`equinox.AbstractVar`][] or [`equinox.AbstractClassVar`][]).
    
    Finally, we should never re-override a method. Once a subclass implements a method, that's it.

This idea is very simple. Now, let's take a deep dive on why this is such a neat pattern, and how Equinox offers special tools to support this.

## Level 1: Abstract base classes (ABCs) as interfaces

When following the above, we tend to write code that looks like the following:
```python
class AbstractOptimiser(eqx.Module):
    @abc.abstractmethod
    def init(self, params):
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, params, grads, state):
        raise NotImplementedError

class Adam(AbstractOptimiser):
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.999

    def init(self, params):
        ...  # some implementation
        return initial_state

    def update(self, params, grads, state):
        ...  # some implementation
        return new_params, new_state

@eqx.filter_jit
def make_step(params, data, opt_state, optimiser: AbstractOptimiser):
    grads = eqx.filter_grad(compute_loss)(params, data)
    new_params, new_opt_state = optimiser.update(params, grads, opt_state)
    return new_params, new_opt_state

def train(params, dataloader, optimiser: AbstractOptimiser):
    opt_state = optimiser.init(params)
    for data in dataloader:
        params, opt_state = make_step(params, data, opt_state, optimiser)
    return params

params = ...  # some model
dataloader = ...  # some dataloader
optimiser = Adam(learning_rate=3e-4)
train(params, dataloader, optimiser)
```

Hopefully the above is indeed easy to read! The `AbstractOptimiser` defines an interface using `init` and `update`. Subsequently, we can write our `train` and `make_step` functions without needing to worry exactly which optimiser we have been passed.

For readability, it's worth following the convention that all abstract classes begin with the word "Abstract".

This idea is very common. Indeed Python has a whole module, [abc](https://docs.python.org/3/library/abc.html), for declaring such `abc.abstractmethod`s. And here we see our first example of Equinox making this approach easy for you: Equinox modules automatically inherit from `abc.ABC`, so you don't need to do that yourself.

## Level 2: intermediate ABCS, abstract attributes, and concrete-`__init__`-only

Now here's a natural extension to the above: intermediate ABCs, that introduce partial implementations.

```python
class AbstractInterpolation(eqx.Module):
    @abc.abstractmethod
    def __call__(self, x: Array) -> Array:
        raise NotImplementedError


class AbstractPolynomialInterpolation(AbstractInterpolation)
    coeffs: AbstractVar[Array]

    def degree(self) -> int:
        return len(self.coeffs)

    def __call__(self, x: Array) -> Array:
        return jnp.polyval(self.coeffs, x)


class CubicInterpolation(AbstractPolynomialInterpolation):
    coeffs: Array

    def __init__(self, ts: Array, xs: Array):
        self.coeffs = ...  # some implementation
```
in this case, the intermediate ABC `AbstractPolynomialInterpolation` implements the `__call__` method. However, it isn't yet a concrete (non-abstract) class, as it introduces a new abstract variable `coeffs` -- we need to wait until `CubicInterpolation` for that to be defined.

We also see the use of [`equinox.AbstractVar`][] -- this is an Equinox-specific extension to the `abc` module, making it possible to define abstract attributes. (There is also [`equinox.AbstractClassVar`][], to define abstract class attributes.) This is another example of Equinox being designed to make this design pattern easy.

As a final more subtle point, note that `AbstractPolynomialInterpolation` did **not** provide an `__init__` method! We could have written this instead:
```python
class AbstractPolynomialInterpolation(AbstractInterpolation)
    coeffs: Array

    def __init__(self, coeffs: Array):
        self.coeffs = coeffs

    def degree(self) -> int:
        return len(self.coeffs)

    def __call__(self, x: Array) -> Array:
        return jnp.polyval(self.coeffs, x)


class CubicInterpolation(AbstractPolynomialInterpolation):
    def __init__(self, ts: Array, xs: Array):
        coeffs = ...  # some implementation
        super().__init__(coeffs)
```
but once you have multiple classes involved, then splitting up your initialisation like this very quickly becomes far less readable. (And a reliable source of bugs.) Overall, we mandate that `__init__` methods and (non-abstract) fields may only be defined on concrete classes. Equinox supports checking this via a `strict=True` flag, passes as `class Foo(eqx.Module, strict=True)`.

## Level 3: implement methods precisely once, and concrete-means-final

Our "concrete `__init__` only" rule means that `__init__` is defined precisely once, is never overridden, and we never call `super().__init__`. Why stop there -- perhaps we should enforce that we never override *any* method?

In practice, we argue that's a good idea! This rule means that when you see code like:
```python
def foo(interp: AbstractPolynomialInterpolation)
    ... = interp(...)
```
you know that it is calling `AbstractPolynomialInterpolation.__call__`, and not anything else. This is great for code readability. Once again, this may be checked via a `strict=True` flag, passed as `class Foo(eqx.Module, strict=True)`.

If we assume this, then we now find ourselves arriving at a conclusion: concrete means final. That is, once we have a concrete class (every abstract method/attribute defined in our ABCs is now overriden with an implementation, so we can instantiate this class), then it is now final (we're not allowed to re-override things, so subclassing is pointless).

What about when you have an existing concrete class that you want to tweak just-a-little-bit? In this case, prefer composition over inheritance. Write a wrapper that forwards each method as appropriate.

## Level 4: `__check_init__`

It's pretty common to want to validate that certain invariants hold, even in abstract base classes. For this, we have the `__check_init__` method:
```python
class AbstractPolynomialInterpolation(AbstractInterpolation)
    coeffs: AbstractVar[Array]

    def __check_init__(self):
        if not jnp.issubdtype(self.coeffs.dtype, jnp.floating):
            raise ValueError("Coefficients must be floating-point!")

    ...
```
This method is something that Equinox will look for, and if present it will be ran after initialisation. This is again an Equinox-specific extension designed to support this design pattern.

See [checking invariants](../api/module/advanced_fields/#checking-invariants) for more details.

## Extensions and FAQ

**Does `super()` ever get used at all?**

Ideally, no! This design pattern means that you should never need to write `super()` at all.

**Does this pattern work with multiple inheritance?**

Yup. Nothing changes on that front. Take a look at [Diffrax](https://github.com/patrick-kidger/diffrax), for example. Simplifying a little, this happily has diamond inheritance patterns that look like:
```python
class AbstractSolver(eqx.Module):
    @abc.abstractmethod
    def step(...): ...

class AbstractAdaptiveSolver(AbstractSolver):
    ...

class AbstractImplicitSolver(AbstractSolver):
    root_finder: eqx.AbstractVar[AbstractRootFinder]

class ImplicitEuler(AbstractAdaptiveSolver, AbstractImplicitSolver):
    root_finder: AbstractRootFinder = Newton()

    def step(...): ...
```

**That's a lot of `Abstract`s**

Yes.

**What about co-operative multiple inheritance?**

If you're a Python nerd, you'll now be wondering about co-operative multiple inheritance, and the ubiquitous use of `super()`.

The TL;DR of this is that almost no-one ever uses this properly, and the abstract+final pattern is intended as a direct alternative. One sees a lot of code that looks like this:
```python
class A:
    def __init__(self, x):
        self.x = x
        # Not calling super().__init__, because the superclass is just `object`, right??

class AA:
    def __init__(...): ...

class B(A, AA):
    pass

B() # bug!
```
And in this case `B()` calls `A.__init__` which then fails to call `AA.__init__`. Bug! Co-operative multiple inheritance only works if everyone, well, co-operates.

Besides that, when you call `super().__init__`, then because `super()` could be pointing at almost any class at all, then in general it's essentially impossible to pass it the right arguments. "Only use keyword arguments" is the closest to a resolution that this issue has, and it's still fragile.

**These ideas have appeared in &lt;XYZ language&gt;?**

Yup! Variants of this design pattern are very common, especially in modern languages like Julia/Rust/etc.etc. There's not really anything new here -- but because Equinox is specifically designed to support this design pattern, this guide is intended as a self-contained reference to it.
