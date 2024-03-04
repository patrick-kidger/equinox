# The abstract/final pattern

The following is a useful design pattern. It's not mandatory, but it does come recommended, as it's designed to produce very readable code. This is also the pattern used throughout the Equinox ecosystem -- [Lineax](https://github.com/google/lineax), [Diffrax](https://github.com/patrick-kidger/diffrax) etc. (And most of `equinox.nn` -- although not all of it, for backward compatibility reasons!)

!!! Tip "The abstract/final design pattern"

    Every subclass of `eqx.Module` must be either

    (a) abstract (it can be subclassed, but not instantiated); or  
    (b) final (it can be instantiated, but not subclassed).

    Unless they're abstract, then methods and attributes must never be overridden. Once they've been implemented, that's it.

    The `__init__` method, and all dataclass fields, must all be defined in one class. No defining fields in multiple parts of a class hierarchy.

Collectively, these rules serve to avoid all the worst ways in which you might spaghetti your code. Abstract classes are used to define interfaces or partial implementations. Final classes are what get passed around at runtime. No overriding (of methods or of non-abstract classes) means there are no surprises with exactly what is being called. Keeping all fields together means that initialisation is readable.

Equinox will enforce these rules when subclassing with `strict=True`, e.g.
```python
class Foo(eqx.Module, strict=True):
    x: int
# ...so far so good...

class Bar(Foo, strict=True):
    y: int
# ...error raised here: can't define fields in two different classes.
```
(In addition, `strict=True` checks two other things: that all abstract classes have a name that starts with `"Abstract"`, and that modules aren't mixed with non-module classes.)

!!! faq "FAQ"

    Some quick FAQs:

    - If we want to subclass to override just one method, just to tweak it a little bit -- this can be done by wrapping the original class, not subclassing it.
    - In practice we typically define all fields, and the `__init__` method, on the final class that we instantiate. Every so often we could do them on an abstract class though, in which case the final subclasses will only be implementing methods.
    - You should never use the `hasattr` builtin. You probably meant to declare this attribute/method on an abstract base class.
    - To access an attribute on an abstract class, it can be declared using [`equinox.AbstractVar`][]; the eventual final subclass must provide this as a field or property.

    And for the CS nerds:

    - This is going all-in on nominal subtyping, not structural/duck subtyping.
    - Yes, this design pattern looks a lot like Rust/Julia/something. For example the abstract-or-final rule is inspired by Julia's type system, and the no-overriding is inspired by how many languages (including Rust) approach the orphan problem.
    - This has nothing to do with object-oriented (OO) programming, and everything to do with type theory. OO is about mutating classes through their methods, and as Equinox modules are immutable then we never do that here. (It is true that a lot of OO codebases could benefit from rules like these, though.)

---

The above is the really important bit. For those who are curious to know more, here's how we arrive at the above design pattern.

## Level 1: Abstract base classes (ABCs) as interfaces

Let's start off with something very standard: using ABCs to define interfaces. Here's an example.

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

Hopefully the above is indeed easy to read. The `AbstractOptimiser` defines an interface using `init(params)` and `update(params, grad, state)`. (And we should also add type annotations to their arguments, actually.)

Subsequently, the `train` and `make_step` functions can be written without needing to know exactly which optimiser has been passed. (We can later implement some other optimiser and use that in the same place.)

For readability, it's worth following the convention that all abstract classes begin with the word "Abstract".

The above is very common. Indeed Python has a whole module, [abc](https://docs.python.org/3/library/abc.html), for declaring such `abc.abstractmethod`s. This is also our first example of Equinox making this approach easy: Equinox modules automatically inherit from `abc.ABC`, so you don't need to do that yourself.

## Level 2: intermediate ABCS, abstract attributes, and `__init__`-only-once

Now let's move on to a natural extension to the above: intermediate ABCs, that introduce partial implementations.

```python
class AbstractInterpolation(eqx.Module):
    @abc.abstractmethod
    def __call__(self, x: Array) -> Array:
        raise NotImplementedError


class AbstractPolynomialInterpolation(AbstractInterpolation)
    coeffs: eqx.AbstractVar[Array]

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

Using an abstract attribute ([`equinox.AbstractVar`][]) here means that we can write `self.coeffs` inside `degree` and `__call__`, and know that this is safe. Unless all abstract attributes are defined then Equinox won't allow us to instantiate the class.

Why didn't we just define `AbstractPolynomialInterpolation.coeffs` as a concrete field? (Just `coeffs: Array`.) Indeed we could have written this:
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
but this is now much less readable: we've split up initialisation across two different classes. This is a reliable source of bugs. Thus we arrive at the rule that all fields, and the `__init__` method, should all be defined together. Using `strict=True` when subclassing (e.g. `class CubicInterpolation(AbstractPolynomialInterpolation, strict=True)`) will mean that Equinox checks for this, and raises an error if need be.

## Level 3: implement methods precisely once, and concrete-means-final

Our "`__init__` only once" rule means that `__init__` is defined precisely once, is never overridden, and we never call `super().__init__`. Why stop there: perhaps we should enforce that we never override *any* method?

In practice, we argue that's a good idea! This rule means that when you see code like:
```python
def foo(interp: AbstractPolynomialInterpolation)
    ... = interp.degree()
```
you know that it is calling precisely `AbstractPolynomialInterpolation.degree`, and not an override in some subclass. This is excellent for code readability. Thus we get the rule that no method should be overriden. (And this rule will also be checked via the `strict=True` flag.)

If we assume this, then we now find ourselves arriving at a conclusion: concrete means final. That is, once we have a concrete class (every abstract method/attribute defined in our ABCs is now overriden with an implementation, so we can instantiate this class), then it is now final (we're not allowed to re-override things, so subclassing is pointless). This is how we arrive at the abstract-or-final rule itself!

What about when you have an existing concrete class that you want to tweak just-a-little-bit? In this case, prefer composition over inheritance. Write a wrapper that forwards each method as appropriate. This is just as expressive, and means we keep these readable type-safe rules.

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
This method is something that Equinox will look for, and if present it will be ran after initialisation. This is an Equinox-specific extension designed to support this design pattern.

See [checking invariants](../api/module/advanced_fields/#checking-invariants) for more details.

## Extensions and FAQ

**Does this pattern work with multiple inheritance?**

Yes. For example, here's a diamond inheritance pattern (for building a differential equation solver):
```python
class AbstractSolver(eqx.Module):
    @abc.abstractmethod
    def step(...):
        raise NotImplementedError

class AbstractAdaptiveSolver(AbstractSolver):
    tolerance: eqx.AbstractVar[float]

class AbstractImplicitSolver(AbstractSolver):
    root_finder: eqx.AbstractVar[AbstractRootFinder]

class ImplicitEuler(AbstractAdaptiveSolver, AbstractImplicitSolver):
    tolerance: float
    root_finder: AbstractRootFinder = Newton()

    def step(...):
        ...  # some implementation

solver = ImplicitEuler(tolerance=1e-3)  # this can be instantiated
```

**That's a lot of `Abstract`s**

Yes.

**Does `super()` ever get used at all?**

No. This design pattern means that you should never need to write `super()` at all.

**What about co-operative multiple inheritance?**

If you're a Python nerd, you'll now be wondering about co-operative multiple inheritance, which specifies using `super()` ubiquitously.

The TL;DR of this is that almost no-one ever uses this properly, and the abstract+final pattern is intended as a direct alternative. One sees a lot of code that looks like this:
```python
class A:
    def __init__(self, x):
        self.x = x
        # Not calling super().__init__, because the superclass is just `object`, right?

class AA:
    def __init__(...):
        super().__init__(...)  # Being a good citizen.
        ...  # Do anything else that needs to happen.

class B(A, AA):
    pass

B()  # AA.__init__ is not called.
```
In this case `B()` calls `A.__init__` and this then fails to call `AA.__init__`. Co-operative multiple inheritance only works if everyone, well, co-operates.

Even if everyone wants to do their best, there is another issue. When writing `super().__init__`, it isn't actually know what method is being called -- as above, `super()` could be pointing at almost any class at all. This actually means that it's not possible to know what arguments to pass to `super().__init__`! "Only use keyword arguments" is the closest to a resolution that this issue has, and it's still fragile.

In contrast, our no-overriding and abstract-or-final rules means that we never come across this scenario. We always know precisely what is being called.

**Hang on, I don't buy the abstract-or-concrete part. Can't a concrete subclass add a new method to a concrete superclass?**

You're thinking of something like this this:
```python
# This is clearly something we can instantiate: it has no abstract methods/attributes.
class ConcreteArray(eqx.Module):
    def some_method(self):
        pass

# This is clearly also without abstract methods/attributes, and also doesn't break the
# rule about overriding methods.
class ConcreteArrayTwo(ConcreteArray):
    def another_method(self):
        pass
```
We didn't discuss it above, but we do ban things like this as well. (And `strict=True` will prevent this.) The reason is to simplify things when writing something like:
```python
def add(x: ConcreteArray, y: ConcreteArray) -> ConcreteArray:
    ...
```
so that there are never any questions about what the return type should be. (If we passed in `ConcreteArrayTwo` in to `x` or `y`, maybe we should try to return a `ConcreteArrayTwo` instead? What if `x = ConcreteArrayTwo()` but `y = ConcreteArrayThree()`, and these two types don't know about each other? Better to avoid the question in the first place.)

**These ideas have appeared in &lt;XYZ language&gt;?**

Yup! Variants of this design pattern are very common, especially in modern languages like Julia/Rust/etc., or in older languages with a strong emphasis on typing.
