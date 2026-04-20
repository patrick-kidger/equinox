# `equinox.Module`

This is the core of the library: subclassing `equinox.Module` ensures that your class is both a PyTree and a frozen dataclass.

The above would be just a couple of lines of code, of course (and this idea exists in numerous libraries, from both before and after Equinox's inception). What else is going on here? Or in other words, why do we think Equinox's implementation is probably the most complete amongst this class of ideas?

## Safety / no footguns:

- Equinox has a lot of nice error messages for every kind of edge case we can think of:
  - Accidentally marking arrays as static fields;
  - Accidentally creating cycles via method assignment;
  - Accidentally closing over arrays;
  - Accidentally ignoring `__post_init__` because an `__init__` method already exists.
- Good behaviour around edge cases:
  - Bound methods like `foo = SomeModule().some_method` are also pytrees, so that things like `jax.jit(...)(foo)` work correctly.
  - Subclassing propagates `__init__` methods, `__doc__`strings.
  - Mutating attributes during `__init__` is totally fine because we don't do any `__setattr__` magic.
- The class is frozen *except* during initialisation, giving both the safety of a frozen dataclass with the nice initialisation syntax of `def __init__(self, ...): self.x = x`.

## Speed:

Flattening/unflattening is pretty fast: we've optimised any overhead down to microseconds.

## A few useful features:

- `equinox.{AbstractVar, AbstractClassVar}`: abstract attributes, analogues for `abc.abstractmethod`.
- `equinox.field(..., converter=..., static=...)` for conversion of fields on assignment, and for marking that a field should be excluded from the PyTree structure.
- `equinox.Module.__check_init__`: called at all levels of the class hierarchy, to assert that invariants are true after initialisation.
- Pretty printing: just like you'd see if you printed it out via Black/`ruff format`, as we use the [Wadler-Lindig](https://github.com/patrick-kidger/wadler_lindig) library.

## Easy to understand:

Equinox Modules should feel familiar from the get-go: they're just frozen dataclasses/pytrees. You can reason about how your code works.

We implement all of the above in a fairly concise, readable amount of code. If you want to understand how Equinox works under the hood, then you should find the code in this directory to be pretty easy to read.
