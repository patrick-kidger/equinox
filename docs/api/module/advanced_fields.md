# Advanced features

## Converters and static fields

Equinox modules are [dataclasses](https://docs.python.org/3/library/dataclasses.html). Equinox extends this support with converters and static fields.

::: equinox.field

## Abstract attributes

Equinox modules can be used as [abstract base classes](https://docs.python.org/3/library/abc.html), which means they support [`abc.abstractmethod`](https://docs.python.org/3/library/abc.html#abc.abstractmethod). Equinox extends this with support for abstract instance attributes and abstract class attributes.

::: equinox.AbstractVar
    selection:
        members: false

::: equinox.AbstractClassVar
    selection:
        members: false

## Creating wrapper modules

::: equinox.module_update_wrapper
