# Extra features

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

## Checking invariants

Equinox extends dataclasses with a `__check_init__` method, which is automatically ran after initialisation. This can be used to check invariants like so:

```python
class Positive(eqx.Module):
    x: int

    def __check_init__(self):
        if self.x <= 0:
            raise ValueError("Oh no!")
```

This method has three key differences compared to the `__post_init__` provided by dataclasses:

- It is not overridden by an `__init__` method of a subclass. In contrast, the following code has a silent bug:
    
    ```python
    class Parent(eqx.Module):
        x: int

        def __post_init__(self):
            if self.x <= 0:
                raise ValueError("Oh no!")

    class Child(Parent):
        x_as_str: str

        def __init__(self, x):
            self.x = x
            self.x_as_str = str(x)

    Child(-1)  # No error!
    ```

- It is automatically called for parent classes; `super().__check_init__()` is not required:

    ```python
    class Parent(eqx.Module):
        def __check_init__(self):
            print("Parent")

    class Child(Parent):
        def __check_init__(self):
            print("Child")

    Child()  # prints out both Child and Parent
    ```

    As with the previous bullet point, this is to prevent child classes accidentally failing to check that the invariants of their parent hold.

- Assignment is not allowed:
    
    ```python
    class MyModule(eqx.Module):
        foo: int

        def __check_init__(self):
            self.foo = 1  # will raise an error
    ```

    This is to prevent `__check_init__` from doing anything too surprising: as the name suggests, it's meant to be used for checking invariants.

## Creating wrapper modules

::: equinox.module_update_wrapper
