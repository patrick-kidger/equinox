import dataclasses
import warnings
from collections.abc import Callable
from typing import Any


def field(
    *,
    converter: Callable[[Any], Any] | None = None,
    static: bool = False,
    **kwargs,
):
    """Equinox supports extra functionality on top of the default dataclasses.

    **Arguments:**

    - `converter`: a function to call on this field when the model is initialised. For
        example, `field(converter=jax.numpy.asarray)` to convert
        `bool`/`int`/`float`/`complex` values to JAX arrays. This is ran after the
        `__init__` method (i.e. when using a user-provided `__init__`), and after
        `__post_init__` (i.e. when using the default dataclass initialisation).
        If `converter` is `None`, then no converter is registered.
    - `static`: whether the field should not interact with any JAX transform at all (by
        making it part of the PyTree structure rather than a leaf).
    - `**kwargs`: All other keyword arguments are passed on to `dataclass.field`.

    !!! example "Example for `converter`"

        ```python
        class MyModule(eqx.Module):
            foo: Array = eqx.field(converter=jax.numpy.asarray)

        mymodule = MyModule(1.0)
        assert isinstance(mymodule.foo, jax.Array)
        ```

    !!! example "Example for `static`"

        ```python
        class MyModule(eqx.Module):
            normal_field: int
            static_field: int = eqx.field(static=True)

        mymodule = MyModule("normal", "static")
        leaves, treedef = jax.tree_util.tree_flatten(mymodule)
        assert leaves == ["normal"]
        assert "static" in str(treedef)
        ```

    `static=True` means that this field is not a node of the PyTree, so it does not
    interact with any JAX transforms, like JIT or grad. This means that it is usually a
    bug to make JAX arrays be static fields. `static=True` should very rarely be used.
    It is preferred to just filter out each field with `eqx.partition` whenever you need
    to select only some fields.
    """
    try:
        metadata = dict(kwargs.pop("metadata"))  # safety copy
    except KeyError:
        metadata = {}
    if "converter" in metadata:
        raise ValueError("Cannot use metadata with `converter` already set.")
    if "static" in metadata:
        raise ValueError("Cannot use metadata with `static` already set.")
    # We don't just use `lambda x: x` as the default, so that this works:
    # ```
    # class Abstract(eqx.Module):
    #     x: int = eqx.field()
    #
    # class Concrete(Abstract):
    #    @property
    #    def x(self):
    #        pass
    # ```
    # otherwise we try to call the default converter on a property without a setter,
    # and an error is raised.
    # Oddities like the above are to be discouraged, of course, but in particular
    # `field(init=False)` was sometimes used to denote an abstract field (prior to the
    # introduction of `AbstractVar`), so we do want to support this.
    if converter is not None:
        metadata["converter"] = converter
    if static:
        metadata["static"] = True
    return dataclasses.field(metadata=metadata, **kwargs)


def static_field(**kwargs):
    warnings.warn(
        "`equinox.static_field` is deprecated in favour of "
        "`equinox.field(static=True)`",
        stacklevel=2,
    )
    return field(**kwargs, static=True)
