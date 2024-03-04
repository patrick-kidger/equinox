# This is some mildly unpleasant code.
#
# Basically, Optax make the decision *not* to register their optimisers as PyTrees.
# This means that we often end up with spurious recompilation, just because a learning
# rate changed. That results in a new optimiser instance, which is just a function and
# is treated statically.
#
# So here we simply replace all function closures with pytrees, with each of their cell
# contents as their subnodes.

import inspect
import types
from typing import Any, Optional

import jax.tree_util as jtu

from .._module import Module


def _make_cell(val):
    fn = lambda: val
    return fn.__closure__[0]  # pyright: ignore


def _adjust_function_closure(fn, closure):
    out = types.FunctionType(
        code=fn.__code__,
        globals=fn.__globals__,
        name=fn.__name__,
        argdefs=fn.__defaults__,
        closure=closure,
    )
    out.__module__ = fn.__module__
    out.__qualname__ = fn.__qualname__
    out.__doc__ = fn.__doc__
    out.__annotations__.update(fn.__annotations__)
    if fn.__kwdefaults__ is not None:
        out.__kwdefaults__ = fn.__kwdefaults__.copy()
    return out


# Not a pytree.
# Used so that two different local functions, with different identities, can still
# compare equal. This is needed as these leaves are compared statically when
# filter-jit'ing.
class _FunctionWithEquality:
    def __init__(self, fn: types.FunctionType):
        self.fn = fn

    def information(self):
        try:
            # `fn` not defined in REPL.
            source = inspect.getsource(self.fn)
        except OSError:
            # `fn` defined in REPL. In practice this will lead to recompilations based
            # on function identity, but correctness >> speed.
            return self.fn
        else:
            return self.fn.__qualname__, self.fn.__module__, source

    def __hash__(self):
        return hash(self.information())

    def __eq__(self, other):
        return type(self) == type(other) and self.information() == other.information()


class _Closure(Module):
    fn: _FunctionWithEquality
    contents: Optional[tuple[Any, ...]]

    def __init__(self, fn: types.FunctionType):
        self.fn = _FunctionWithEquality(fn)
        if fn.__closure__ is None:
            contents = None
        else:
            contents = tuple(
                closure_to_pytree(cell.cell_contents) for cell in fn.__closure__
            )
        self.contents = contents

    def __call__(self, *args, **kwargs):
        if self.contents is None:
            closure = None
        else:
            closure = tuple(_make_cell(contents) for contents in self.contents)
        fn = _adjust_function_closure(self.fn.fn, closure)
        return fn(*args, **kwargs)


def _fixup_closure(leaf):
    if isinstance(leaf, types.FunctionType):
        return _Closure(leaf)
    else:
        return leaf


def closure_to_pytree(tree):
    """Convert all function closures into pytree nodes.

    **Arguments:**

    - `tree`: Any pytree.

    **Returns:**

    A copy of `tree`, where all function closures have been replaced by a new object
    that is (a) callable like the original function, but (b) iterates over its
    `__closure__` as subnodes in the pytree.

    !!! Example

        ```python
        def some_fn():
            a = jnp.array(1.)

            @closure_to_pytree
            def f(x):
                return x + a

            print(jax.tree_util.tree_leaves(f))  # prints out `a`
        ```

    !!! Warning

        One annoying technical detail in the above example: we had to wrap the whole lot
        in a `some_fn`, so that we're in a local scope. Python treats functions at the
        global scope differently, and this conversion won't result in any global
        variable being treated as part of the pytree.

        In practice, the intended use case of this function is to fix Optax, which
        always uses local functions.
    """
    return jtu.tree_map(_fixup_closure, tree)
