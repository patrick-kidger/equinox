from collections.abc import Hashable

import jax
import jax.core
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree

from .._doc_utils import WithRepr
from .._pretty_print import pformat_short_array_text, tree_pprint


_dce_store = {}


def _register_alive(name: Hashable, tag: object):
    def _register_alive_impl(i, x):
        leaves, _ = _dce_store[name][tag]
        leaves[i.item()] = (x.shape, x.dtype.name)
        return x

    return _register_alive_impl


def store_dce(x: PyTree[Array], name: Hashable = None):
    """Used to check whether an array (or pytree of arrays) is DCE'd. (That is, whether
    this code has been removed in the compiler, due to dead code eliminitation.)

    `store_dce` must be used within a JIT'd function, and acts as the identity
    function. When the JIT'd function is called, then whether each array got DCE'd or
    not is recorded. This can subsequently be inspected using `inspect_dce`.

    !!! Example

        ```python
        @jax.jit
        def f(x):
            a, _ = eqxi.store_dce((x**2, x + 1))
            return a

        f(1)
        eqxi.inspect_dce()
        # Found 1 call to `equinox.debug.store_dce`.
        # Entry 0:
        # (i32[], <DCE'd>)
        ```

    **Arguments:**

    - `x`: Any PyTree of JAX arrays.
    - `name`: Optional argument. Any hashable value used to distinguish this call site
        from another call site. If used, then it should be passed to `inspect_dce` to
        print only those entries with this name.

    **Returns:**

    `x` is returned unchanged.
    """
    if not isinstance(jnp.array(1) + 1, jax.core.Tracer):
        raise RuntimeError("`equinox.debug.store_dce` should be used inside of JIT.")
    tag = object()
    leaves, treedef = jtu.tree_flatten(x)
    try:
        tag_store = _dce_store[name]
    except KeyError:
        tag_store = _dce_store[name] = {}
    tag_store[tag] = ({}, treedef)
    leaves = [
        jax.pure_callback(  # pyright: ignore
            _register_alive(name, tag), x, i, x, vectorized=True
        )
        for i, x in enumerate(leaves)
    ]
    return jtu.tree_unflatten(treedef, leaves)


def inspect_dce(name: Hashable = None):
    """Used in conjunction with `equinox.debug.check_dce`; see documentation there.

    Must be called outside of any JIT'd function.

    **Arguments:**

    - `name`: Optional argument. Whatever name was used with `check_dce`.

    **Returns:**

    Nothing. DCE information is printed to stdout.
    """
    if isinstance(jnp.array(1) + 1, jax.core.Tracer):
        raise RuntimeError("`equinox.debug.inspect_dce` should be used outside of JIT.")
    tag_store = _dce_store.get(name, {})
    new_leaves = []
    maybe_s = "" if len(tag_store) == 1 else "s"
    print(f"Found {len(tag_store)} call{maybe_s} to `equinox.debug.store_dce`.")
    for i, (leaves, treedef) in enumerate(tag_store.values()):
        for j in range(treedef.num_leaves):
            try:
                shape, dtype = leaves[j]
            except KeyError:
                value = "<DCE'd>"
            else:
                value = pformat_short_array_text(shape, dtype)
            new_leaves.append(WithRepr(None, value))
        tree = jtu.tree_unflatten(treedef, new_leaves)
        print(f"Entry {i}:")
        tree_pprint(tree)
