import equinox as eqx
import jax.random as jr
import jax.tree_util as jtu


def random_pytree(key, treedef):
    keys = jr.split(key, treedef.num_leaves)
    leaves = []
    for key in keys:
        dimkey, sizekey, valuekey = jr.split(key, 3)
        num_dims = jr.randint(dimkey, (), 0, 5).item()
        dim_sizes = jr.randint(sizekey, (num_dims,), 0, 5).tolist()
        value = jr.normal(valuekey, dim_sizes)
        leaves.append(value)
    return jtu.tree_unflatten(treedef, leaves)


treedefs = [
    jtu.tree_structure(x)
    for x in (
        0,
        None,
        {"a": [0, 0], "b": 0},
    )
]


def tree_allclose(x, y, *, rtol=1e-5, atol=1e-8):
    return eqx.tree_equal(x, y, typematch=True, rtol=rtol, atol=atol)
