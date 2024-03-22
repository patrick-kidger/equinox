import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import Array, Float, Int


def test_shared_array(getkey):
    class MyModule(eqx.Module):
        shared: eqx.nn.Shared

        def __init__(self):
            embedding = eqx.nn.Embedding(
                num_embeddings=3, embedding_size=4, key=getkey()
            )
            head = eqx.nn.Linear(4, 3, key=getkey())
            where = lambda pair: pair[1].weight
            get = lambda pair: pair[0].weight
            self.shared = eqx.nn.Shared((embedding, head), where, get)

        def __call__(self, token: Int[Array, ""]):
            nonlocal called
            called = True
            embedding, head = self.shared()
            assert embedding.weight is head.weight
            return head(embedding(token))

    called = False
    module = MyModule()
    module(jnp.array(0))
    assert called


# We share a non-leaf node
def test_shared_node(getkey):
    class MyModule(eqx.Module):
        shared: eqx.nn.Shared

        def __init__(self):
            attention = eqx.nn.MultiheadAttention(
                num_heads=3, query_size=12, key=getkey()
            )
            my_proj = eqx.nn.Linear(12, 12, use_bias=False, key=getkey())
            where = lambda pair: pair[1].key_proj
            get = lambda pair: pair[0]
            self.shared = eqx.nn.Shared((my_proj, attention), where, get)

        def __call__(self, x: Float[Array, "seq 12"]):
            nonlocal called
            called = True
            my_proj, attention = self.shared()
            eq = eqx.tree_equal(my_proj, attention.key_proj)
            x = attention(x, x, x)
            out = jax.vmap(my_proj)(x)
            return out, eq

    called = False
    module = MyModule()
    x = jr.normal(getkey(), (5, 12))

    @eqx.filter_jit
    @eqx.filter_grad(has_aux=True)
    def f(module, x):
        out, eq = module(x)
        return jnp.sum(out), eq

    d_module, eq = f(module, x)
    assert called
    assert eq
    module = eqx.apply_updates(module, d_module)
    d_module, eq = f(module, x)
    assert eq
    module = eqx.apply_updates(module, d_module)


def test_mismatched_structure(getkey):
    x = jr.normal(getkey(), (3, 4))
    y = jr.normal(getkey(), (4, 3))
    with pytest.raises(ValueError, match="Every node being shared together"):
        eqx.nn.Shared((x, y), lambda pair: pair[0], lambda pair: pair[1])


def test_multi_shared(getkey):
    class MyModule(eqx.Module):
        shared: eqx.nn.Shared

        def __init__(self):
            my_proj = eqx.nn.Linear(12, 12, use_bias=False, key=getkey())
            attention = eqx.nn.MultiheadAttention(
                num_heads=3, query_size=12, key=getkey()
            )
            where = lambda pair: (pair[1].key_proj, pair[1].query_proj.weight)
            get = lambda pair: (pair[0], pair[0].weight + 1)
            self.shared = eqx.nn.Shared((my_proj, attention), where, get)

        def __call__(self, x: Float[Array, "seq 12"]):
            nonlocal called
            called = True
            my_proj, attention = self.shared()
            eq1 = eqx.tree_equal(my_proj, attention.key_proj)
            eq2 = (my_proj.weight + 1 == attention.query_proj.weight).all()
            x = attention(x, x, x)
            out = jax.vmap(my_proj)(x)
            eq = eq1 & eq2
            return out, eq

    called = False
    module = MyModule()
    x = jr.normal(getkey(), (5, 12))

    @eqx.filter_jit
    @eqx.filter_grad(has_aux=True)
    def f(module, x):
        out, eq = module(x)
        return jnp.sum(out), eq

    d_module, eq = f(module, x)
    assert called
    assert eq
    module = eqx.apply_updates(module, d_module)
    d_module, eq = f(module, x)
    assert eq
    module = eqx.apply_updates(module, d_module)
