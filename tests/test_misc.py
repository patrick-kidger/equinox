import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

from .helpers import random_pytree, tree_allclose, treedefs


def test_ω_add_mul(getkey):
    # ω(...) initialisation
    ω = eqxi.ω
    a = [0, 1]
    b = [1, 2]
    c = (ω(a) + ω(b)).ω
    assert c == [1, 3]

    # ...**ω initialisation
    for treedef in treedefs:
        a = b = c = random_pytree(getkey(), treedef)

        e1 = (a**ω * 2 + b**ω * c**ω - 3).ω
        e2 = jtu.tree_map(lambda ai, bi, ci: ai * 2 + bi * ci - 3, a, b, c)
        assert tree_allclose(e1, e2)


def test_ω_inplace(getkey):
    ω = eqxi.ω
    for treedef in treedefs:
        a = random_pytree(getkey(), treedef)
        b1 = ω(a).at[()].set(3).ω
        b2 = jtu.tree_map(lambda ai: ai.at[()].set(3), a)
        assert tree_allclose(b1, b2)

        a2 = jtu.tree_map(lambda x: x + 1, a)

        b3 = ω(a).at[()].set(ω(a2)).ω
        b4 = jtu.tree_map(lambda ai, a2i: ai.at[()].set(a2i[()]), a, a2)
        assert tree_allclose(b3, b4)


def test_ω_is_leaf(getkey):
    ω = eqxi.ω
    for treedef in treedefs:
        a = b = random_pytree(getkey(), treedef)
        with pytest.raises(ValueError):
            ω(a) + ω(b, is_leaf=lambda x: isinstance(x, int))  # pyright: ignore
        with pytest.raises(ValueError):
            ω(a, is_leaf=lambda x: isinstance(x, int)) + ω(b)  # pyright: ignore
        with pytest.raises(ValueError):
            ω(a, is_leaf=lambda x: isinstance(x, int)) + ω(
                b, is_leaf=lambda x: isinstance(x, (int, str))
            )  # pyright: ignore

        out = ω(a, is_leaf=lambda x: isinstance(x, int)) + ω(
            b, is_leaf=lambda x: isinstance(x, int)
        )
        assert out.is_leaf(4)
        assert not out.is_leaf("hi")

        b = ω(a, is_leaf=lambda x: isinstance(x, int)).at[()].set(3)
        assert out.is_leaf(4)
        assert not out.is_leaf("hi")

        a2 = jtu.tree_map(lambda x: x + 1, a)

        c = (
            ω(a, is_leaf=lambda x: isinstance(x, int))
            .at[()]
            .set(ω(a2, is_leaf=lambda x: isinstance(x, int)))
        )
        assert c.is_leaf(4)
        assert not c.is_leaf("hi")

        with pytest.raises(ValueError):
            ω(a, is_leaf=lambda x: isinstance(x, int)).at[()].set(ω(a2))
        with pytest.raises(ValueError):
            ω(a).at[()].set(ω(a2, is_leaf=lambda x: isinstance(x, int)))


def test_unvmap():
    unvmap_all = eqxi.unvmap_all
    unvmap_any = eqxi.unvmap_any
    jit_unvmap_all = jax.jit(unvmap_all)
    jit_unvmap_any = jax.jit(unvmap_any)
    vmap_unvmap_all = jax.vmap(unvmap_all, out_axes=None)
    vmap_unvmap_any = jax.vmap(unvmap_any, out_axes=None)

    tt = jnp.array([True, True])
    tf = jnp.array([True, False])
    ff = jnp.array([False, False])

    assert jnp.array_equal(unvmap_all(tt), jnp.array(True))
    assert jnp.array_equal(unvmap_all(tf), jnp.array(False))
    assert jnp.array_equal(unvmap_all(ff), jnp.array(False))
    assert jnp.array_equal(unvmap_any(tt), jnp.array(True))
    assert jnp.array_equal(unvmap_any(tf), jnp.array(True))
    assert jnp.array_equal(unvmap_any(ff), jnp.array(False))

    assert jnp.array_equal(jit_unvmap_all(tt), jnp.array(True))
    assert jnp.array_equal(jit_unvmap_all(tf), jnp.array(False))
    assert jnp.array_equal(jit_unvmap_all(ff), jnp.array(False))
    assert jnp.array_equal(jit_unvmap_any(tt), jnp.array(True))
    assert jnp.array_equal(jit_unvmap_any(tf), jnp.array(True))
    assert jnp.array_equal(jit_unvmap_any(ff), jnp.array(False))

    assert jnp.array_equal(vmap_unvmap_all(tt), jnp.array(True))
    assert jnp.array_equal(vmap_unvmap_all(tf), jnp.array(False))
    assert jnp.array_equal(vmap_unvmap_all(ff), jnp.array(False))
    assert jnp.array_equal(vmap_unvmap_any(tt), jnp.array(True))
    assert jnp.array_equal(vmap_unvmap_any(tf), jnp.array(True))
    assert jnp.array_equal(vmap_unvmap_any(ff), jnp.array(False))

    unvmap_max = eqxi.unvmap_max
    jit_unvmap_max = jax.jit(unvmap_max)
    vmap_unvmap_max = jax.vmap(unvmap_max, out_axes=None)

    _21 = jnp.array([2, 1])
    _11 = jnp.array([1, 1])

    assert jnp.array_equal(unvmap_max(_21), jnp.array(2))
    assert jnp.array_equal(unvmap_max(_11), jnp.array(1))

    assert jnp.array_equal(jit_unvmap_max(_21), jnp.array(2))
    assert jnp.array_equal(jit_unvmap_max(_11), jnp.array(1))

    assert jnp.array_equal(vmap_unvmap_max(_21), jnp.array(2))
    assert jnp.array_equal(vmap_unvmap_max(_11), jnp.array(1))
