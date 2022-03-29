import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

import equinox as eqx


def test_tree_at_replace(getkey):
    key = getkey()
    key1, key2 = jrandom.split(key, 2)
    pytree = [1, 2, {"a": jnp.array([1.0, 2.0])}, eqx.nn.Linear(1, 2, key=key1)]
    true_pytree1 = [1, 2, {"a": "hi"}, eqx.nn.Linear(1, 2, key=key1)]
    true_pytree2 = [1, 2, {"a": jnp.array([1.0, 2.0])}, eqx.nn.Linear(1, 2, key=key2)]
    where1 = lambda tree: tree[2]["a"]
    where2 = lambda tree: (tree[3].weight, tree[3].bias)
    weight2 = true_pytree2[3].weight
    bias2 = true_pytree2[3].bias
    pytree1 = eqx.tree_at(where1, pytree, replace="hi")
    pytree2 = eqx.tree_at(where2, pytree, replace=(weight2, bias2))

    assert pytree1[:-2] == true_pytree1[:-2]
    assert pytree2[:-2] == true_pytree2[:-2]
    assert jnp.all(pytree1[-2]["a"] == true_pytree1[-2]["a"])
    assert jnp.all(pytree2[-2]["a"] == true_pytree2[-2]["a"])
    assert jnp.all(pytree1[-1].weight == true_pytree1[-1].weight)
    assert jnp.all(pytree1[-1].bias == true_pytree1[-1].bias)
    assert jnp.all(pytree2[-1].weight == true_pytree2[-1].weight)
    assert jnp.all(pytree2[-1].bias == true_pytree2[-1].bias)

    true_pytree3 = ["hi", 2, {"a": 4}, eqx.nn.Linear(1, 2, key=key1)]
    where3 = lambda tree: (tree[0], tree[2]["a"])
    pytree3 = eqx.tree_at(where3, pytree, replace=("hi", 4))
    assert pytree3[:-1] == true_pytree3[:-1]
    assert jnp.all(pytree3[-1].weight == true_pytree3[-1].weight)
    assert jnp.all(pytree3[-1].bias == true_pytree3[-1].bias)

    with pytest.raises(TypeError):
        eqx.tree_at(where3, pytree, replace=4)
    with pytest.raises(ValueError):
        eqx.tree_at(where3, pytree, replace=(3, 4, 5))


def test_tree_at_replace_fn(getkey):
    key = getkey()
    pytree = [1, 2, 3, {"a": jnp.array([1.0, 2.0])}, eqx.nn.Linear(1, 2, key=key)]

    def replace_fn(x):
        if isinstance(x, int):
            return "found an int"
        else:
            return x

    true_pytree1 = [
        "found an int",
        "found an int",
        3,
        {"a": jnp.array([1.0, 2.0])},
        eqx.nn.Linear(1, 2, key=key),
    ]
    where = lambda tree: (tree[0], tree[1])
    pytree1 = eqx.tree_at(where, pytree, replace_fn=replace_fn)

    assert pytree1[:3] == true_pytree1[:3]
    assert jnp.all(pytree1[3]["a"] == true_pytree1[3]["a"])
    assert jnp.all(pytree1[-1].weight == true_pytree1[-1].weight)
    assert jnp.all(pytree1[-1].bias == true_pytree1[-1].bias)

    with pytest.raises(ValueError):
        eqx.tree_at(where, pytree, replace=(0, 1), replace_fn=replace_fn)


def test_tree_equal():
    key1 = jrandom.PRNGKey(0)
    key2 = jrandom.PRNGKey(1)
    # Not using getkey as ever-so-in-principle two random keys could produce the same weights
    # (like that's ever going to happen)
    pytree1 = [1, 2, 3, {"a": jnp.array([1.0, 2.0])}, eqx.nn.Linear(1, 2, key=key1)]
    pytree2 = [1, 2, 3, {"a": jnp.array([1.0, 2.0])}, eqx.nn.Linear(1, 2, key=key1)]
    pytree3 = [1, 2, 3, {"a": jnp.array([1.0, 2.0])}, eqx.nn.Linear(1, 2, key=key2)]
    pytree4 = [1, 2, 3, {"a": jnp.array([1.0, 4.0])}, eqx.nn.Linear(1, 2, key=key1)]
    pytree5 = [1, 2, 4, {"a": jnp.array([1.0, 2.0])}, eqx.nn.Linear(1, 2, key=key1)]

    assert eqx.tree_equal(pytree1, pytree1, pytree1)
    assert eqx.tree_equal(pytree1, pytree2)
    assert not eqx.tree_equal(pytree1, pytree3)
    assert not eqx.tree_equal(pytree1, pytree4)
    assert not eqx.tree_equal(pytree1, pytree5)


def test_filter_tree_map(getkey):
    model = eqx.nn.MLP(4, 4, 4, 4, key=getkey())
    model = jax.tree_map(lambda x: x + 100 if eqx.is_array(x) else x, model)

    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weight = lambda linear: linear.weight
    clip_weight = lambda weight: jnp.clip(weight, -1, 1)
    apply = lambda linear: eqx.tree_at(get_weight, linear, replace_fn=clip_weight)

    clipped_model = eqx.filter_tree_map(apply, model, is_leaf=is_linear)
    for layer in clipped_model.layers:
        assert jnp.all(layer.weight <= 1)
        assert jnp.all(layer.weight >= -1)
        assert jnp.any(layer.bias > 1)
