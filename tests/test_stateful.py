import jax.tree_util as jtu
import pytest

import equinox as eqx


def test_delete_init_state():
    model = eqx.nn.BatchNorm(3, "batch")
    eqx.nn.State(model)
    model2 = eqx.nn.delete_init_state(model)

    eqx.nn.State(model)
    with pytest.raises(ValueError):
        eqx.nn.State(model2)

    leaves = [x for x in jtu.tree_leaves(model) if eqx.is_array(x)]
    leaves2 = [x for x in jtu.tree_leaves(model2) if eqx.is_array(x)]
    assert len(leaves) == len(leaves2) + 3
