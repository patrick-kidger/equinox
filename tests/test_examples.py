import pathlib
import sys

import jax.numpy as jnp


_here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(_here / ".." / "examples"))

import build_model
import frozen_layer
import train_mlp


def test_build_model():
    build_model.main()


def test_frozen_layer():
    original_model, model = frozen_layer.main()
    assert jnp.all(original_model.layers[0].weight == model.layers[0].weight)
    assert jnp.all(original_model.layers[0].bias == model.layers[0].bias)
    assert jnp.any(original_model.layers[-1].weight != model.layers[-1].weight)
    assert jnp.any(original_model.layers[-1].weight != model.layers[-1].weight)


def test_train_mlp():
    loss = train_mlp.main()
    assert loss < 5
