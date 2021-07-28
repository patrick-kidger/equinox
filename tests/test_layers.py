import equinox as eqx
import jax.random as jrandom
import pytest


def test_linear(getkey):
    # Positional arguments
    linear = eqx.Linear(3, 4, key=getkey())
    x = jrandom.normal(getkey(), (3,))
    assert linear(x).shape == (4,)

    # Some keyword arguments
    linear = eqx.Linear(3, out_features=4, key=getkey())
    x = jrandom.normal(getkey(), (3,))
    assert linear(x).shape == (4,)

    # All keyword arguments
    linear = eqx.Linear(in_features=3, out_features=4, key=getkey())
    x = jrandom.normal(getkey(), (3,))
    assert linear(x).shape == (4,)


def test_mlp(getkey):
    mlp = eqx.MLP(2, 3, 8, 2, key=getkey())
    x = jrandom.normal(getkey(), (2,))
    assert mlp(x).shape == (3,)

    mlp = eqx.MLP(in_size=2, out_size=3, width_size=8, depth=2, key=getkey())
    x = jrandom.normal(getkey(), (2,))
    assert mlp(x).shape == (3,)


def test_custom_init():
    with pytest.raises(TypeError):
        eqx.Linear(1, 1, 1)  # Matches the number of dataclass fields Linear has

    with pytest.raises(TypeError):
        eqx.Linear(3, 4)

    with pytest.raises(TypeError):
        eqx.Linear(3)

    with pytest.raises(TypeError):
        eqx.Linear(out_features=4)
