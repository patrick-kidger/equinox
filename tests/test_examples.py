import pathlib
import sys

import jax.numpy as jnp


_here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(_here / ".." / "examples"))

import build_model
import filtered_transformations
import frozen_layer
import modules_to_initapply
import train_rnn


def test_build_model():
    build_model.main()


def test_frozen_layer():
    original_model, model = frozen_layer.main()
    assert jnp.all(original_model.layers[0].weight == model.layers[0].weight)
    assert jnp.all(original_model.layers[0].bias == model.layers[0].bias)
    assert jnp.any(original_model.layers[-1].weight != model.layers[-1].weight)
    assert jnp.any(original_model.layers[-1].weight != model.layers[-1].weight)


def test_filtered_transformations():
    loss = filtered_transformations.main()
    assert loss < 0.1


def test_train_rnn():
    loss = train_rnn.main()
    assert loss < 0.01


def test_modules_to_initapply():
    modules_to_initapply.main()


def test_readme():
    with open(_here / ".." / "README.md") as f:
        program = []
        maybe_program = False
        start_program = False
        for line in f.readlines():
            if "```python" in line:
                maybe_program = True
            elif maybe_program:
                maybe_program = False
                if "import equinox as eqx" in line:
                    program.append(line)
                    start_program = True
            elif start_program:
                if "```" in line:
                    exec("\n".join(program), dict())
                    program = []
                    start_program = False
                else:
                    program.append(line)
