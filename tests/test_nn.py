import warnings
from typing import Union

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from jax._src.dtypes import TypePromotionError


def test_custom_init():
    with pytest.raises(TypeError):
        eqx.nn.Linear(3, 4)  # pyright: ignore

    with pytest.raises(TypeError):
        eqx.nn.Linear(3)  # pyright: ignore

    with pytest.raises(TypeError):
        eqx.nn.Linear(out_features=4)  # pyright: ignore


def test_linear(getkey):
    # Zero input shape
    linear = eqx.nn.Linear(0, 4, key=getkey())
    x = jrandom.normal(getkey(), (0,))
    assert linear(x).shape == (4,)

    # Zero output shape
    linear = eqx.nn.Linear(4, 0, key=getkey())
    x = jrandom.normal(getkey(), (4,))
    assert linear(x).shape == (0,)

    # Positional arguments
    linear = eqx.nn.Linear(3, 4, key=getkey())
    x = jrandom.normal(getkey(), (3,))
    assert linear(x).shape == (4,)

    # Some keyword arguments
    linear = eqx.nn.Linear(3, out_features=4, key=getkey())
    x = jrandom.normal(getkey(), (3,))
    assert linear(x).shape == (4,)

    # All keyword arguments
    linear = eqx.nn.Linear(in_features=3, out_features=4, key=getkey())
    x = jrandom.normal(getkey(), (3,))
    assert linear(x).shape == (4,)

    linear = eqx.nn.Linear("scalar", 2, key=getkey())
    x = jrandom.normal(getkey(), ())
    assert linear(x).shape == (2,)

    linear = eqx.nn.Linear(2, "scalar", key=getkey())
    x = jrandom.normal(getkey(), (2,))
    assert linear(x).shape == ()

    linear = eqx.nn.Linear(2, "scalar", key=getkey(), dtype=jnp.float16)
    x = jrandom.normal(getkey(), (2,), dtype=jnp.float16)
    assert linear(x).dtype == jnp.float16

    linear = eqx.nn.Linear(2, "scalar", key=getkey(), dtype=jnp.complex64)
    x = jrandom.normal(getkey(), (2,), dtype=jnp.complex64)
    assert linear(x).dtype == jnp.complex64


def test_identity(getkey):
    identity1 = eqx.nn.Identity()
    identity2 = eqx.nn.Identity(1)
    identity3 = eqx.nn.Identity(2, hi=True)
    identity4 = eqx.nn.Identity(eqx.nn.Identity())
    assert identity1 == identity2
    assert identity1 == identity3
    assert identity1 == identity4
    x = jrandom.normal(getkey(), (3, 5, 9))
    assert jnp.all(x == identity1(x))
    assert jnp.all(x == identity2(x))
    assert jnp.all(x == identity3(x))
    assert jnp.all(x == identity4(x))


def test_dropout_basic(getkey):
    dropout = eqx.nn.Dropout()
    x = jrandom.normal(getkey(), (3, 4, 5))
    y = dropout(x, key=getkey())
    assert jnp.all((y == 0) | (y == x / 0.5))


def test_dropout_inference(getkey):
    dropout = eqx.nn.Dropout()
    x = jrandom.normal(getkey(), (3, 4, 5))
    z1 = dropout(x, key=getkey(), inference=True)
    z2 = dropout(x, inference=True)
    assert jnp.all(x == z1)
    assert jnp.all(x == z2)

    dropout2 = eqx.nn.Dropout(inference=True)
    assert jnp.all(x == dropout2(x))

    dropout3 = eqx.tree_at(lambda d: d.inference, dropout2, replace=False)
    assert jnp.any(x != dropout3(x, key=jrandom.PRNGKey(0)))


def test_dropout_deterministic(getkey):
    with warnings.catch_warnings():
        dropout = eqx.nn.Dropout()
        x = jrandom.normal(getkey(), (3, 4, 5))
        warnings.simplefilter("ignore")
        z1 = dropout(x, key=getkey(), deterministic=True)
        z2 = dropout(x, deterministic=True)
        assert jnp.all(x == z1)
        assert jnp.all(x == z2)

        dropout2 = eqx.nn.Dropout(deterministic=True)
        assert jnp.all(x == dropout2(x))

        dropout3 = eqx.tree_at(lambda d: d.deterministic, dropout2, replace=False)
        assert jnp.any(x != dropout3(x, key=jrandom.PRNGKey(0)))


def test_gru_cell(getkey):
    gru = eqx.nn.GRUCell(2, 8, key=getkey())
    h = jrandom.normal(getkey(), (8,))
    x = jrandom.normal(getkey(), (5, 2))
    for xi in x:
        h = gru(xi, h)
        assert h.shape == (8,)


def test_lstm_cell(getkey):
    gru = eqx.nn.LSTMCell(2, 8, key=getkey())
    h = jrandom.normal(getkey(), (8,)), jrandom.normal(getkey(), (8,))
    x = jrandom.normal(getkey(), (5, 2))
    for xi in x:
        h = gru(xi, h)
        h_, c_ = h
        assert h_.shape == (8,)
        assert c_.shape == (8,)


def test_sequential(getkey):
    seq = eqx.nn.Sequential(
        [
            eqx.nn.Linear(2, 4, key=getkey()),
            eqx.nn.Linear(4, 1, key=getkey()),
            eqx.nn.BatchNorm(1, axis_name="batch"),
            eqx.nn.Linear(1, 3, key=getkey()),
        ]
    )
    x = jrandom.normal(getkey(), (1, 2))
    batch_seq = jax.vmap(seq, axis_name="batch", in_axes=(0, None), out_axes=(0, None))
    state = eqx.nn.State(seq)

    output, state = batch_seq(x, state)
    assert output[0].shape == (3,)
    # Test indexing
    assert isinstance(seq[0], eqx.nn.Linear)
    assert isinstance(seq[1:], eqx.nn.Sequential)

    x = jrandom.normal(getkey(), (1, 4))
    batch_seq = jax.vmap(
        seq[1:],
        axis_name="batch",
        in_axes=(0, None),
        out_axes=(0, None),
    )
    output, state = batch_seq(x, state)
    assert output[0].shape == (3,)
    assert len(seq) == 4
    assert eqx.nn.Sequential(list(seq)) == seq


@pytest.mark.parametrize("inner_stateful", (False, True))
@pytest.mark.parametrize("outer_stateful", (False, True))
def test_nested_sequential(inner_stateful, outer_stateful, getkey):
    def make():
        inner_seq = eqx.nn.Sequential(
            [
                eqx.nn.Linear(2, 4, key=getkey()),
                eqx.nn.BatchNorm(4, axis_name="batch")
                if inner_stateful
                else eqx.nn.Identity(),
                eqx.nn.Linear(4, 3, key=getkey()),
            ]
        )
        outer_seq = eqx.nn.Sequential(
            [
                eqx.nn.Linear(5, 2, key=getkey()),
                inner_seq,
                eqx.nn.BatchNorm(3, axis_name="batch")
                if outer_stateful
                else eqx.nn.Identity(),
                eqx.nn.Linear(3, 6, key=getkey()),
            ]
        )
        return outer_seq

    x = jrandom.normal(getkey(), (1, 5))
    if inner_stateful or outer_stateful:
        model, state = eqx.nn.make_with_state(make)()
        out, state = jax.vmap(
            model, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
        )(x, state)
        assert isinstance(state, eqx.nn.State)
    else:
        model = make()
        out = jax.vmap(model, axis_name="batch")(x)
    assert isinstance(out, jax.Array)
    assert out.shape == (1, 6)


def test_mlp(getkey):
    mlp = eqx.nn.MLP(2, 3, 8, 2, key=getkey())
    x = jrandom.normal(getkey(), (2,))
    assert mlp(x).shape == (3,)

    mlp = eqx.nn.MLP(in_size=2, out_size=3, width_size=8, depth=2, key=getkey())
    x = jrandom.normal(getkey(), (2,))
    assert mlp(x).shape == (3,)

    mlp = eqx.nn.MLP("scalar", 2, 2, 2, key=getkey())
    x = jrandom.normal(getkey(), ())
    assert mlp(x).shape == (2,)

    mlp = eqx.nn.MLP(2, "scalar", 2, 2, key=getkey())
    x = jrandom.normal(getkey(), (2,))
    assert mlp(x).shape == ()
    assert [mlp.layers[i].use_bias for i in range(0, 3)] == [True, True, True]

    mlp = eqx.nn.MLP(2, 3, 8, 2, use_bias=False, use_final_bias=True, key=getkey())
    x = jrandom.normal(getkey(), (2,))
    assert mlp(x).shape == (3,)
    assert [mlp.layers[i].use_bias for i in range(0, 3)] == [False, False, True]

    mlp = eqx.nn.MLP(2, 3, 8, 2, use_bias=True, use_final_bias=False, key=getkey())
    x = jrandom.normal(getkey(), (2,))
    assert mlp(x).shape == (3,)
    assert [mlp.layers[i].use_bias for i in range(0, 3)] == [True, True, False]


def test_mlp_learnt_activation():
    mlp = eqx.nn.MLP(
        2,
        5,
        8,
        2,
        activation=eqx.nn.PReLU(),
        final_activation=eqx.nn.PReLU(),
        key=jrandom.PRNGKey(5678),
    )
    x = jnp.array([0.5, 0.7])
    assert mlp.activation.negative_slope.shape == (2, 8)
    assert mlp.final_activation.negative_slope.shape == (5,)

    @eqx.filter_jit
    @eqx.filter_grad
    def grad(mlp, x):
        return jnp.sum(mlp(x))

    grads = grad(mlp, x)
    assert grads.activation.negative_slope.shape == (2, 8)
    assert grads.final_activation.negative_slope.shape == (5,)


def test_conv1d(getkey):
    # Positional arguments
    conv = eqx.nn.Conv1d(1, 3, 3, key=getkey())
    x = jrandom.normal(getkey(), (1, 32))
    assert conv(x).shape == (3, 30)

    # Some keyword arguments
    conv = eqx.nn.Conv1d(1, out_channels=3, kernel_size=(3,), key=getkey())
    x = jrandom.normal(getkey(), (1, 32))
    assert conv(x).shape == (3, 30)

    # All keyword arguments
    conv = eqx.nn.Conv1d(
        in_channels=1,
        out_channels=3,
        kernel_size=(3,),
        padding=1,
        use_bias=False,
        key=getkey(),
    )
    x = jrandom.normal(getkey(), (1, 32))
    assert conv(x).shape == (3, 32)

    # Test strides
    conv = eqx.nn.Conv1d(
        in_channels=3,
        out_channels=1,
        kernel_size=(3,),
        stride=2,
        padding=1,
        use_bias=True,
        key=getkey(),
    )
    x = jrandom.normal(getkey(), (3, 32))
    assert conv(x).shape == (1, 16)

    # Test value matches
    conv = eqx.nn.Conv1d(1, 3, kernel_size=3, padding=1, key=getkey())
    new_weight = jnp.arange(9).reshape(3, 1, 3)
    new_bias = jnp.array([1, 2, 3]).reshape(3, 1)
    data = jnp.arange(-3, 3).reshape(1, -1)
    assert new_weight.shape == conv.weight.shape
    assert new_bias.shape == conv.bias.shape  # pyright: ignore
    conv = eqx.tree_at(lambda x: (x.weight, x.bias), conv, (new_weight, new_bias))
    answer = jnp.array(
        [-6, -3, 0, 3, 6, 3, -20, -20, -8, 4, 16, 13, -34, -37, -16, 5, 26, 23]
    ).reshape(3, 6)
    assert jnp.allclose(conv(data), answer)


def test_conv2d(getkey):
    # Positional arguments
    conv = eqx.nn.Conv2d(1, 3, 3, key=getkey())
    x = jrandom.normal(getkey(), (1, 32, 32))
    assert conv(x).shape == (3, 30, 30)

    # Some keyword arguments
    conv = eqx.nn.Conv2d(1, out_channels=3, kernel_size=(3, 3), key=getkey())
    x = jrandom.normal(getkey(), (1, 32, 32))
    assert conv(x).shape == (3, 30, 30)

    # All keyword arguments
    conv = eqx.nn.Conv2d(
        in_channels=1,
        out_channels=3,
        kernel_size=(3, 3),
        padding=1,
        use_bias=False,
        key=getkey(),
    )
    x = jrandom.normal(getkey(), (1, 32, 32))
    assert conv(x).shape == (3, 32, 32)

    # Test strides
    conv = eqx.nn.Conv2d(
        in_channels=3,
        out_channels=1,
        kernel_size=(3, 3),
        stride=2,
        padding=1,
        use_bias=True,
        key=getkey(),
    )
    x = jrandom.normal(getkey(), (3, 32, 32))
    assert conv(x).shape == (1, 16, 16)

    # Test value matches
    conv = eqx.nn.Conv2d(1, 1, kernel_size=3, padding=1, key=getkey())
    new_weight = jnp.arange(9).reshape(1, 1, 3, 3)
    new_bias = jnp.array([1]).reshape(1, 1, 1)
    data = jnp.arange(-4, 5).reshape(1, 3, 3)
    assert new_weight.shape == conv.weight.shape
    assert new_bias.shape == conv.bias.shape  # pyright: ignore
    conv = eqx.tree_at(lambda x: (x.weight, x.bias), conv, (new_weight, new_bias))
    answer = jnp.array([-37, -31, -9, 25, 61, 49, 23, 41, 27]).reshape(1, 3, 3)
    assert jnp.allclose(conv(data), answer)

    # Test complex value matches
    conv = eqx.nn.Conv2d(1, 1, 3, padding=1, dtype=jnp.complex64, key=getkey())
    new_weight = jnp.arange(9, dtype=jnp.complex64).reshape(1, 1, 3, 3)
    new_bias = jnp.array([1 + 1j], dtype=jnp.complex64).reshape(1, 1, 1)
    data = (1 + 1j) * jnp.arange(-4, 5, dtype=jnp.complex64).reshape(1, 3, 3)
    assert new_weight.shape == conv.weight.shape
    assert new_bias.shape == conv.bias.shape  # pyright: ignore
    conv = eqx.tree_at(lambda x: (x.weight, x.bias), conv, (new_weight, new_bias))
    answer = jnp.array([-37, -31, -9, 25, 61, 49, 23, 41, 27]).reshape(1, 3, 3)
    answer = (1 + 1j) * answer.astype(jnp.complex64)
    assert jnp.allclose(conv(data), answer)

    # Test groups
    conv = eqx.nn.Conv2d(2, 2, kernel_size=3, padding=1, key=getkey(), groups=2)
    # we will duplicate the weights from the "value matches" case
    # and multiply one copy by 2. Also, we modify the bias
    new_weight = jnp.concatenate(
        [
            1 * jnp.arange(9).reshape(1, 1, 3, 3),
            2 * jnp.arange(9).reshape(1, 1, 3, 3),
        ],
        axis=0,
    )
    new_bias = jnp.array([1, 2]).reshape(2, 1, 1)

    data = jnp.broadcast_to(
        jnp.arange(-4, 5).reshape(1, 3, 3),
        (2, 3, 3),
    )
    assert new_weight.shape == conv.weight.shape
    assert new_bias.shape == conv.bias.shape  # pyright: ignore
    conv = eqx.tree_at(lambda x: (x.weight, x.bias), conv, (new_weight, new_bias))
    # this is the multiplication part, without the bias
    answer_part = jnp.array([-38, -32, -10, 24, 60, 48, 22, 40, 26]).reshape(1, 3, 3)
    answer = (
        jnp.concatenate(
            [
                1 * answer_part,
                2 * answer_part,
            ],
            axis=0,
        )
        + new_bias
    )
    assert jnp.allclose(conv(data), answer)


def test_conv3d(getkey):
    # Positional arguments
    conv = eqx.nn.Conv3d(1, 3, 3, key=getkey())
    x = jrandom.normal(getkey(), (1, 3, 32, 32))
    assert conv(x).shape == (3, 1, 30, 30)

    # Some keyword arguments
    conv = eqx.nn.Conv3d(1, out_channels=3, kernel_size=(3, 3, 3), key=getkey())
    x = jrandom.normal(getkey(), (1, 3, 32, 32))
    assert conv(x).shape == (3, 1, 30, 30)

    # All keyword arguments
    conv = eqx.nn.Conv3d(
        in_channels=1,
        out_channels=3,
        kernel_size=(3, 3, 3),
        padding=1,
        use_bias=False,
        key=getkey(),
    )
    x = jrandom.normal(getkey(), (1, 3, 32, 32))
    assert conv(x).shape == (3, 3, 32, 32)

    # Test strides
    conv = eqx.nn.Conv3d(
        in_channels=3,
        out_channels=1,
        kernel_size=(3, 3, 3),
        stride=2,
        padding=1,
        use_bias=True,
        key=getkey(),
    )
    x = jrandom.normal(getkey(), (3, 3, 32, 32))
    assert conv(x).shape == (1, 2, 16, 16)

    # Test value matches
    conv = eqx.nn.Conv3d(1, 1, kernel_size=(2, 1, 1), padding=(1, 0, 0), key=getkey())
    new_weight = jnp.arange(2).reshape(1, 1, 2, 1, 1)
    new_bias = jnp.array([1]).reshape(1, 1, 1, 1)
    data = jnp.arange(-4, 4).reshape(1, 2, 2, 2)
    assert new_weight.shape == conv.weight.shape
    assert new_bias.shape == conv.bias.shape  # pyright: ignore
    conv = eqx.tree_at(lambda x: (x.weight, x.bias), conv, (new_weight, new_bias))
    answer = jnp.array([-3, -2, -1, 0, 1, 2, 3, 4, 1, 1, 1, 1]).reshape(1, 3, 2, 2)
    assert jnp.allclose(conv(data), answer)


def test_conv_padding(getkey):
    x = jrandom.normal(getkey(), (3, 32, 32))
    conv = eqx.nn.Conv2d(3, 8, 1, 2, padding=((0, 1), (0, 3)), key=getkey())
    output = conv(x)
    assert output.shape == (8, 17, 18)


def test_conv_circular(getkey):
    conv = eqx.nn.Conv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        padding="SAME",
        padding_mode="CIRCULAR",
        key=getkey(),
    )

    x = jrandom.normal(getkey(), (1, 6))
    y1 = conv(x)
    y2 = conv(jnp.roll(x, 2))
    y2 = jnp.roll(y2, -2)
    assert jnp.allclose(y1, y2)


def test_convtranspose1d(getkey):
    # Positional arguments
    conv = eqx.nn.ConvTranspose1d(1, 3, 3, key=getkey())
    x = jrandom.normal(getkey(), (1, 32))
    assert conv(x).shape == (3, 34)

    # Test stride and dilation
    conv = eqx.nn.ConvTranspose1d(
        in_channels=3,
        out_channels=1,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
        dilation=2,
        use_bias=False,
        key=getkey(),
    )
    x = jrandom.normal(getkey(), (3, 31))
    assert conv(x).shape == (1, 64)

    # Test value matches
    conv = eqx.nn.ConvTranspose1d(1, 3, kernel_size=3, padding=0, key=getkey())
    new_weight = jnp.arange(9).reshape(3, 1, 3)
    new_bias = jnp.array([1, 2, 3]).reshape(3, 1)
    data = jnp.arange(-3, 3).reshape(1, -1)
    assert new_weight.shape == conv.weight.shape
    assert new_bias.shape == conv.bias.shape  # pyright: ignore
    conv = eqx.tree_at(lambda x: (x.weight, x.bias), conv, (new_weight, new_bias))
    answer = jnp.array(
        [
            -5,
            -6,
            -3,
            0,
            3,
            6,
            3,
            1,
            -13,
            -20,
            -20,
            -8,
            4,
            16,
            13,
            8,
            -21,
            -34,
            -37,
            -16,
            5,
            26,
            23,
            15,
        ]
    ).reshape(3, 8)
    assert jnp.all(conv(data) == answer)


def test_convtranspose2d(getkey):
    # Positional arguments
    conv = eqx.nn.ConvTranspose2d(1, 3, 3, key=getkey())
    x = jrandom.normal(getkey(), (1, 32, 32))
    assert conv(x).shape == (3, 34, 34)

    # Test stride and dilation
    conv = eqx.nn.ConvTranspose2d(
        in_channels=3,
        out_channels=1,
        kernel_size=(3, 3),
        stride=2,
        padding=1,
        output_padding=1,
        dilation=2,
        use_bias=False,
        key=getkey(),
    )
    x = jrandom.normal(getkey(), (3, 31, 31))
    assert conv(x).shape == (1, 64, 64)

    # Test value matches
    conv = eqx.nn.ConvTranspose2d(1, 1, kernel_size=3, padding=1, key=getkey())
    new_weight = jnp.arange(9).reshape(1, 1, 3, 3)
    new_bias = jnp.array([1]).reshape(1, 1, 1)
    data = jnp.arange(-4, 5).reshape(1, 3, 3)
    assert new_weight.shape == conv.weight.shape
    assert new_bias.shape == conv.bias.shape  # pyright: ignore
    conv = eqx.tree_at(lambda x: (x.weight, x.bias), conv, (new_weight, new_bias))
    answer = jnp.array([-37, -31, -9, 25, 61, 49, 23, 41, 27]).reshape(1, 3, 3)
    assert jnp.all(conv(data) == answer)

    # Test groups
    conv = eqx.nn.ConvTranspose2d(
        2, 2, kernel_size=3, padding=1, key=getkey(), groups=2
    )
    # we will duplicate the weights from the "value matches" case
    # and multiply one copy by 2. Also, we modify the bias
    new_weight = jnp.concatenate(
        [
            1 * jnp.arange(9).reshape(1, 1, 3, 3),
            2 * jnp.arange(9).reshape(1, 1, 3, 3),
        ],
        axis=0,
    )
    new_bias = jnp.array([1, 2]).reshape(2, 1, 1)

    data = jnp.broadcast_to(
        jnp.arange(-4, 5).reshape(1, 3, 3),
        (2, 3, 3),
    )
    assert new_weight.shape == conv.weight.shape
    assert new_bias.shape == conv.bias.shape  # pyright: ignore
    conv = eqx.tree_at(lambda x: (x.weight, x.bias), conv, (new_weight, new_bias))
    # this is the multiplication part, without the bias
    answer_part = jnp.array([-38, -32, -10, 24, 60, 48, 22, 40, 26]).reshape(1, 3, 3)
    answer = (
        jnp.concatenate(
            [
                1 * answer_part,
                2 * answer_part,
            ],
            axis=0,
        )
        + new_bias
    )
    assert jnp.allclose(conv(data), answer)


def test_convtranspose3d(getkey):
    # Positional arguments
    conv = eqx.nn.ConvTranspose3d(1, 3, 3, key=getkey())
    x = jrandom.normal(getkey(), (1, 3, 32, 32))
    assert conv(x).shape == (3, 5, 34, 34)

    # Test stride and dilation
    conv = eqx.nn.ConvTranspose3d(
        in_channels=3,
        out_channels=1,
        kernel_size=(3, 3, 3),
        stride=2,
        padding=1,
        output_padding=1,
        dilation=2,
        use_bias=False,
        key=getkey(),
    )
    x = jrandom.normal(getkey(), (3, 2, 31, 31))
    assert conv(x).shape == (1, 6, 64, 64)

    # Test value matches
    conv = eqx.nn.ConvTranspose3d(
        1, 1, kernel_size=(2, 2, 2), padding=(0, 0, 0), key=getkey()
    )
    new_weight = jnp.arange(8).reshape(1, 1, 2, 2, 2)
    new_bias = jnp.array([1]).reshape(1, 1, 1, 1)
    data = jnp.arange(-4, 4).reshape(1, 2, 2, 2)
    assert new_weight.shape == conv.weight.shape
    assert new_bias.shape == conv.bias.shape  # pyright: ignore
    conv = eqx.tree_at(lambda x: (x.weight, x.bias), conv, (new_weight, new_bias))
    answer = jnp.array(
        [
            -27,
            -44,
            -17,
            -33,
            -49,
            -17,
            -9,
            -12,
            -3,
            -11,
            -9,
            1,
            5,
            29,
            21,
            9,
            23,
            13,
            1,
            4,
            3,
            7,
            15,
            7,
            3,
            4,
            1,
        ]
    ).reshape(1, 3, 3, 3)
    assert jnp.all(conv(data) == answer)


def test_convtranspose_padding(getkey):
    x = jrandom.normal(getkey(), (8, 17, 18))
    conv = eqx.nn.ConvTranspose2d(8, 3, 1, 2, padding=((0, 1), (0, 3)), key=getkey())
    output = conv(x)
    assert output.shape == (3, 32, 32)


def test_convtranspose_same_padding(getkey):
    conv = eqx.nn.Conv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=2,
        padding="SAME",
        dilation=2,
        use_bias=False,
        key=getkey(),
    )

    convt = eqx.nn.ConvTranspose1d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=2,
        padding="SAME",
        output_padding=1,
        dilation=2,
        use_bias=False,
        key=getkey(),
    )

    conv = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), conv)
    convt = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), convt)

    x = jrandom.normal(getkey(), (1, 5))
    y = conv(x)

    jac1 = jax.jacobian(conv)(x).reshape(-1, x.size)
    jac2 = jax.jacobian(convt)(y).reshape(x.size, -1)

    # connectivity should be the same
    assert jnp.allclose(jac1, jac2.T)


def test_convtranspose_circular(getkey):
    conv = eqx.nn.Conv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=2,
        dilation=2,
        padding="SAME",
        use_bias=False,
        padding_mode="CIRCULAR",
        key=getkey(),
    )

    convt = eqx.nn.ConvTranspose1d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=2,
        dilation=2,
        padding="SAME",
        use_bias=False,
        padding_mode="CIRCULAR",
        key=getkey(),
    )

    conv = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), conv)
    convt = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), convt)

    x = jrandom.normal(getkey(), (1, 6))
    y = conv(x)

    jac1 = jax.jacobian(conv)(x).reshape(-1, x.size)
    jac2 = jax.jacobian(convt)(y).reshape(x.size, -1)

    # connectivity should be the same
    assert jnp.allclose(jac1, jac2.T)


def test_dot_product_attention_weights(getkey):
    q = jnp.array([[0.0, 2**0.5]])
    k = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    weights = eqx.nn._attention.dot_product_attention_weights(q, k)
    assert weights.shape == (1, 2)
    assert jnp.allclose(weights, jnp.array([[1, jnp.e]]) / (1 + jnp.e))
    mask = jnp.array([[True, False]])
    weights = eqx.nn._attention.dot_product_attention_weights(q, k, mask)
    assert jnp.allclose(weights, jnp.array([[1.0, 0.0]]))

    q = jnp.array([[1.0]], dtype="float16")
    k = jnp.array([[9.0], [1.0]], dtype="float16")
    weights = eqx.nn._attention.dot_product_attention_weights(q, k)
    assert weights.dtype == q.dtype
    assert weights.max() < 1


def test_dot_product_attention(getkey):
    q = jnp.array([[0.0, 2**0.5]])
    k = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    v = jnp.array([[1.0], [0.0]])
    attn = eqx.nn._attention.dot_product_attention(q, k, v)
    assert attn.shape == (1, 1)
    assert jnp.allclose(attn, jnp.array([[1 / (1 + jnp.e)]]))
    mask = jnp.array([[True, False]])
    attn = eqx.nn._attention.dot_product_attention(q, k, v, mask)
    assert attn == jnp.array([[1.0]])


def test_multihead_attention(getkey):
    attn = eqx.nn.MultiheadAttention(
        num_heads=2,
        query_size=3,
        key_size=5,
        value_size=7,
        output_size=11,
        qk_size=13,
        vo_size=17,
        key=getkey(),
    )
    q = jrandom.uniform(getkey(), (19, 3))
    k = jrandom.uniform(getkey(), (23, 5))
    v = jrandom.uniform(getkey(), (23, 7))
    assert attn(q, k, v).shape == (19, 11)

    attn = eqx.nn.MultiheadAttention(num_heads=2, query_size=4, key=getkey())
    attn = eqx.tree_at(
        lambda x: (
            x.query_proj.weight,
            x.key_proj.weight,
            x.value_proj.weight,
            x.output_proj.weight,
        ),
        attn,
        [jnp.arange(16.0).reshape(4, 4) for _ in range(4)],
    )
    x = jnp.array([[1.0, 2.0, 3.0, 4.0]])
    assert jnp.allclose(attn(x, x, x), jnp.array([[680.0, 1960.0, 3240.0, 4520.0]]))

    x = jnp.arange(1, 13, dtype=jnp.float32).reshape(3, 4)
    mask = jnp.broadcast_to(jnp.array([True, False, False]), (2, 3, 3))
    assert jnp.allclose(
        attn(x, x, x, mask),
        jnp.broadcast_to(jnp.array([[680.0, 1960.0, 3240.0, 4520.0]]), (3, 4)),
    )

    mask = jnp.broadcast_to(jnp.array([True, False, False]), (3, 3))
    assert jnp.allclose(
        attn(x, x, x, mask),
        jnp.broadcast_to(jnp.array([[680.0, 1960.0, 3240.0, 4520.0]]), (3, 4)),
    )


def test_multihead_attention_inference(getkey):
    attn = eqx.nn.MultiheadAttention(
        num_heads=2, query_size=10, dropout_p=0.5, key=getkey()
    )
    x = jrandom.normal(getkey(), (3, 10))
    z1 = attn(x, x, x, key=getkey(), inference=True)
    z2 = attn(x, x, x, inference=True)
    assert jnp.all(z1 == z2)

    z1 = attn(x, x, x, key=getkey())
    z2 = attn(x, x, x, key=getkey())
    assert jnp.any(z1 != z2)

    attn2 = eqx.nn.MultiheadAttention(
        num_heads=2, query_size=10, dropout_p=0.5, inference=True, key=getkey()
    )
    z1 = attn2(x, x, x, key=getkey())
    z2 = attn2(x, x, x, key=getkey())
    assert jnp.all(z1 == z2)


def test_multihead_attention_deterministic(getkey):
    with warnings.catch_warnings():
        attn = eqx.nn.MultiheadAttention(
            num_heads=2, query_size=10, dropout_p=0.5, key=getkey()
        )
        x = jrandom.normal(getkey(), (3, 10))
        warnings.simplefilter("ignore")
        z1 = attn(x, x, x, key=getkey(), deterministic=True)
        z2 = attn(x, x, x, deterministic=True)
        assert jnp.all(z1 == z2)


def test_embedding(getkey):
    emb = eqx.nn.Embedding(100, 512, key=getkey())
    x = jnp.array(1)
    assert emb(x).shape == (512,)

    emb = eqx.nn.Embedding(num_embeddings=10, embedding_size=20, key=getkey())
    x = jnp.array(0)
    assert emb(x).shape == (20,)

    emb = eqx.nn.Embedding(
        10, 10, weight=jnp.linspace(0.1, 10, 100).reshape(10, 10), key=getkey()
    )
    x = jnp.array(-1)
    assert jnp.allclose(emb(x), jnp.linspace(9.1, 10.0, 10))

    emb = eqx.nn.Embedding(weight=jnp.linspace(0.1, 10, 100).reshape(10, 10))
    x = jnp.array(-1)
    assert jnp.allclose(emb(x), jnp.linspace(9.1, 10.0, 10))


def test_layer_norm(getkey):
    ln = eqx.nn.LayerNorm(128)
    x = jrandom.uniform(getkey(), (128,))
    assert ln(x).shape == (128,)

    ln = eqx.nn.LayerNorm(shape=(128, 128), use_weight=False, use_bias=False)
    x = jrandom.uniform(getkey(), (128, 128))
    assert ln(x).shape == (128, 128)

    ln = eqx.nn.LayerNorm(10)
    x1 = jnp.linspace(0.1, 1, 10)
    x2 = jnp.linspace(0, 1, 10)
    x3 = (x1 - x1.mean()) / jnp.sqrt(x1.var() + 1e-5)
    assert jnp.allclose(ln(x1), ln(x2), atol=1e-4)
    assert jnp.allclose(ln(x1), x3, atol=1e-4)

    ln = eqx.nn.LayerNorm(128, dtype=jnp.bfloat16)
    x = jrandom.uniform(getkey(), (128,), dtype=jnp.bfloat16)
    assert ln(x).dtype == jnp.bfloat16


def test_group_norm(getkey):
    gn = eqx.nn.GroupNorm(groups=4, channels=128)
    x = jrandom.uniform(getkey(), (128,))
    assert gn(x).shape == (128,)

    gn = eqx.nn.GroupNorm(groups=4, channels=128, dtype=jnp.bfloat16)
    x = jrandom.uniform(getkey(), (128,), dtype=jnp.bfloat16)
    assert gn(x).dtype == jnp.bfloat16

    gn = eqx.nn.GroupNorm(groups=4, channels=128)
    x = jrandom.uniform(getkey(), (128, 4, 5))
    assert gn(x).shape == (128, 4, 5)

    gn = eqx.nn.GroupNorm(groups=4, channels=128, channelwise_affine=False)
    x = jrandom.uniform(getkey(), (128, 4, 5))
    assert gn(x).shape == (128, 4, 5)

    gn = eqx.nn.GroupNorm(1, 10)
    x1 = jnp.linspace(0.1, 1, 10)
    x2 = jnp.linspace(0, 1, 10)
    x3 = (x1 - x1.mean()) / jnp.sqrt(x1.var() + 1e-5)
    assert jnp.allclose(gn(x1), gn(x2), atol=1e-4)
    assert jnp.allclose(gn(x1), x3, atol=1e-4)

    gn = eqx.nn.GroupNorm(2, 10)
    x1 = jnp.linspace(0.1, 1, 10)
    x2 = jnp.linspace(0, 1, 10)
    x1_ = x1.reshape(2, 5)
    x3_ = (x1_ - x1_.mean(axis=1, keepdims=True)) / jnp.sqrt(
        x1_.var(axis=1, keepdims=True) + 1e-5
    )
    x3 = x3_.reshape(10)
    assert jnp.allclose(gn(x1), gn(x2), atol=1e-4)
    assert jnp.allclose(gn(x1), x3, atol=1e-4)

    # channels not divisible by groups
    with pytest.raises(ValueError):
        gn = eqx.nn.GroupNorm(groups=3, channels=4)

    # test w/ channels=None
    gn = eqx.nn.GroupNorm(groups=4, channelwise_affine=False)
    x = jrandom.uniform(getkey(), (128,))
    assert gn(x).shape == (128,)

    # Unknown channels w/ channelwise_affine=True
    with pytest.raises(ValueError):
        gn = eqx.nn.GroupNorm(groups=4, channels=None, channelwise_affine=True)


def test_batch_norm(getkey):
    x0 = jrandom.uniform(getkey(), (5,))
    x1 = jrandom.uniform(getkey(), (10, 5))
    x2 = jrandom.uniform(getkey(), (10, 5, 6))
    x3 = jrandom.uniform(getkey(), (10, 5, 7, 8))

    # Test that it works with a single vmap'd axis_name

    bn = eqx.nn.BatchNorm(5, "batch")
    state = eqx.nn.State(bn)
    vbn = jax.vmap(bn, axis_name="batch", in_axes=(0, None), out_axes=(0, None))

    for x in (x1, x2, x3):
        out, state = vbn(x, state)
        assert out.shape == x.shape
        running_mean, running_var = state.get(bn.state_index)
        assert running_mean.shape == (5,)
        assert running_var.shape == (5,)

    # Test that it fails without any vmap'd axis_name

    with pytest.raises(NameError):
        jax.vmap(bn, in_axes=(0, None), out_axes=(0, None))(x1, state)
    state = eqx.nn.State(bn)

    with pytest.raises(NameError):
        bn(x0, state)
    state = eqx.nn.State(bn)

    # Test that it vmaps with other vmaps without axis_name

    out, state = jax.vmap(
        jax.vmap(bn, axis_name="batch", in_axes=(1, None), out_axes=(1, None)),
        in_axes=(0, None),
    )(x2, state)
    assert out.shape == x2.shape
    running_mean, running_var = state.get(bn.state_index)
    assert running_mean.shape == (10, 5)
    assert running_var.shape == (10, 5)

    # Test that it handles multiple axis_names

    vvbn = eqx.nn.BatchNorm(6, ("batch1", "batch2"))
    vvstate = eqx.nn.State(vvbn)
    for axis_name in ("batch1", "batch2"):
        vvbn = jax.vmap(
            vvbn, axis_name=axis_name, in_axes=(0, None), out_axes=(0, None)
        )
    out, out_vvstate = vvbn(x2, vvstate)
    assert out.shape == x2.shape
    running_mean, running_var = out_vvstate.get(vvbn.state_index)
    assert running_mean.shape == (6,)
    assert running_var.shape == (6,)

    # Test that it normalises

    x1alt = jrandom.normal(jrandom.PRNGKey(5678), (10, 5))  # avoid flakey test
    bn = eqx.nn.BatchNorm(5, "batch", channelwise_affine=False)
    state = eqx.nn.State(bn)
    vbn = jax.vmap(bn, axis_name="batch", in_axes=(0, None), out_axes=(0, None))
    out, state = vbn(x1alt, state)
    true_out = (x1alt - jnp.mean(x1alt, axis=0, keepdims=True)) / jnp.sqrt(
        jnp.var(x1alt, axis=0, keepdims=True) + 1e-5
    )
    assert jnp.allclose(out, true_out)

    # Test that the statistics update during training
    out, state = vbn(x1, state)
    running_mean, running_var = state.get(bn.state_index)
    out, state = vbn(3 * x1 + 10, state)
    running_mean2, running_var2 = state.get(bn.state_index)
    assert not jnp.allclose(running_mean, running_mean2)
    assert not jnp.allclose(running_var, running_var2)

    # Test that the statistics don't update at inference

    ibn = eqx.nn.inference_mode(bn, value=True)
    vibn = jax.vmap(ibn, axis_name="batch", in_axes=(0, None), out_axes=(0, None))
    out, state = vibn(4 * x1 + 20, state)
    running_mean3, running_var3 = state.get(bn.state_index)
    assert jnp.array_equal(running_mean2, running_mean3)
    assert jnp.array_equal(running_var2, running_var3)

    # Test that we can differentiate through it

    @jax.grad
    def f(x):
        out, _ = vbn(x, state)
        return jnp.sum(out)

    f(jrandom.normal(getkey(), (1, 5)))


def test_spectral_norm(getkey):
    def λ1():
        u, v = state.get(spectral.uv_index)
        σ = jnp.einsum("i,ij,j->", u, spectral.layer.weight, v)
        _, s, _ = jnp.linalg.svd(spectral.layer.weight / σ)  # pyright: ignore
        return s[0]

    x = jrandom.normal(getkey(), (5,))
    spectral = eqx.nn.SpectralNorm(
        eqx.nn.Linear(5, 6, key=getkey()), "weight", key=getkey()
    )
    state = eqx.nn.State(spectral)
    for _ in range(100):
        _, state = spectral(x, state)
    assert jnp.allclose(λ1(), 1)

    # "gradient descent"
    spectral = eqx.tree_at(
        lambda s: s.layer.weight, spectral, spectral.layer.weight + 1
    )
    assert not jnp.allclose(λ1(), 1)
    for _ in range(100):
        _, state = spectral(x, state)
    assert jnp.allclose(λ1(), 1)

    # Test not updated at inference time
    spectral = eqx.tree_at(
        lambda s: s.layer.weight, spectral, spectral.layer.weight + 1
    )
    spectral = eqx.nn.inference_mode(spectral, value=True)
    assert not jnp.allclose(λ1(), 1)
    for _ in range(100):
        _, state = spectral(x, state)
    assert not jnp.allclose(λ1(), 1)

    # Test >2 dimensional input

    x = jrandom.normal(getkey(), (5, 8, 8, 8))
    conv = eqx.nn.Conv3d(5, 4, 3, key=getkey())
    spectral = eqx.nn.SpectralNorm(conv, "weight", key=getkey())
    state = eqx.nn.State(spectral)
    out, _ = spectral(x, state)
    assert out.shape == (4, 6, 6, 6)


def test_weight_norm(getkey):
    # Linear
    linear = eqx.nn.Linear(4, 4, key=getkey())
    weight_norm_linear = eqx.nn.WeightNorm(layer=linear, weight_name="weight")

    x = jrandom.normal(getkey(), (4,))
    out_weight_norm = weight_norm_linear(x)
    out_linear = linear(x)

    assert jnp.allclose(out_weight_norm, out_linear)

    # Axis == None
    linear = eqx.nn.Linear(4, 4, key=getkey())
    weight_norm_linear = eqx.nn.WeightNorm(
        layer=linear, weight_name="weight", axis=None
    )

    x = jrandom.normal(getkey(), (4,))
    out_weight_norm = weight_norm_linear(x)
    out_linear = linear(x)

    assert jnp.allclose(out_weight_norm, out_linear)

    # Conv3d (ndim weight matrices > 2)
    conv = eqx.nn.Conv3d(2, 3, 3, key=getkey())
    weight_norm_conv = eqx.nn.WeightNorm(layer=conv, weight_name="weight")
    x = jrandom.normal(getkey(), (2, 3, 3, 3))
    out_weight_norm = weight_norm_conv(x)
    out_conv = conv(x)

    assert jnp.allclose(out_weight_norm, out_conv)

    # Grads get generated for reparametrized weights, not original
    grads = eqx.filter_grad(lambda model, x: jnp.mean(model(x)))(
        weight_norm_linear, jrandom.normal(getkey(), (4,))
    )

    assert jnp.any(grads.layer.weight)
    assert jnp.any(grads.g)


def test_maxpool1d():
    x = jnp.arange(14).reshape(1, 14)
    max_pool = eqx.nn.MaxPool1d(2, 3)
    output = max_pool(x)
    answer = jnp.array([[1, 4, 7, 10, 13]])

    assert jnp.all(output == answer)

    max_pool = eqx.nn.MaxPool1d(kernel_size=3, stride=3, padding=0, use_ceil=True)
    answer = jnp.array([[2, 5, 8, 11, 13]])
    output = max_pool(x)
    assert jnp.all(output == answer)


def test_avgpool1d():
    x = jnp.arange(14).reshape(1, 14)
    avg_pool = eqx.nn.AvgPool1d(2, 3)
    output = avg_pool(x)
    answer = jnp.array([[0.5, 3.5, 6.5, 9.5, 12.5]])

    assert jnp.all(output == answer)


def test_adaptive_avgpool1d():
    x = jnp.arange(14.0).reshape(1, 14)
    adaptive_pool = eqx.nn.AdaptiveAvgPool1d(4)
    output = adaptive_pool(x)
    answer = jnp.array([[1.5, 5.5, 9.0, 12.0]])
    assert jnp.all(output == answer)

    adaptive_pool = eqx.nn.AdaptiveAvgPool1d(14)
    output = adaptive_pool(x)
    assert jnp.all(output == x)


def test_adaptive_maxpool1d():
    x = jnp.arange(14.0).reshape(1, 14)
    adaptive_pool = eqx.nn.AdaptiveMaxPool1d(4)
    output = adaptive_pool(x)
    answer = jnp.array([[3.0, 7.0, 10.0, 13.0]])
    assert jnp.all(output == answer)

    adaptive_pool = eqx.nn.AdaptiveMaxPool1d(14)
    output = adaptive_pool(x)
    assert jnp.all(output == x)


def test_maxpool2d():
    x = jnp.arange(36).reshape(1, 6, 6)
    max_pool = eqx.nn.MaxPool2d(2, (3, 2))
    output = max_pool(x)
    answer = jnp.array([[[7, 9, 11], [25, 27, 29]]])

    assert jnp.all(output == answer)

    max_pool = eqx.nn.MaxPool2d((3, 3), 2, (1, 1), use_ceil=True)
    output = max_pool(x)
    answer = jnp.array(
        [[[7, 9, 11, 11], [19, 21, 23, 23], [31, 33, 35, 35], [31, 33, 35, 35]]]
    )

    assert jnp.all(output == answer)


def test_avgpool2d():
    x = jnp.arange(36).reshape(1, 6, 6)
    avg_pool = eqx.nn.AvgPool2d((1, 3), 2)
    output = avg_pool(x)
    answer = jnp.array([[[1.0, 3.0], [13.0, 15.0], [25.0, 27.0]]])

    assert jnp.all(output == answer)


def test_adaptive_avgpool2d():
    x = jnp.arange(12.0).reshape(1, 3, 4)
    adaptive_pool = eqx.nn.AdaptiveAvgPool2d((2, 3))
    output = adaptive_pool(x)
    answer = jnp.array([[[2.5, 4.0, 5.0], [8.5, 10.0, 11.0]]])
    assert jnp.all(output == answer)

    adaptive_pool = eqx.nn.AdaptiveAvgPool2d((3, 4))
    output = adaptive_pool(x)
    assert jnp.all(output == x)


def test_adaptive_maxpool2d():
    x = jnp.arange(12.0).reshape(1, 3, 4)
    adaptive_pool = eqx.nn.AdaptiveMaxPool2d((2, 3))
    output = adaptive_pool(x)
    answer = jnp.array([[[5.0, 6.0, 7.0], [9.0, 10.0, 11.0]]])
    assert jnp.all(output == answer)

    adaptive_pool = eqx.nn.AdaptiveMaxPool2d((3, 4))
    output = adaptive_pool(x)
    assert jnp.all(output == x)


def test_maxpool3d():
    x = jnp.arange(64.0).reshape(1, 4, 4, 4)
    max_pool = eqx.nn.MaxPool3d(2, (3, 2, 1))
    output = max_pool(x)
    answer = jnp.array([[[[21.0, 22.0, 23.0], [29.0, 30.0, 31.0]]]])

    assert jnp.all(output == answer)

    max_pool = eqx.nn.MaxPool3d(
        kernel_size=3, padding=(0, 1, 1), stride=2, use_ceil=True
    )
    answer = jnp.asarray(
        [
            [
                [[37.0, 39.0, 39.0], [45.0, 47.0, 47.0], [45.0, 47.0, 47.0]],
                [[53.0, 55.0, 55.0], [61.0, 63.0, 63.0], [61.0, 63.0, 63.0]],
            ]
        ]
    )
    output = max_pool(x)
    assert jnp.all(output == answer)


def test_avgpool3d():
    x = jnp.arange(64.0).reshape(1, 4, 4, 4)
    avg_pool = eqx.nn.AvgPool3d((1, 3, 1), 2)
    output = avg_pool(x)
    answer = jnp.array([[[[4.0, 6.0]], [[36.0, 38.0]]]])

    assert jnp.all(output == answer)


def test_adaptive_avgpool3d():
    x = jnp.arange(18.0).reshape(1, 3, 2, 3)
    adaptive_pool = eqx.nn.AdaptiveAvgPool3d((2, 1, 3))
    output = adaptive_pool(x)
    answer = jnp.array([[[[4.5, 5.5, 6.5]], [[13.5, 14.5, 15.5]]]])
    assert jnp.all(output == answer)

    adaptive_pool = eqx.nn.AdaptiveAvgPool3d((3, 2, 3))
    output = adaptive_pool(x)
    assert jnp.all(output == x)


def test_adaptive_maxpool3d():
    x = jnp.arange(18.0).reshape(1, 3, 2, 3)
    adaptive_pool = eqx.nn.AdaptiveMaxPool3d((2, 1, 3))
    output = adaptive_pool(x)
    answer = jnp.array([[[[9.0, 10.0, 11.0]], [[15.0, 16.0, 17.0]]]])
    assert jnp.all(output == answer)

    adaptive_pool = eqx.nn.AdaptiveMaxPool3d((3, 2, 3))
    output = adaptive_pool(x)
    assert jnp.all(output == x)


def test_poolpadding():
    x = jnp.arange(64.0).reshape(1, 4, 4, 4)
    max_pool = eqx.nn.MaxPool3d(2, 1, ((0, 1), (0, 1), (0, 1)))
    output = max_pool(x)

    assert output.shape == (1, 4, 4, 4)


def test_poolbackprop():
    def max_pool_mean(x):
        max_pool = eqx.nn.MaxPool3d((2, 2, 2), (1, 1, 1), ((0, 1), (0, 1), (0, 1)))
        return jnp.mean(max_pool(x))

    x = jnp.arange(64.0, dtype=jnp.float32).reshape(1, 4, 4, 4)
    grad_fn = jax.value_and_grad(max_pool_mean)

    grad_fn(x)


def test_poolnetworkbackprop(getkey):
    class CNN(eqx.Module):
        conv_layer: list[Union[eqx.nn.Conv2d, eqx.nn.MaxPool2d]]
        linear_layers: list[eqx.nn.Linear]

        def __init__(self, key):
            key1, key2, key3 = jax.random.split(key, 3)
            self.conv_layer = [eqx.nn.Conv2d(3, 2, 3, key=key1), eqx.nn.MaxPool2d(2, 2)]
            self.linear_layers = [
                eqx.nn.Linear(450, 256, key=key2),
                eqx.nn.Linear(256, 10, key=key3),
            ]

        def __call__(self, x):
            for layer in self.conv_layer:
                x = layer(x)
            x = jnp.ravel(x)
            for layer in self.linear_layers:
                x = layer(x)
                x = jax.nn.relu(x)
            return x

    cnn = CNN(getkey())

    @jax.vmap
    @jax.value_and_grad
    def loss_grad(x, y):
        return jax.numpy.mean((y - cnn(x)) ** 2)

    x = jrandom.normal(getkey(), (10, 3, 32, 32))
    y = jrandom.normal(getkey(), (10, 10))
    loss_grad(x, y)


def test_lambda_layer(getkey):
    net = eqx.nn.Sequential(
        [
            eqx.nn.Identity(),
            eqx.nn.Lambda(jnn.relu),
        ]
    )
    x = jnp.array([[-1, -2, -3], [1, 2, 3]])
    output = net(x)
    assert output.shape == (2, 3)
    assert (output >= 0).all()


def test_prelu(getkey):
    # Test single-channel mode
    activation = eqx.nn.PReLU(0.1)

    x = jnp.array([[-1, -2, -3], [-1, 2, 3]])
    expected = jnp.array([[-0.1, -0.2, -0.3], [-0.1, 2, 3]])
    output = activation(x)

    assert jnp.all(output == expected)

    # Test multi-channel mode
    per_channel_alphas = jnp.array([0.2, 0.1, 0.1])
    activation = eqx.nn.PReLU(per_channel_alphas)

    expected_multi_output = jnp.array([[-0.2, -0.2, -0.3], [-0.2, 2, 3]])
    output = activation(x)

    assert activation.negative_slope.shape == (x.shape[-1],)
    assert jnp.all(output == expected_multi_output)


def test_rope_embeddings_shapes(getkey):
    embedding_size = 32
    rope_embeddings = eqx.nn.RotaryPositionalEmbedding(embedding_size)

    n_heads = 4
    seq_length = 8
    query_size = 32
    key_size = 32

    query_heads = jax.random.normal(
        key=getkey(), shape=(seq_length, n_heads, query_size)
    )
    key_heads = jax.random.normal(key=getkey(), shape=(seq_length, n_heads, key_size))
    query_heads = jax.vmap(rope_embeddings, in_axes=1, out_axes=1)(query_heads)
    key_heads = jax.vmap(rope_embeddings, in_axes=1, out_axes=1)(key_heads)

    assert query_heads.shape == (seq_length, n_heads, query_size)
    assert key_heads.shape == (seq_length, n_heads, key_size)


def test_rope_embeddings_freqs_cis():
    # values are generated using
    # Metas Rope embedding code. See this gist which generates these
    # expected values: https://gist.github.com/Artur-Galstyan/8d0bb5743f00650aa6c0d7427595a0ff
    theta = 10_000.0
    expected_freqs_cis = jnp.array(
        [
            [1.0000 + 0.0000j, 1.0000 + 0.0000j, 1.0000 + 0.0000j, 1.0000 + 0.0000j],
            [0.5403 + 0.8415j, 0.9950 + 0.0998j, 0.9999 + 0.0100j, 1.0000 + 0.0010j],
            [-0.4161 + 0.9093j, 0.9801 + 0.1987j, 0.9998 + 0.0200j, 1.0000 + 0.0020j],
            [-0.9900 + 0.1411j, 0.9553 + 0.2955j, 0.9996 + 0.0300j, 1.0000 + 0.0030j],
            [-0.6536 - 0.7568j, 0.9211 + 0.3894j, 0.9992 + 0.0400j, 1.0000 + 0.0040j],
            [0.2837 - 0.9589j, 0.8776 + 0.4794j, 0.9988 + 0.0500j, 1.0000 + 0.0050j],
            [0.9602 - 0.2794j, 0.8253 + 0.5646j, 0.9982 + 0.0600j, 1.0000 + 0.0060j],
            [0.7539 + 0.6570j, 0.7648 + 0.6442j, 0.9976 + 0.0699j, 1.0000 + 0.0070j],
            [-0.1455 + 0.9894j, 0.6967 + 0.7174j, 0.9968 + 0.0799j, 1.0000 + 0.0080j],
            [-0.9111 + 0.4121j, 0.6216 + 0.7833j, 0.9960 + 0.0899j, 1.0000 + 0.0090j],
            [-0.8391 - 0.5440j, 0.5403 + 0.8415j, 0.9950 + 0.0998j, 0.9999 + 0.0100j],
            [0.0044 - 1.0000j, 0.4536 + 0.8912j, 0.9940 + 0.1098j, 0.9999 + 0.0110j],
            [0.8439 - 0.5366j, 0.3624 + 0.9320j, 0.9928 + 0.1197j, 0.9999 + 0.0120j],
            [0.9074 + 0.4202j, 0.2675 + 0.9636j, 0.9916 + 0.1296j, 0.9999 + 0.0130j],
            [0.1367 + 0.9906j, 0.1700 + 0.9854j, 0.9902 + 0.1395j, 0.9999 + 0.0140j],
            [-0.7597 + 0.6503j, 0.0707 + 0.9975j, 0.9888 + 0.1494j, 0.9999 + 0.0150j],
        ]
    )
    embedding_size = 8
    seq_length = 16
    freqs_cis = eqx.nn.RotaryPositionalEmbedding.precompute_freqs_cis(
        embedding_size, seq_length, theta, jnp.float32
    )
    assert jnp.allclose(
        freqs_cis[0], expected_freqs_cis.real, atol=1e-4
    ) and jnp.allclose(freqs_cis[1], expected_freqs_cis.imag, atol=1e-4)

    freqs_cis = eqx.nn.RotaryPositionalEmbedding.precompute_freqs_cis(
        embedding_size, seq_length, theta, jnp.float16
    )
    assert jnp.allclose(
        freqs_cis[0].astype(jnp.float32), expected_freqs_cis.real, rtol=1e-2
    ) and jnp.allclose(
        freqs_cis[1].astype(jnp.float32), expected_freqs_cis.imag, rtol=1e-2
    )


def test_rope_embeddings_values():
    # values are generated using
    # the script in this gist:
    # https://gist.github.com/Artur-Galstyan/d33eda74072fea61545127adb90197b5
    # Those values are generated based on the HuggingFace implementation
    # of the Rope embeddings
    # (see here: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_flax_llama.py#L169)
    expected_values = jnp.array(
        [
            [
                0.0,
                1.0,
                2.0,
                3.0,
            ],
            [-2.887617, 4.9297514, 6.6076975, 7.0496492],
            [-12.422148, 8.778215, 3.1129112, 11.177788],
            [-13.85559, 12.544218, -12.166454, 15.383192],
            [3.1641474, 16.226604, -23.874424, 19.664621],
            [26.769577, 19.824234, -12.937918, 24.020819],
            [30.30889, 23.335985, 18.258457, 28.450514],
            [1.3996639, 26.760752, 41.01269, 32.952423],
        ]
    )

    seq_length = 8
    embedding_size = 4

    x = jnp.arange(seq_length * embedding_size * 1.0).reshape(
        seq_length, embedding_size
    )

    rope_embeddings = eqx.nn.RotaryPositionalEmbedding(
        embedding_size, dtype=jnp.float32
    )
    res = rope_embeddings(x)

    assert jnp.allclose(res, expected_values, atol=1e-6)

    with jax.numpy_dtype_promotion("standard"):
        # Test that high precision rope on low precision input is more
        # accurate than low precision rope on low precision input
        res = rope_embeddings(x.astype(jnp.float16))
        assert jnp.allclose(
            res.astype(jnp.float16),
            expected_values.astype(jnp.float16),
            rtol=1e-3,
        )

    # check that without dtype promotion we throw an error
    with pytest.raises(TypePromotionError):
        rope_embeddings(x.astype(jnp.float16))

    rope_embeddings = eqx.nn.RotaryPositionalEmbedding(
        embedding_size, dtype=jnp.float16
    )
    res = rope_embeddings(x.astype(jnp.float16))

    assert (
        jnp.allclose(res.astype(jnp.float32), expected_values, rtol=1e-2)
        and res.dtype == jnp.float16
    )
