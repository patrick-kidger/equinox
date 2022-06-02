import functools as ft
import warnings
from typing import List, Union

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

import equinox as eqx


def test_custom_init():
    with pytest.raises(TypeError):
        eqx.nn.Linear(1, 1, 1)  # Matches the number of dataclass fields Linear has

    with pytest.raises(TypeError):
        eqx.nn.Linear(3, 4)

    with pytest.raises(TypeError):
        eqx.nn.Linear(3)

    with pytest.raises(TypeError):
        eqx.nn.Linear(out_features=4)


def test_linear(getkey):
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
            eqx.nn.Linear(1, 3, key=getkey()),
        ]
    )
    x = jrandom.normal(getkey(), (2,))
    assert seq(x).shape == (3,)


def test_mlp(getkey):
    mlp = eqx.nn.MLP(2, 3, 8, 2, key=getkey())
    x = jrandom.normal(getkey(), (2,))
    assert mlp(x).shape == (3,)

    mlp = eqx.nn.MLP(in_size=2, out_size=3, width_size=8, depth=2, key=getkey())
    x = jrandom.normal(getkey(), (2,))
    assert mlp(x).shape == (3,)


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
    assert new_bias.shape == conv.bias.shape
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
    assert new_bias.shape == conv.bias.shape
    conv = eqx.tree_at(lambda x: (x.weight, x.bias), conv, (new_weight, new_bias))
    answer = jnp.array([-37, -31, -9, 25, 61, 49, 23, 41, 27]).reshape(1, 3, 3)
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
    assert new_bias.shape == conv.bias.shape
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
    assert new_bias.shape == conv.bias.shape
    conv = eqx.tree_at(lambda x: (x.weight, x.bias), conv, (new_weight, new_bias))
    answer = jnp.array([-3, -2, -1, 0, 1, 2, 3, 4, 1, 1, 1, 1]).reshape(1, 3, 2, 2)
    assert jnp.allclose(conv(data), answer)


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
    assert new_bias.shape == conv.bias.shape
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
    assert new_bias.shape == conv.bias.shape
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
    assert new_bias.shape == conv.bias.shape
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
    assert new_bias.shape == conv.bias.shape
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
        [jnp.arange(16).reshape(4, 4) for _ in range(4)],
    )
    x = jnp.array([[1, 2, 3, 4]])
    assert jnp.allclose(attn(x, x, x), jnp.array([[680.0, 1960.0, 3240.0, 4520.0]]))


def test_embedding(getkey):
    emb = eqx.nn.Embedding(100, 512, key=getkey())
    x = jnp.array([1])
    assert emb(x).shape == (1, 512)

    emb = eqx.nn.Embedding(num_embeddings=10, embedding_size=20, key=getkey())
    x = jnp.array([0])
    assert emb(x).shape == (1, 20)

    emb = eqx.nn.Embedding(
        10, 10, weight=jnp.linspace(0.1, 10, 100).reshape(10, 10), key=getkey()
    )
    x = jnp.array([-1])
    assert jnp.allclose(emb(x), jnp.linspace(9.1, 10.0, 10))


def test_layer_norm(getkey):
    ln = eqx.nn.LayerNorm(128)
    x = jrandom.uniform(getkey(), (128,))
    assert ln(x).shape == (128,)

    ln = eqx.nn.LayerNorm(shape=(128, 128))
    x = jrandom.uniform(getkey(), (128, 128))
    assert ln(x).shape == (128, 128)

    ln = eqx.nn.LayerNorm(10)
    x1 = jnp.linspace(0.1, 1, 10)
    x2 = jnp.linspace(0, 1, 10)
    x3 = (x1 - x1.mean()) / jnp.sqrt(x1.var() + 1e-5)
    assert jnp.allclose(ln(x1), ln(x2), atol=1e-4)
    assert jnp.allclose(ln(x1), x3, atol=1e-4)


def test_batch_norm(getkey):
    x0 = jrandom.uniform(getkey(), (5,))
    x1 = jrandom.uniform(getkey(), (10, 5))
    x2 = jrandom.uniform(getkey(), (10, 5, 6))
    x3 = jrandom.uniform(getkey(), (10, 5, 7, 8))

    # Test that it works with a single vmap'd axis_name

    bn = eqx.experimental.BatchNorm(5, "batch")

    assert jax.vmap(bn, axis_name="batch")(x1).shape == x1.shape
    running_mean, running_var = bn.state_index.unsafe_get()[0]
    assert running_mean.shape == (5,)
    assert running_var.shape == (5,)

    assert jax.vmap(bn, axis_name="batch")(x2).shape == x2.shape
    running_mean, running_var = bn.state_index.unsafe_get()[0]
    assert running_mean.shape == (5,)
    assert running_var.shape == (5,)

    assert jax.vmap(bn, axis_name="batch")(x3).shape == x3.shape
    running_mean, running_var = bn.state_index.unsafe_get()[0]
    assert running_mean.shape == (5,)
    assert running_var.shape == (5,)

    # Test that it fails without any vmap'd axis_name

    with pytest.raises(NameError):
        jax.vmap(bn)(x1)

    with pytest.raises(NameError):
        bn(x0)

    # Test that it vmaps with other vmaps without axis_name

    bn = eqx.experimental.BatchNorm(5, "batch")

    assert (
        jax.vmap(jax.vmap(bn, axis_name="batch", in_axes=1, out_axes=1))(x2).shape
        == x2.shape
    )
    running_mean, running_var = bn.state_index.unsafe_get()[0]
    assert running_mean.shape == (10, 5)
    assert running_var.shape == (10, 5)

    assert (
        jax.vmap(
            jax.vmap(bn, axis_name="batch", in_axes=1, out_axes=1),
            axis_name="not-batch",
        )(x2).shape
        == x2.shape
    )
    running_mean, running_var = bn.state_index.unsafe_get()[0]
    assert running_mean.shape == (10, 5)
    assert running_var.shape == (10, 5)

    # Test that switching to a different amount of batching raises an error

    with pytest.raises(RuntimeError):
        jax.vmap(bn, axis_name="batch")(x1)

    # Test that it handles multiple axis_names

    bn = eqx.experimental.BatchNorm(6, ("batch1", "batch2"))
    assert (
        jax.vmap(jax.vmap(bn, axis_name="batch1"), axis_name="batch2")(x2).shape
        == x2.shape
    )

    # Test that it normalises

    x1alt = jrandom.normal(jrandom.PRNGKey(5678), (10, 5))  # avoid flakey test
    bn = eqx.experimental.BatchNorm(5, "batch", channelwise_affine=False)
    out = jax.vmap(bn, axis_name="batch")(x1alt)
    true_out = (x1alt - jnp.mean(x1alt, axis=0)) / jnp.sqrt(
        jnp.var(x1alt, axis=0) + 1e-5
    )
    assert jnp.allclose(out, true_out)

    # Test that the statistics update during training

    bn = eqx.experimental.BatchNorm(5, "batch")
    with pytest.raises(KeyError):
        bn.state_index.unsafe_get()
    jax.vmap(bn, axis_name="batch")(x1)
    running_mean, running_var = bn.state_index.unsafe_get()[0]
    jax.vmap(bn, axis_name="batch")(3 * x1 + 10)
    running_mean2, running_var2 = bn.state_index.unsafe_get()[0]
    assert not jnp.allclose(running_mean, running_mean2)
    assert not jnp.allclose(running_var, running_var2)

    # Test that the statistics don't update at inference

    bn = eqx.experimental.BatchNorm(5, "batch")
    jax.vmap(bn, axis_name="batch")(x1)
    running_mean, running_var = bn.state_index.unsafe_get()[0]
    jax.vmap(ft.partial(bn, inference=True), axis_name="batch")(3 * x1 + 10)
    running_mean2, running_var2 = bn.state_index.unsafe_get()[0]
    assert jnp.allclose(running_mean, running_mean2)
    assert jnp.allclose(running_var, running_var2)


def test_spectral_norm(getkey):
    weight = jrandom.normal(getkey(), (5, 6))
    spectral = eqx.experimental.SpectralNorm(weight, key=getkey())
    for _ in range(100):
        spectral.__jax_array__()
    _, s, _ = jnp.linalg.svd(spectral.__jax_array__())
    assert jnp.allclose(s[0], 1)
    # "gradient descent"
    spectral = eqx.tree_at(lambda s: s.weight, spectral, spectral.weight + 1)
    _, s, _ = jnp.linalg.svd(spectral.__jax_array__())
    assert not jnp.allclose(s[0], 1)
    for _ in range(100):
        spectral.__jax_array__()
    _, s, _ = jnp.linalg.svd(spectral.__jax_array__())
    assert jnp.allclose(s[0], 1)

    # Test not updated at inference time
    spectral = eqx.tree_at(
        lambda s: (s.weight, s.inference), spectral, (spectral.weight + 1, True)
    )
    _, s, _ = jnp.linalg.svd(spectral.__jax_array__())
    assert not jnp.allclose(s[0], 1)
    for _ in range(100):
        spectral.__jax_array__()
    _, s, _ = jnp.linalg.svd(spectral.__jax_array__())
    assert not jnp.allclose(s[0], 1)

    # Test withkey

    mlp = eqx.nn.MLP(2, 2, 2, 2, key=getkey())
    spectral = eqx.experimental.SpectralNorm.withkey(getkey())

    def get_weights(m):
        is_linear = lambda x: isinstance(x, eqx.nn.Linear)
        return tuple(
            k.weight for k in jax.tree_leaves(m, is_leaf=is_linear) if is_linear(k)
        )

    spectral_mlp = eqx.tree_at(get_weights, mlp, replace_fn=spectral)
    for layer in spectral_mlp.layers:
        assert isinstance(layer.weight, eqx.experimental.SpectralNorm)

    # Test >2 dimensional input

    conv = eqx.nn.Conv3d(5, 4, 3, key=getkey())
    conv = eqx.tree_at(lambda c: c.weight, conv, replace_fn=spectral)
    assert conv(jrandom.normal(getkey(), (5, 8, 8, 8))).shape == (4, 6, 6, 6)


def test_maxpool1d():

    x = jnp.arange(14).reshape(1, 14)
    max_pool = eqx.nn.MaxPool1D(2, 3)
    output = max_pool(x)
    answer = jnp.array([1, 4, 7, 10, 13])

    assert jnp.all(output == answer)


def test_avgpool1d():

    x = jnp.arange(14).reshape(1, 14)
    avg_pool = eqx.nn.AvgPool1D(2, 3)
    output = avg_pool(x)
    answer = jnp.array([0.5, 3.5, 6.5, 9.5, 12.5])

    assert jnp.all(output == answer)


def test_maxpool2d():

    x = jnp.arange(36).reshape(1, 6, 6)
    max_pool = eqx.nn.MaxPool2D(2, (3, 2))
    output = max_pool(x)
    answer = jnp.array([[7, 9, 11], [25, 27, 29]])

    assert jnp.all(output == answer)


def test_avgpool2d():

    x = jnp.arange(36).reshape(1, 6, 6)
    avg_pool = eqx.nn.AvgPool2D((1, 3), 2)
    output = avg_pool(x)
    answer = jnp.array([[1, 3], [13, 15], [25, 27]])

    assert jnp.all(output == answer)


def test_maxpool3d():

    x = jnp.arange(64).reshape(1, 4, 4, 4)
    max_pool = eqx.nn.MaxPool3D(2, (3, 2, 1))
    output = max_pool(x)
    answer = jnp.array([[[21, 22, 23], [29, 30, 31]]])

    assert jnp.all(output == answer)


def test_avgpool3d():

    x = jnp.arange(64).reshape(1, 4, 4, 4)
    avg_pool = eqx.nn.AvgPool3D((1, 3, 1), 2)
    output = avg_pool(x)
    answer = jnp.array([[[4, 6]], [[36, 38]]])

    assert jnp.all(output == answer)


def test_poolpadding():
    x = jnp.arange(64).reshape(1, 4, 4, 4)
    max_pool = eqx.nn.MaxPool3D(2, 1, ((0, 1), (0, 1), (0, 1)))
    output = max_pool(x)

    assert output.shape == (1, 4, 4, 4)


def test_poolbackprop():
    def max_pool_mean(x):
        max_pool = eqx.nn.MaxPool3D((2, 2, 2), (1, 1, 1), ((0, 1), (0, 1), (0, 1)))
        return jnp.mean(max_pool(x))

    x = jnp.arange(64, dtype=jnp.float32).reshape(1, 4, 4, 4)
    grad_fn = jax.value_and_grad(max_pool_mean)

    grad_fn(x)


def test_poolnetworkbackprop(getkey):
    class CNN(eqx.Module):
        conv_layer: List[Union[eqx.nn.Conv2d, eqx.nn.MaxPool2D]]
        linear_layers: List[eqx.nn.Linear]

        def __init__(self, key):
            key1, key2, key3 = jax.random.split(key, 3)
            self.conv_layer = [eqx.nn.Conv2d(3, 2, 3, key=key1), eqx.nn.MaxPool2D(2, 2)]
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
