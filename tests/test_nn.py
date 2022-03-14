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


def test_dropout(getkey):
    dropout = eqx.nn.Dropout()
    x = jrandom.normal(getkey(), (3, 4, 5))
    y = dropout(x, key=getkey())
    assert jnp.all((y == 0) | (y == x / 0.5))
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
    k = jrandom.uniform(getkey(), (19, 5))
    v = jrandom.uniform(getkey(), (19, 7))
    assert attn(q, k, v).shape == (19, 11)

    attn = eqx.nn.MultiheadAttention(num_heads=2, query_size=4, key=getkey())
    attn = eqx.tree_at(
        lambda x: (
            x.query_proj.weight,
            x.key_proj.weight,
            x.value_proj.weight,
            x.output_proj.weight,
            x.output_proj.bias,
        ),
        attn,
        [jnp.arange(16).reshape(4, 4) for _ in range(4)]
        + [jnp.zeros_like(attn.output_proj.bias)],
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
    ln = eqx.nn.LayerNorm(128, key=getkey())
    x = jrandom.uniform(getkey(), (128,))
    assert ln(x).shape == (128,)

    ln = eqx.nn.LayerNorm(shape=(128, 128), key=getkey())
    x = jrandom.uniform(getkey(), (128, 128))
    assert ln(x).shape == (128, 128)

    ln = eqx.nn.LayerNorm(10, key=getkey())
    x1 = jnp.linspace(0.1, 1, 10)
    x2 = jnp.linspace(0, 1, 10)
    x3 = (x1 - x1.mean()) / jnp.sqrt(x1.var() + 1e-5)
    assert jnp.allclose(ln(x1), ln(x2), atol=1e-4)
    assert jnp.allclose(ln(x1), x3, atol=1e-4)
