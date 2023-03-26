"""Implementation of a U-Net.

Author: Ben Walker (https://github.com/Benjamin-Walker)

This implementation is intended for use in a score-based diffusion, so it accepts a `t`
argument.
"""

import math
from typing import Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange

import equinox as eqx


class SinusoidalPosEmb(eqx.Module):
    emb: jax.Array

    def __init__(self, dim):
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.emb = jnp.exp(jnp.arange(half_dim) * -emb)

    def __call__(self, x):
        emb = x * self.emb
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb


class LinearTimeSelfAttention(eqx.Module):
    group_norm: eqx.nn.GroupNorm
    heads: int
    to_qkv: eqx.nn.Conv2d
    to_out: eqx.nn.Conv2d

    def __init__(
        self,
        dim,
        key,
        heads=4,
        dim_head=32,
    ):
        keys = jax.random.split(key, 2)
        self.group_norm = eqx.nn.GroupNorm(min(dim // 4, 32), dim)
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = eqx.nn.Conv2d(dim, hidden_dim * 3, 1, key=keys[0])
        self.to_out = eqx.nn.Conv2d(hidden_dim, dim, 1, key=keys[1])

    def __call__(self, x):
        c, h, w = x.shape
        x = self.group_norm(x)
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "(qkv heads c) h w -> qkv heads c (h w)", heads=self.heads, qkv=3
        )
        k = jax.nn.softmax(k, axis=-1)
        context = jnp.einsum("hdn,hen->hde", k, v)
        out = jnp.einsum("hde,hdn->hen", context, q)
        out = rearrange(
            out, "heads c (h w) -> (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


def upsample_2d(y, factor=2):
    C, H, W = y.shape
    y = jnp.reshape(y, [C, H, 1, W, 1])
    y = jnp.tile(y, [1, 1, factor, 1, factor])
    return jnp.reshape(y, [C, H * factor, W * factor])


def downsample_2d(y, factor=2):
    C, H, W = y.shape
    y = jnp.reshape(y, [C, H // factor, factor, W // factor, factor])
    return jnp.mean(y, axis=[2, 4])


def exact_zip(*args):
    _len = len(args[0])
    for arg in args:
        assert len(arg) == _len
    return zip(*args)


def key_split_allowing_none(key):
    if key is None:
        return key, None
    else:
        return jr.split(key)


class Residual(eqx.Module):
    fn: LinearTimeSelfAttention

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class ResnetBlock(eqx.Module):
    dim_out: int
    is_biggan: bool
    up: bool
    down: bool
    dropout_rate: float
    time_emb_dim: int
    mlp_layers: List[Union[Callable, eqx.nn.Linear]]
    scaling: Union[None, Callable, eqx.nn.ConvTranspose2d, eqx.nn.Conv2d]
    block1_groupnorm: eqx.nn.GroupNorm
    block1_conv: eqx.nn.Conv2d
    block2_layers: List[
        Union[eqx.nn.GroupNorm, eqx.nn.Dropout, eqx.nn.Conv2d, Callable]
    ]
    res_conv: eqx.nn.Conv2d
    attn: Optional[Residual]

    def __init__(
        self,
        dim_in,
        dim_out,
        is_biggan,
        up,
        down,
        time_emb_dim,
        dropout_rate,
        is_attn,
        heads,
        dim_head,
        *,
        key
    ):
        keys = jax.random.split(key, 7)
        self.dim_out = dim_out
        self.is_biggan = is_biggan
        self.up = up
        self.down = down
        self.dropout_rate = dropout_rate
        self.time_emb_dim = time_emb_dim

        self.mlp_layers = [
            jax.nn.silu,
            eqx.nn.Linear(time_emb_dim, dim_out, key=keys[0]),
        ]
        self.block1_groupnorm = eqx.nn.GroupNorm(min(dim_in // 4, 32), dim_in)
        self.block1_conv = eqx.nn.Conv2d(dim_in, dim_out, 3, padding=1, key=keys[1])
        self.block2_layers = [
            eqx.nn.GroupNorm(min(dim_out // 4, 32), dim_out),
            jax.nn.silu,
            eqx.nn.Dropout(dropout_rate),
            eqx.nn.Conv2d(dim_out, dim_out, 3, padding=1, key=keys[2]),
        ]

        assert not self.up or not self.down

        if is_biggan:
            if self.up:
                self.scaling = upsample_2d
            elif self.down:
                self.scaling = downsample_2d
            else:
                self.scaling = None
        else:
            if self.up:
                self.scaling = eqx.nn.ConvTranspose2d(
                    dim_in,
                    dim_in,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    key=keys[3],
                )
            elif self.down:
                self.scaling = eqx.nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    key=keys[4],
                )
            else:
                self.scaling = None
        # For DDPM Yang use their own custom layer called NIN, which is
        # equivalent to a 1x1 conv
        self.res_conv = eqx.nn.Conv2d(dim_in, dim_out, kernel_size=1, key=keys[5])

        if is_attn:
            self.attn = Residual(
                LinearTimeSelfAttention(
                    dim_out,
                    heads=heads,
                    dim_head=dim_head,
                    key=keys[6],
                )
            )
        else:
            self.attn = None

    def __call__(self, x, t, *, key):

        C, _, _ = x.shape
        # In DDPM, each set of resblocks ends with an up/down sampling. In
        # biggan there is a final resblock after the up/downsampling. In this
        # code, the biggan approach is taken for both.
        # norm -> nonlinearity -> up/downsample -> conv follows Yang
        # https://github.dev/yang-song/score_sde/blob/main/models/layerspp.py
        h = jax.nn.silu(self.block1_groupnorm(x))
        if self.up or self.down:
            h = self.scaling(h)  # pyright: ignore
            x = self.scaling(x)  # pyright: ignore
        h = self.block1_conv(h)

        for layer in self.mlp_layers:
            t = layer(t)
        h = h + t[..., None, None]
        for layer in self.block2_layers:
            # Precisely 1 dropout layer in block2_layers which requires a key.
            if isinstance(layer, eqx.nn.Dropout):
                h = layer(h, key=key)
            else:
                h = layer(h)

        if C != self.dim_out or self.up or self.down:
            x = self.res_conv(x)

        out = (h + x) / jnp.sqrt(2)
        if self.attn is not None:
            out = self.attn(out)
        return out


class UNet(eqx.Module):
    time_pos_emb: SinusoidalPosEmb
    mlp: eqx.nn.MLP
    first_conv: eqx.nn.Conv2d
    down_res_blocks: List[List[ResnetBlock]]
    mid_block1: ResnetBlock
    mid_block2: ResnetBlock
    ups_res_blocks: List[List[ResnetBlock]]
    final_conv_layers: List[Union[Callable, eqx.nn.LayerNorm, eqx.nn.Conv2d]]

    def __init__(
        self,
        data_shape: Tuple[int, int, int],
        is_biggan: bool,
        dim_mults: List[int],
        hidden_size: int,
        heads: int,
        dim_head: int,
        dropout_rate: float,
        num_res_blocks: int,
        attn_resolutions: List[int],
        *,
        key
    ):

        keys = jax.random.split(key, 7)
        del key

        data_channels, in_height, in_width = data_shape

        dims = [hidden_size] + [hidden_size * m for m in dim_mults]
        in_out = list(exact_zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(hidden_size)
        self.mlp = eqx.nn.MLP(
            hidden_size,
            hidden_size,
            4 * hidden_size,
            1,
            activation=jax.nn.silu,
            key=keys[0],
        )
        self.first_conv = eqx.nn.Conv2d(
            data_channels, hidden_size, kernel_size=3, padding=1, key=keys[1]
        )

        h, w = in_height, in_width
        self.down_res_blocks = []
        num_keys = len(in_out) * num_res_blocks - 1
        keys_resblock = jr.split(keys[2], num_keys)
        i = 0
        for ind, (dim_in, dim_out) in enumerate(in_out):
            if h in attn_resolutions and w in attn_resolutions:
                is_attn = True
            else:
                is_attn = False
            res_blocks = [
                ResnetBlock(
                    dim_in=dim_in,
                    dim_out=dim_out,
                    is_biggan=is_biggan,
                    up=False,
                    down=False,
                    time_emb_dim=hidden_size,
                    dropout_rate=dropout_rate,
                    is_attn=is_attn,
                    heads=heads,
                    dim_head=dim_head,
                    key=keys_resblock[i],
                )
            ]
            i += 1
            for _ in range(num_res_blocks - 2):
                res_blocks.append(
                    ResnetBlock(
                        dim_in=dim_out,
                        dim_out=dim_out,
                        is_biggan=is_biggan,
                        up=False,
                        down=False,
                        time_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                        key=keys_resblock[i],
                    )
                )
                i += 1
            if ind < (len(in_out) - 1):
                res_blocks.append(
                    ResnetBlock(
                        dim_in=dim_out,
                        dim_out=dim_out,
                        is_biggan=is_biggan,
                        up=False,
                        down=True,
                        time_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                        key=keys_resblock[i],
                    )
                )
                i += 1
                h, w = h // 2, w // 2
            self.down_res_blocks.append(res_blocks)
        assert i == num_keys

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(
            dim_in=mid_dim,
            dim_out=mid_dim,
            is_biggan=is_biggan,
            up=False,
            down=False,
            time_emb_dim=hidden_size,
            dropout_rate=dropout_rate,
            is_attn=True,
            heads=heads,
            dim_head=dim_head,
            key=keys[3],
        )
        self.mid_block2 = ResnetBlock(
            dim_in=mid_dim,
            dim_out=mid_dim,
            is_biggan=is_biggan,
            up=False,
            down=False,
            time_emb_dim=hidden_size,
            dropout_rate=dropout_rate,
            is_attn=False,
            heads=heads,
            dim_head=dim_head,
            key=keys[4],
        )

        self.ups_res_blocks = []
        num_keys = len(in_out) * (num_res_blocks + 1) - 1
        keys_resblock = jr.split(keys[5], num_keys)
        i = 0
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            if h in attn_resolutions and w in attn_resolutions:
                is_attn = True
            else:
                is_attn = False
            res_blocks = []
            for _ in range(num_res_blocks - 1):
                res_blocks.append(
                    ResnetBlock(
                        dim_in=dim_out * 2,
                        dim_out=dim_out,
                        is_biggan=is_biggan,
                        up=False,
                        down=False,
                        time_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                        key=keys_resblock[i],
                    )
                )
                i += 1
            res_blocks.append(
                ResnetBlock(
                    dim_in=dim_out + dim_in,
                    dim_out=dim_in,
                    is_biggan=is_biggan,
                    up=False,
                    down=False,
                    time_emb_dim=hidden_size,
                    dropout_rate=dropout_rate,
                    is_attn=is_attn,
                    heads=heads,
                    dim_head=dim_head,
                    key=keys_resblock[i],
                )
            )
            i += 1
            if ind < (len(in_out) - 1):
                res_blocks.append(
                    ResnetBlock(
                        dim_in=dim_in,
                        dim_out=dim_in,
                        is_biggan=is_biggan,
                        up=True,
                        down=False,
                        time_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                        key=keys_resblock[i],
                    )
                )
                i += 1
                h, w = h * 2, w * 2

            self.ups_res_blocks.append(res_blocks)
        assert i == num_keys

        self.final_conv_layers = [
            eqx.nn.GroupNorm(min(hidden_size // 4, 32), hidden_size),
            jax.nn.silu,
            eqx.nn.Conv2d(hidden_size, data_channels, 1, key=keys[6]),
        ]

    def __call__(self, t, y, *, key=None):
        t = self.time_pos_emb(t)
        t = self.mlp(t)
        h = self.first_conv(y)
        hs = [h]
        for res_blocks in self.down_res_blocks:
            for res_block in res_blocks:
                key, subkey = key_split_allowing_none(key)
                h = res_block(h, t, key=subkey)
                hs.append(h)

        key, subkey = key_split_allowing_none(key)
        h = self.mid_block1(h, t, key=subkey)
        key, subkey = key_split_allowing_none(key)
        h = self.mid_block2(h, t, key=subkey)

        for res_blocks in self.ups_res_blocks:
            for res_block in res_blocks:
                key, subkey = key_split_allowing_none(key)
                if res_block.up:
                    h = res_block(h, t, key=subkey)
                else:
                    h = res_block(jnp.concatenate((h, hs.pop()), axis=0), t, key=subkey)

        assert len(hs) == 0

        for layer in self.final_conv_layers:
            h = layer(h)
        return h
