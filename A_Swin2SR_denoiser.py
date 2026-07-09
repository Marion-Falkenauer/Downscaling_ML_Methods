#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Swin2SR-based denoising backbone for the ERA5 -> CERRA diffusion model.
# Drop-in replacement for get_network() from denoising_unet_edit_V2.py, same functional signature so DiffusionModel.__init__ only needs one line changed.
#
# Differences vs the original Swin2SRGenerator_UV:
# - UpsampleX5 removed entirely, backbone runs at full HR resolution (320 x 400) throughout, bilinear pre-upsampling of LR already done in the data pipeline (same as the U-Net path)
# - Input is (B, 320, 400, input_frames + output_frames): channels 0:input_frames = bilinearly upsampled LR ERA5 (clean), channels input_frames: = noisy HR CERRA target
# - Sinusoidal noise-variance embedding (identical to the U-Net) injected after the shallow feature extraction, concatenated with the initial feature map and projected back to embed_dim
# - Output is (B, 320, 400, output_frames), predicted noise. Output head uses kernel_initializer='zeros' for stable early training (same convention as the U-Net)
#
# Swin2 specifics kept from Swin2SR: log-spaced Continuous Position Bias (CPB-MLP) instead of the learnable position table of Swin V1, scaled cosine attention with a learnable per-head temperature tau, post-normalization, Residual Swin Transformer Blocks (RSTB) with conv residual shortcut.
#
# window_size=8 check: 320/8=40 and 400/8=50, both integer, no padding needed.

import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.saving import register_keras_serializable

from setup_V2 import embedding_dims, embedding_max_frequency


# sinusoidal noise-variance embedding, identical to U-Net
@register_keras_serializable()
def sinusoidal_embedding_swin(x):
    # encodes noise variance sigma^2 scalar -> (B, 1, 1, embedding_dims), log-spaced sin/cos bank
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    return tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )

class SinusoidalEmbeddingLayer(layers.Layer):
    # replaces layers.Lambda(sinusoidal_embedding_swin), Lambda can't infer its output shape when input dtype is bfloat16 (NotImplementedError at build time)
    # always casts input to float32 first, frequency embeddings should stay full precision regardless of the mixed-precision policy
    supports_masking = True

    def call(self, x):
        return sinusoidal_embedding_swin(tf.cast(x, tf.float32))

    def compute_output_shape(self, input_shape):
        # (B, 1, 1, 1) -> (B, 1, 1, embedding_dims)
        return (input_shape[0], 1, 1, embedding_dims)

# window partition / reverse
def window_partition(x, window_size):
    # partitions a feature map into non-overlapping windows
    # x: (B, H, W, C) -> windows: (B * num_windows, window_size, window_size, C), num_windows = (H // window_size) * (W // window_size)
    B  = tf.shape(x)[0]
    H, W, C = x.shape[1], x.shape[2], x.shape[3]
    x = tf.reshape(x, [B,
                        H // window_size, window_size,
                        W // window_size, window_size,
                        C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])               # B, nH, nW, ws, ws, C
    windows = tf.reshape(x, [-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W):
    # reconstructs the feature map from windows, H and W are the static spatial dims of the original feature map
    nH = H // window_size
    nW = W // window_size
    C  = windows.shape[-1]
    x  = tf.reshape(windows, [-1, nH, nW, window_size, window_size, C])
    x  = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x  = tf.reshape(x, [-1, H, W, C])
    return x


# cyclic-shift attention mask, precomputed as numpy, stored as constant
def compute_attn_mask(H, W, window_size, shift_size):
    # boolean attention mask for shifted-window attention, pure NumPy since this runs at model build time and the result is stored as a tf.constant
    img_mask = np.zeros((1, H, W, 1), dtype=np.float32)
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    # pure NumPy window partition, replaces window_partition() which returns a TF EagerTensor and breaks the following .reshape()
    nH = H // window_size
    nW = W // window_size

    img_mask = img_mask.reshape(1, nH, window_size, nW, window_size, 1)
    img_mask = img_mask.transpose(0, 1, 3, 2, 4, 5)        # (1, nH, nW, ws, ws, 1)
    mask_windows = img_mask.reshape(-1, window_size * window_size)

    # pairwise region label difference, non-zero pairs must not attend
    attn_mask = mask_windows[:, :, np.newaxis] - mask_windows[:, np.newaxis, :]
    attn_mask = np.where(attn_mask != 0, -100.0, 0.0).astype(np.float32)
    return attn_mask        # (num_windows, ws^2, ws^2)


# MLP block
class Mlp(layers.Layer):
    # two-layer MLP with GELU activation, used inside each transformer block
    supports_masking = True
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 drop=0.0, **kwargs):
        super().__init__(**kwargs)
        hidden_features = hidden_features or in_features
        out_features    = out_features    or in_features
        self.fc1   = layers.Dense(hidden_features, activation="gelu")
        self.fc2   = layers.Dense(out_features)
        self.drop  = layers.Dropout(drop) if drop > 0.0 else None

    def call(self, x, training=False):
        x = self.fc1(x)
        if self.drop is not None:
            x = self.drop(x, training=training)
        x = self.fc2(x)
        if self.drop is not None:
            x = self.drop(x, training=training)
        return x


# Swin2 window attention with continuous position bias
class WindowAttentionSwin2(layers.Layer):
    # window-based multi-head self-attention with Swin V2 features:
    # 1. log-spaced Continuous Position Bias (CPB-MLP): relative coords -> log-transform -> 2-layer MLP -> per-head bias scalar
    # 2. scaled cosine attention: sim(Q,K) / (||Q|| ||K||) * exp(tau), tau learnable log-scale per head
    # replaces the learnable position-bias table from SwinV1, improves cross-window transferability when changing image resolution at inference
    supports_masking = True

    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim         = dim
        self.window_size = window_size           # Wh, Ww  (both equal here)
        self.num_heads   = num_heads
        self.qkv_bias    = qkv_bias

        # continuous position bias MLP, maps log-spaced relative coords (Wh*Ww)^2 x 2 -> num_heads
        self.cpb_mlp = keras.Sequential([
            layers.Dense(512, activation="relu", use_bias=True),
            layers.Dense(num_heads, use_bias=False),
        ], name="cpb_mlp")

        self.qkv     = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.proj    = layers.Dense(dim)
        self.attn_drop = layers.Dropout(attn_drop) if attn_drop > 0.0 else None
        self.proj_drop = layers.Dropout(proj_drop) if proj_drop > 0.0 else None

        # pre-compute log-spaced relative coordinates for CPB, shape (Wh*Ww, Wh*Ww, 2)
        coords_h = np.arange(window_size, dtype=np.float32)
        coords_w = np.arange(window_size, dtype=np.float32)
        grid_h, grid_w = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords   = np.stack([grid_h.ravel(), grid_w.ravel()], axis=-1)  # (ws^2, 2)
        rel      = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (ws^2, ws^2, 2)
        # log-spaced transform: sign(x) * log2(1 + |x|)
        rel_log  = np.sign(rel) * np.log2(1.0 + np.abs(rel))
        # stored as constant (1, ws^2, ws^2, 2) so CPB-MLP sees it as a batch
        self._rel_coords_log = tf.constant(
            rel_log[np.newaxis].astype(np.float32))   # (1, ws^2, ws^2, 2)

    def build(self, input_shape):
        # learnable log temperature tau, one scalar per head, initialised to 0
        self.logit_scale = self.add_weight(
            name="logit_scale",
            shape=(self.num_heads,),
            initializer=tf.initializers.Constant(
                math.log(10.0)),   # start near tau=10
            trainable=True)
        super().build(input_shape)

    def call(self, x, attn_mask=None, training=False):
        # cast all float32 constants and weights to input dtype at call time so this stays compatible with mixed precision (bfloat16 / float16)
        # self.logit_scale, self._rel_coords_log, and attn_mask are built as float32 but need to match x's dtype when mixed precision is on
        dtype = x.dtype   # bfloat16 under mixed precision, float32 otherwise

        Bn, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        head_dim = C // self.num_heads

        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [Bn, N, 3, self.num_heads, head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_norm = tf.math.l2_normalize(q, axis=-1)
        k_norm = tf.math.l2_normalize(k, axis=-1)

        # cast logit_scale to input dtype before tf.minimum
        tau = tf.exp(
            tf.minimum(
                tf.cast(self.logit_scale, dtype),          # cast here
                tf.constant(math.log(100.0), dtype=dtype)  # match dtype
            )
        )
        tau = tf.reshape(tau, [1, self.num_heads, 1, 1])
        attn = tf.matmul(q_norm, k_norm, transpose_b=True) * tau

        # cast rel_coords_log to input dtype for the CPB-MLP
        rel_coords = tf.cast(self._rel_coords_log, dtype)  # cast here
        cpb  = self.cpb_mlp(rel_coords)
        cpb  = tf.squeeze(cpb, axis=0)
        cpb  = tf.transpose(cpb, [2, 0, 1])
        cpb  = tf.reshape(cpb, [1, self.num_heads,
                                self.window_size ** 2,
                                self.window_size ** 2])
        cpb  = 16.0 * tf.nn.sigmoid(cpb)
        attn = attn + tf.cast(cpb, dtype)                  # cast result

        if attn_mask is not None:
            nW   = tf.shape(attn_mask)[0]
            attn = tf.reshape(attn, [-1, nW, self.num_heads, N, N])
            # cast the precomputed float32 mask to input dtype
            attn = attn + tf.cast(
                tf.reshape(attn_mask, [1, nW, 1, N, N]), dtype)  # cast here
            attn = tf.reshape(attn, [-1, self.num_heads, N, N])

        attn = tf.nn.softmax(attn, axis=-1)
        if self.attn_drop is not None:
            attn = self.attn_drop(attn, training=training)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [Bn, N, C])
        x = self.proj(x)
        if self.proj_drop is not None:
            x = self.proj_drop(x, training=training)
        return x

# Swin Transformer block, one attention + FFN, post-norm
class SwinTransformerBlock(layers.Layer):
    # single Swin Transformer layer with optional cyclic shift
    # uses post-normalization (LayerNorm after residual add) as in Swin2SR, improves training stability at larger model scales
    supports_masking = True

    def __init__(self, dim, num_heads, window_size, shift_size,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0,
                 H=320, W=400, **kwargs):
        super().__init__(**kwargs)
        self.dim        = dim
        self.num_heads  = num_heads
        self.window_size = window_size
        self.shift_size  = shift_size
        self.H, self.W   = H, W

        self.attn = WindowAttentionSwin2(
            dim=dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            name="attn")
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp   = Mlp(dim, hidden_features=int(dim * mlp_ratio),
                         drop=drop, name="mlp")
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        # precompute attention mask for shifted windows (numpy -> tf constant)
        if shift_size > 0:
            mask_np = compute_attn_mask(H, W, window_size, shift_size)
            self._attn_mask = tf.constant(mask_np, dtype=tf.float32)
        else:
            self._attn_mask = None

    def call(self, x, training=False):
        # x : (B, H, W, C)
        B = tf.shape(x)[0]
        H, W, C = self.H, self.W, x.shape[-1]
        shortcut = x

        # cyclic shift
        if self.shift_size > 0:
            x_shifted = tf.roll(x, shift=[-self.shift_size, -self.shift_size],
                                axis=[1, 2])
        else:
            x_shifted = x

        # partition windows: (B*nW, ws, ws, C) -> (B*nW, ws^2, C)
        x_win = window_partition(x_shifted, self.window_size)
        x_win = tf.reshape(x_win, [-1, self.window_size * self.window_size, C])

        # self-attention
        x_win = self.attn(x_win, attn_mask=self._attn_mask, training=training)

        # reverse windows, back to (B, H, W, C)
        x_win = tf.reshape(x_win, [-1, self.window_size, self.window_size, C])
        x     = window_reverse(x_win, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(x, shift=[self.shift_size, self.shift_size], axis=[1, 2])

        # residual + post-norm
        x = shortcut + x
        x = self.norm1(x)

        # FFN
        x = x + self.mlp(x, training=training)
        x = self.norm2(x)
        return x


# stack of SwinTransformerBlocks
class BasicLayer(layers.Layer):
    # one stage: `depth` consecutive SwinTransformerBlocks, alternates between W-MSA (no shift) and SW-MSA (with shift)
    supports_masking = True
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0,
                 H=320, W=400, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                H=H, W=W,
                name=f"block_{i}")
            for i in range(depth)
        ]

    def call(self, x, training=False):
        for blk in self.blocks:
            x = blk(x, training=training)
        return x


# Residual Swin Transformer Block (RSTB)
class RSTB(layers.Layer):
    # BasicLayer (N transformer blocks) + Conv2D(3x3) residual shortcut, identical to the RSTB in the original Swin2SR
    supports_masking = True
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0,
                 H=320, W=400, res_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.res_scale   = res_scale
        self.basic_layer = BasicLayer(
            dim=dim, depth=depth, num_heads=num_heads,
            window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
            H=H, W=W, name="basic_layer")
        self.conv = layers.Conv2D(dim, kernel_size=3, padding="same",
                                  name="residual_conv")
        self.norm = layers.LayerNormalization(epsilon=1e-5)

    def call(self, x, training=False):
        residual = x
        x = self.basic_layer(x, training=training)
        x = self.conv(x)
        x = self.norm(x)
        return residual + self.res_scale * x


# factory function, drop-in for get_network()
def get_swin2_denoising_network(
        image_size_height,
        input_frames,
        output_frames,
        image_size_width,
        embed_dim   = 180,
        depths      = (6, 6, 6, 6, 6, 6),
        num_heads   = (6, 6, 6, 6, 6, 6),
        window_size = 8,
        mlp_ratio   = 4.0,
        qkv_bias    = True,
        drop_rate   = 0.0,
        attn_drop   = 0.0,
):
    # builds the Swin2SR denoising backbone, drop-in replacement for get_network() from denoising_unet_edit_V2.py
    # image_size_height/width: HR pixels, e.g. 320 x 400. input_frames: clean LR channels. output_frames: noisy HR channels.
    # embed_dim: channel width throughout. depths/num_heads: per RSTB stage. window_size must divide H and W.
    # returns a keras.Model with inputs [noisy_images (B,H,W,input_frames+output_frames), noise_variances (B,1,1,1)] and output predicted_noise (B,H,W,output_frames)
    H = image_size_height
    W = image_size_width

    assert H % window_size == 0, \
        f"image_size_height ({H}) must be divisible by window_size ({window_size})."
    assert W % window_size == 0, \
        f"image_size_width ({W}) must be divisible by window_size ({window_size})."
    assert len(depths) == len(num_heads), \
        "depths and num_heads must have the same length (one entry per RSTB stage)."

    # inputs
    noisy_images    = keras.Input(
        shape=(H, W, input_frames + output_frames),
        name="noisy_images")
    noise_variances = keras.Input(
        shape=(1, 1, 1),
        name="noise_variances")

    # sinusoidal noise-variance embedding, (B,1,1,1) -> (B,1,1,embedding_dims) -> broadcast to (B,H,W,embedding_dims)
    e = SinusoidalEmbeddingLayer(name="sinusoidal_embedding")(noise_variances)
    e = layers.UpSampling2D(size=(H, W), interpolation="nearest",
                             name="embedding_broadcast")(e)
    # e : (B, H, W, embedding_dims)

    # shallow feature extraction, project image channels to embed_dim, concat noise embedding, project back to embed_dim so the rest of the network sees a uniform channel width
    x = layers.Conv2D(embed_dim, kernel_size=3, padding="same",
                      name="shallow_conv")(noisy_images)
    # concat noise embedding: (B, H, W, embed_dim + embedding_dims)
    x = layers.Concatenate(name="cat_noise_embed")([x, e])
    # project back to embed_dim: (B, H, W, embed_dim)
    x = layers.Conv2D(embed_dim, kernel_size=1, padding="same",
                      name="noise_proj")(x)
    # keep a copy of the shallow features for the global residual
    shallow = x

    # deep feature extraction, RSTB stages
    for i, (d, nh) in enumerate(zip(depths, num_heads)):
        x = RSTB(
            dim=embed_dim,
            depth=d,
            num_heads=nh,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop,
            H=H, W=W,
            res_scale=1.0,
            name=f"rstb_{i}"
        )(x)

    # feature fusion + global residual, conv after all RSTB stages then add shallow residual (mirrors Swin2SR)
    x = layers.Conv2D(embed_dim, kernel_size=3, padding="same",
                      name="conv_after_body")(x)
    x = layers.Add(name="global_residual")([x, shallow])

    # output head, zero-init keeps predicted noise near zero at the start of training, stabilises the first few epochs (same convention as the U-Net)
    predicted_noise = layers.Conv2D(
        output_frames,
        kernel_size=3,
        padding="same",
        kernel_initializer="zeros",
        name="output_head"
    )(x)

    return keras.Model(
        inputs  = [noisy_images, noise_variances],
        outputs = predicted_noise,
        name    = "swin2sr_denoiser"
    )


# convenience defaults matching the original Swin2SRGenerator_UV, large config: ~60M params at embed_dim=180
SWIN_LARGE_CONFIG = dict(
    embed_dim   = 180,
    depths      = (6, 6, 6, 6, 6, 6),
    num_heads   = (6, 6, 6, 6, 6, 6),
    window_size = 8,
    mlp_ratio   = 4.0,
)

# smaller config for faster experimentation, ~8M params
SWIN_SMALL_CONFIG = dict(
    embed_dim   = 60,
    depths      = (6, 6, 6, 6),
    num_heads   = (6, 6, 6, 6),
    window_size = 8,
    mlp_ratio   = 2.0,
)
