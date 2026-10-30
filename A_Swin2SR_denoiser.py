"""Swin2SR denoising network for the ERA5 to CERRA diffusion model, which is imported in the main scripts.

This code is adapted from the PyTorch implementation of the Swin2SR network.

get_swin2_denoising_network has the same interface as denoising_unet_edit.get_network, so the
DiffusionModel of A_Swin2SR_Diff_V3.py only changes the line that builds its network.

The network works at the full HR resolution (320 x 400, window size 8 divides both sizes).
Input channels: the LR ERA5 channels bilinearly upsampled to the HR grid followed by the noisy HR channels. 
The output is the predicted noise.

SWIN_LARGE_CONFIG and SWIN_SMALL_CONFIG are ready-made hyperparameter sets, to swtich the size of the network to work it on different GPU 
(big size very bulky, only works on the 24G ones)

Author: Marion Falkenauer (with the help of Claude)
"""

import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.saving import register_keras_serializable

from setup import embedding_dims, embedding_max_frequency


@register_keras_serializable()
def sinusoidal_embedding_swin(x):
    """Sinusoidal embedding of the noise variance.

    Args:
        x: tensor (B, 1, 1, 1) of noise variances.
    Returns:
        Tensor (B, 1, 1, embedding_dims) of sine and cosine features at log-spaced frequencies.
    """
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
    """Layer computing the noise-variance embedding in float32."""

    supports_masking = True

    def call(self, x):
        """Embed the noise variance (the input is cast to float32)."""
        return sinusoidal_embedding_swin(tf.cast(x, tf.float32))

    def compute_output_shape(self, input_shape):
        """Output shape (B, 1, 1, embedding_dims)."""
        return (input_shape[0], 1, 1, embedding_dims)


def window_partition(x, window_size):
    """Split a feature map into non-overlapping windows.

    Args:
        x: tensor (B, H, W, C).
        window_size: window side.
    Returns:
        Tensor (B * num_windows, window_size, window_size, C).
    """
    B  = tf.shape(x)[0]
    H, W, C = x.shape[1], x.shape[2], x.shape[3]
    x = tf.reshape(x, [B,
                        H // window_size, window_size,
                        W // window_size, window_size,
                        C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])  # B, nH, nW, ws, ws, C
    windows = tf.reshape(x, [-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W):
    """Reassemble a feature map from its windows.

    Args:
        windows: tensor (B * num_windows, window_size, window_size, C).
        window_size: window side.
        H, W: size of the original feature map.
    Returns:
        Tensor (B, H, W, C).
    """
    nH = H // window_size
    nW = W // window_size
    C  = windows.shape[-1]
    x  = tf.reshape(windows, [-1, nH, nW, window_size, window_size, C])
    x  = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x  = tf.reshape(x, [-1, H, W, C])
    return x


def compute_attn_mask(H, W, window_size, shift_size):
    """Attention mask of the shifted-window attention

    Args:
        H, W: feature map size.
        window_size: window side.
        shift_size: cyclic shift in pixels.
    Returns:
        Array (num_windows, window_size^2, window_size^2), True where attention is not allowed.
    """
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

    # Pure numpy window partition (a TF tensor would break the reshape, types are important to consider with jit compilation mode)
    nH = H // window_size
    nW = W // window_size

    img_mask = img_mask.reshape(1, nH, window_size, nW, window_size, 1)
    img_mask = img_mask.transpose(0, 1, 3, 2, 4, 5)  # (1, nH, nW, ws, ws, 1)
    mask_windows = img_mask.reshape(-1, window_size * window_size)

    # Pairwise region label difference - non-zero pairs must not attend
    attn_mask = mask_windows[:, :, np.newaxis] - mask_windows[:, np.newaxis, :]
    attn_mask = np.where(attn_mask != 0, -100.0, 0.0).astype(np.float32)
    return attn_mask  # (num_windows, ws^2, ws^2)


class Mlp(layers.Layer):
    """Two-layer MLP with GELU activation functions used in each transformer block."""

    supports_masking = True
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 drop=0.0, **kwargs):
        """Build the layers.

        Args:
            in_features: input width.
            hidden_features: hidden width (default in_features).
            out_features: output width (default in_features).
            drop: dropout rate, which allow to decrease overfitting likelihood
        """
        super().__init__(**kwargs)
        hidden_features = hidden_features or in_features
        out_features    = out_features    or in_features
        self.fc1   = layers.Dense(hidden_features, activation="gelu")
        self.fc2   = layers.Dense(out_features)
        self.drop  = layers.Dropout(drop) if drop > 0.0 else None

    def call(self, x, training=False):
        """Apply the MLP to a tensor of shape (..., in_features)."""
        x = self.fc1(x)
        if self.drop is not None:
            x = self.drop(x, training=training)
        x = self.fc2(x)
        if self.drop is not None:
            x = self.drop(x, training=training)
        return x


class WindowAttentionSwin2(layers.Layer):
    """Window multi-head self-attention of Swin2SR."""

    supports_masking = True

    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0, **kwargs):
        """Build the projections of the QKV maps, etc

        Args:
            dim: channel width.
            window_size: window side.
            num_heads: number of heads.
            qkv_bias: use a bias in the QKV projection.
            attn_drop, proj_drop: dropout rates of the attention weights and of the output projection.
        """
        super().__init__(**kwargs)
        self.dim         = dim
        self.window_size = window_size  # Wh, Ww (both equal here)
        self.num_heads   = num_heads
        self.qkv_bias    = qkv_bias

        # Continuous Position Bias MLP (CPB)
        self.cpb_mlp = keras.Sequential([
            layers.Dense(512, activation="relu", use_bias=True),
            layers.Dense(num_heads, use_bias=False),
        ], name="cpb_mlp")

        self.qkv     = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.proj    = layers.Dense(dim)
        self.attn_drop = layers.Dropout(attn_drop) if attn_drop > 0.0 else None
        self.proj_drop = layers.Dropout(proj_drop) if proj_drop > 0.0 else None

        # Pre-compute log-spaced relative coordinates for CPB
        coords_h = np.arange(window_size, dtype=np.float32)
        coords_w = np.arange(window_size, dtype=np.float32)
        grid_h, grid_w = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords   = np.stack([grid_h.ravel(), grid_w.ravel()], axis=-1)  # (ws^2, 2)
        rel      = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (ws^2, ws^2, 2)
        # Log-spaced transform: sign(x) * log2(1 + |x|)
        rel_log  = np.sign(rel) * np.log2(1.0 + np.abs(rel))
        # Store as a constant: (1, ws^2, ws^2, 2) so CPB-MLP sees it as a batch
        self._rel_coords_log = tf.constant(
            rel_log[np.newaxis].astype(np.float32))  # (1, ws^2, ws^2, 2)

    def build(self, input_shape):
        # Learnable log temperature tau: one scalar per head, initialised to log(10)
        self.logit_scale = self.add_weight(
            name="logit_scale",
            shape=(self.num_heads,),
            initializer=tf.initializers.Constant(
                math.log(10.0)),
            trainable=True)
        super().build(input_shape)

    def call(self, x, attn_mask=None, training=False):
        """Apply the window attention, casting constants to the input dtype (mixed precision).

        Args:
            x: windows (B * num_windows, window_size^2, C).
            attn_mask: optional mask of the shifted windows.
            training: training flag.
        Returns:
            Tensor with the shape of x.
        """
        dtype = x.dtype  # bfloat16 under mixed precision, float32 otherwise. 
        #It is to avoid mixed types during operations errors when switching between float16 (more compactly stored) and float32 (more precision needed for some computations)

        Bn, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        head_dim = C // self.num_heads

        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [Bn, N, 3, self.num_heads, head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_norm = tf.math.l2_normalize(q, axis=-1)
        k_norm = tf.math.l2_normalize(k, axis=-1)

        # Cast logit_scale to input dtype before tf.minimum
        tau = tf.exp(
            tf.minimum(
                tf.cast(self.logit_scale, dtype),
                tf.constant(math.log(100.0), dtype=dtype)
            )
        )
        tau = tf.reshape(tau, [1, self.num_heads, 1, 1])
        attn = tf.matmul(q_norm, k_norm, transpose_b=True) * tau

        # Cast rel_coords_log to input dtype for the CPB-MLP
        rel_coords = tf.cast(self._rel_coords_log, dtype)
        cpb  = self.cpb_mlp(rel_coords)
        cpb  = tf.squeeze(cpb, axis=0)
        cpb  = tf.transpose(cpb, [2, 0, 1])
        cpb  = tf.reshape(cpb, [1, self.num_heads,
                                self.window_size ** 2,
                                self.window_size ** 2])
        cpb  = 16.0 * tf.nn.sigmoid(cpb)
        attn = attn + tf.cast(cpb, dtype)

        if attn_mask is not None:
            nW   = tf.shape(attn_mask)[0]
            attn = tf.reshape(attn, [-1, nW, self.num_heads, N, N])
            # Cast the precomputed float32 mask to input dtype
            attn = attn + tf.cast(
                tf.reshape(attn_mask, [1, nW, 1, N, N]), dtype)
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


class SwinTransformerBlock(layers.Layer):
    """Swin transformer layer with an optional cyclic shift and post-normalisation."""

    supports_masking = True

    def __init__(self, dim, num_heads, window_size, shift_size,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0,
                 H=320, W=400, **kwargs):
        """Build the attention, MLP and normalisation layers.

        Args:
            dim: channel width.
            num_heads: number of heads.
            window_size: window side.
            shift_size: cyclic shift in pixels (0 for no shift).
            mlp_ratio: hidden width of the MLP relative to dim.
            qkv_bias: use a bias in the QKV projection.
            drop, attn_drop: dropout rates.
            H, W: feature map size.
        """
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

        if shift_size > 0:
            mask_np = compute_attn_mask(H, W, window_size, shift_size)
            self._attn_mask = tf.constant(mask_np, dtype=tf.float32)
        else:
            self._attn_mask = None

    def call(self, x, training=False):
        """Apply shifted-window attention and the MLP with residual connections. (big connections between blocks to allow some bigger features to pass through the network)

        Args:
            x: tensor (B, H, W, C).
            training: training flag.
        Returns:
            Tensor (B, H, W, C).
        """
        B = tf.shape(x)[0]
        H, W, C = self.H, self.W, x.shape[-1]
        shortcut = x

        # Cyclic shift
        if self.shift_size > 0:
            x_shifted = tf.roll(x, shift=[-self.shift_size, -self.shift_size],
                                axis=[1, 2])
        else:
            x_shifted = x

        x_win = window_partition(x_shifted, self.window_size)
        x_win = tf.reshape(x_win, [-1, self.window_size * self.window_size, C])

        # Self-attention
        x_win = self.attn(x_win, attn_mask=self._attn_mask, training=training)

        # Reverse windows: back to the shape (B, H, W, C)
        x_win = tf.reshape(x_win, [-1, self.window_size, self.window_size, C])
        x     = window_reverse(x_win, self.window_size, H, W)


        if self.shift_size > 0:
            x = tf.roll(x, shift=[self.shift_size, self.shift_size], axis=[1, 2])

        # Residual + post-normalisation
        x = shortcut + x
        x = self.norm1(x)

        x = x + self.mlp(x, training=training)
        x = self.norm2(x)
        return x


class BasicLayer(layers.Layer):
    """Stage of consecutive Swin transformer blocks, alternating unshifted and shifted windows."""

    supports_masking = True
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0,
                 H=320, W=400, **kwargs):
        """Build the blocks.

        Args:
            dim: channel width.
            depth: number of blocks.
            num_heads: number of heads.
            window_size: window side.
            mlp_ratio, qkv_bias, drop, attn_drop: block settings.
            H, W: feature map size.
        """
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
        """Apply the blocks in sequence to a tensor (B, H, W, C)."""
        for blk in self.blocks:
            x = blk(x, training=training)
        return x


class RSTB(layers.Layer):
    """Residual Swin transformer block: a BasicLayer followed by a 3x3 convolution and a scaled shortcut."""

    supports_masking = True
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0,
                 H=320, W=400, res_scale=1.0, **kwargs):
        """Build the BasicLayer and the convolution.

        Args:
            dim: channel width.
            depth: number of transformer blocks.
            num_heads: number of heads.
            window_size: window side.
            mlp_ratio, qkv_bias, drop, attn_drop: block settings.
            H, W: feature map size.
            res_scale: scale of the BasicLayer branch.
        """
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
        """Apply the block to a tensor (B, H, W, C)."""
        residual = x
        x = self.basic_layer(x, training=training)
        x = self.conv(x)
        x = self.norm(x)
        return residual + self.res_scale * x


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
    """Build the Swin2SR denoising network of the diffusion model.

    Args:
        image_size_height, image_size_width: HR grid size, multiples of window_size.
        input_frames: number of clean LR channels.
        output_frames: number of noisy HR channels.
        embed_dim: channel width.
        depths: number of transformer blocks of each RSTB stage.
        num_heads: number of heads of each stage.
        window_size: attention window side.
        mlp_ratio: hidden width of the MLP relative to embed_dim.
        qkv_bias: use a bias in the QKV projection.
        drop_rate, attn_drop: dropout rates.
    Returns:
        keras.Model taking [noisy_images (B, H, W, input_frames + output_frames),
        noise_variances (B, 1, 1, 1)] and returning the predicted noise (B, H, W, output_frames).
    """
    H = image_size_height
    W = image_size_width

    assert H % window_size == 0, \
        f"image_size_height ({H}) must be divisible by window_size ({window_size})."
    assert W % window_size == 0, \
        f"image_size_width ({W}) must be divisible by window_size ({window_size})."
    assert len(depths) == len(num_heads), \
        "depths and num_heads must have the same length (one entry per RSTB stage)."

    # Inputs
    noisy_images    = keras.Input(
        shape=(H, W, input_frames + output_frames),
        name="noisy_images")
    noise_variances = keras.Input(
        shape=(1, 1, 1),
        name="noise_variances")

    # Sinusoidal noise-variance embedding
    e = SinusoidalEmbeddingLayer(name="sinusoidal_embedding")(noise_variances)
    e = layers.UpSampling2D(size=(H, W), interpolation="nearest",
                             name="embedding_broadcast")(e)
    # e : (B, H, W, embedding_dims)

    # Shallow feature extraction
    # Project the image channels to embed_dim, concatenate the noise embedding and project back to embed_dim
    x = layers.Conv2D(embed_dim, kernel_size=3, padding="same",
                      name="shallow_conv")(noisy_images)
    # Concatenate noise embedding: (B, H, W, embed_dim + embedding_dims)
    x = layers.Concatenate(name="cat_noise_embed")([x, e])
    # Project back to embed_dim: (B, H, W, embed_dim)
    x = layers.Conv2D(embed_dim, kernel_size=1, padding="same",
                      name="noise_proj")(x)
    # Keep a copy of the shallow features for the global residual
    shallow = x

    # Deep feature extraction - RSTB stages
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

    # Feature fusion + global residual
    # Conv after all RSTB stages, then add shallow residual (mirrors Swin2SR)
    x = layers.Conv2D(embed_dim, kernel_size=3, padding="same",
                      name="conv_after_body")(x)
    x = layers.Add(name="global_residual")([x, shallow])

    # Output head
    # Zero initialisation keeps the predicted noise near zero at the start of training
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

# Two configuraitons available here so there is no need to switch manually the embedding dimension and number of heads.

SWIN_LARGE_CONFIG = dict(
    embed_dim   = 180,
    depths      = (6, 6, 6, 6, 6, 6),
    num_heads   = (6, 6, 6, 6, 6, 6),
    window_size = 8,
    mlp_ratio   = 4.0,
)

SWIN_SMALL_CONFIG = dict(
    embed_dim   = 60,
    depths      = (6, 6, 6, 6),
    num_heads   = (6, 6, 6, 6),
    window_size = 8,
    mlp_ratio   = 2.0,
)
