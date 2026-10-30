"""Swin2SR (SwinV2 transformer)

Tensorflow version of the PyTorch Swin2SR of Conde et al., "Swin2SR: SwinV2 Transformer for Compressed Image
Super-Resolution and Restoration", ECCV AIM 2022 (https://github.com/mv-lab/swin2sr).
Tensors are BHWC; the model upsamples by 5 (custom): (B, 64, 80, C) to (B, 320, 400, C).

Main entry points:
    build_swin2sr, build_swin2sr_uv: Swin2SR for one channel or for the u and v channels.
    build_swin2sr_uv_spectral: u and v model trained with MSE plus an FFT loss.
    Swin2SRGenerator, Swin2SRGenerator_UV, Swin2SRGenerator_UV_Spectral: wrappers that build the
        model, optionally load weights and expose predict and save.
    train_step, infer: one training step and numpy inference.

Author: Marion Falkenauer (with the help of Claude)
"""

import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Optional, Tuple


def _to_2tuple(x):
    """Return (x, x) for an int, or x as a tuple."""
    return (x, x) if isinstance(x, int) else tuple(x)


def window_partition(x, window_size):
    """Split a feature map into non-overlapping windows.

    Args:
        x: tensor (B, H, W, C).
        window_size: window side.
    Returns:
        Tensor (num_windows * B, window_size, window_size, C).
    """
    B = tf.shape(x)[0]
    H = tf.shape(x)[1]
    W = tf.shape(x)[2]
    C = tf.shape(x)[3]
    x = tf.reshape(x, [B, H // window_size, window_size,
                        W // window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, [-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W):
    """Reassemble a feature map from its windows.

    Args:
        windows: tensor (num_windows * B, window_size, window_size, C).
        window_size: window side.
        H, W: size of the original feature map.
    Returns:
        Tensor (B, H, W, C).
    """
    C = tf.shape(windows)[-1]
    x = tf.reshape(windows, [-1, H // window_size, W // window_size,
                              window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [-1, H, W, C])
    return x


class DropPath(layers.Layer):
    """Stochastic depth: randomly drops the residual branch of each sample during training."""

    def __init__(self, drop_prob=0., **kwargs):
        """Store the drop probability."""
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=False):
        """Scale the kept samples by 1 / keep probability and zero the dropped ones (training only)."""
        if not training or self.drop_prob == 0.:
            return x
        keep = 1.0 - self.drop_prob
        shape = [tf.shape(x)[0]] + [1] * (len(x.shape) - 1)
        random_tensor = tf.random.uniform(shape, dtype=x.dtype)
        random_tensor = tf.math.floor(random_tensor + keep)
        return x * random_tensor / keep

    def get_config(self):
        """Configuration including drop_prob."""
        config = super().get_config()
        config.update({'drop_prob': self.drop_prob})
        return config


class WindowAttentionV2(layers.Layer):
    """Window multi-head self-attention of SwinV2.

    Uses cosine attention scaled by a learnable temperature per head, a log-spaced continuous
    position bias computed by an MLP, and an optional shifted-window mask.
    """

    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=0, **kwargs):
        """Build the projections, the position-bias MLP and the relative position tables.

        Args:
            dim: channel width.
            window_size: window side (int or tuple).
            num_heads: number of heads.
            qkv_bias: use separate q and v biases.
            attn_drop, proj_drop: dropout rates.
            pretrained_window_size: window size of the pretraining (0 uses window_size).
        """
        super().__init__(**kwargs)
        self.supports_masking = True
        self.dim = dim
        self.window_size = _to_2tuple(window_size)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self._qkv_bias = qkv_bias

        # Continuous position bias MLP (same topology as PyTorch version)
        self.cpb_mlp = keras.Sequential([
            layers.Dense(512, use_bias=True, activation='relu', name='cpb_fc1'),
            layers.Dense(num_heads, use_bias=False, name='cpb_fc2'),
        ], name='cpb_mlp')

        # Precompute relative_coords_table as a tf.constant
        ws = window_size
        coords_h = np.arange(-(ws - 1), ws, dtype=np.float32)
        coords_w = np.arange(-(ws - 1), ws, dtype=np.float32)
        cH, cW = np.meshgrid(coords_h, coords_w, indexing='ij')
        table = np.stack([cH, cW], axis=-1)[np.newaxis]  # (1, 2Wh-1, 2Ww-1, 2)
        denom = (pretrained_window_size - 1) if pretrained_window_size > 0 else (ws - 1)
        table[..., 0] /= denom
        table[..., 1] /= denom
        table *= 8
        table = np.sign(table) * np.log2(np.abs(table) + 1.0) / 3.0
        self.relative_coords_table = tf.constant(table, dtype=tf.float32)

        # Precompute relative_position_index as a tf.constant
        ch = np.arange(ws)
        cw = np.arange(ws)
        gH, gW = np.meshgrid(ch, cw, indexing='ij')
        coords = np.stack([gH.ravel(), gW.ravel()])  # (2, N)
        rel = coords[:, :, None] - coords[:, None, :]  # (2, N, N)
        rel = rel.transpose(1, 2, 0)  # (N, N, 2)
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.relative_position_index = tf.constant(
            rel.sum(-1).astype(np.int32))  # (N, N)

        # Projection layers
        self.qkv      = layers.Dense(dim * 3, use_bias=False, name='qkv')
        self.proj      = layers.Dense(dim, name='proj')
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj_drop = layers.Dropout(proj_drop)

    def build(self, input_shape):
        """Create the q and v biases and the log temperature of each head."""
        if self._qkv_bias:
            self.q_bias = self.add_weight(
                shape=(self.dim,), initializer='zeros',
                trainable=True, name='q_bias')
            self.v_bias = self.add_weight(
                shape=(self.dim,), initializer='zeros',
                trainable=True, name='v_bias')
        else:
            self.q_bias = None
            self.v_bias = None

        self.logit_scale = self.add_weight(
            shape=(self.num_heads, 1, 1),
            initializer=tf.keras.initializers.Constant(math.log(10.0)),
            trainable=True, name='logit_scale')
        super().build(input_shape)

    def call(self, x, mask=None, training=False):
        """Apply the window attention.

        Args:
            x: windows (num_windows * B, N, C) with N = window_size^2.
            mask: optional shifted-window mask (num_windows, N, N).
            training: training flag.
        Returns:
            Tensor (num_windows * B, N, C).
        """
        # (B_, N, C)
        B_ = tf.shape(x)[0]
        N  = tf.shape(x)[1]
        C  = self.dim

        # QKV projection with optional per-component bias
        qkv_out = self.qkv(x)  # (B_, N, 3C)
        if self.q_bias is not None:
            qkv_bias = tf.concat(
                [self.q_bias, tf.zeros_like(self.v_bias), self.v_bias], axis=0)
            qkv_out = qkv_out + qkv_bias

        qkv_out = tf.reshape(qkv_out, [B_, N, 3, self.num_heads, self.head_dim])
        qkv_out = tf.transpose(qkv_out, [2, 0, 3, 1, 4])  # (3, B_, heads, N, hd)
        q, k, v = tf.unstack(qkv_out, axis=0)  # each (B_, heads, N, hd)

        # Scaled cosine attention
        q_norm = tf.math.l2_normalize(q, axis=-1)
        k_norm = tf.math.l2_normalize(k, axis=-1)
        attn = tf.matmul(q_norm, k_norm, transpose_b=True)

        logit_scale = tf.exp(
            tf.clip_by_value(self.logit_scale,-1e9, clip_value_max=math.log(1. / 0.01)))
        attn = attn * logit_scale  # (B_, heads, N, N)

        # Continuous position bias
        bias_table = self.cpb_mlp(
            self.relative_coords_table, training=training)  # (1, 2W-1, 2W-1, heads)
        bias_table = tf.reshape(bias_table, [-1, self.num_heads])
        idx        = tf.reshape(self.relative_position_index, [-1])
        rel_bias   = tf.gather(bias_table, idx)
        N_win      = self.window_size[0] * self.window_size[1]
        rel_bias   = tf.reshape(rel_bias, [N_win, N_win, self.num_heads])
        rel_bias   = tf.transpose(rel_bias, [2, 0, 1])  # (heads, N, N)
        rel_bias   = 16.0 * tf.sigmoid(rel_bias)
        attn = attn + tf.expand_dims(rel_bias, 0)  # broadcast over B_

        # Cyclic-shift attention mask (SW-MSA only)
        if mask is not None:
            nW   = tf.shape(mask)[0]
            attn = tf.reshape(attn, [-1, nW, self.num_heads, N, N])
            attn = attn + tf.cast(
                tf.expand_dims(tf.expand_dims(mask, 1), 0), attn.dtype)
            attn = tf.reshape(attn, [-1, self.num_heads, N, N])

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        x = tf.matmul(attn, v)  # (B_, heads, N, hd)
        x = tf.transpose(x, [0, 2, 1, 3])  # (B_, N, heads, hd)
        x = tf.reshape(x, [B_, N, C])
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x


class SwinTransformerV2Layer(layers.Layer):
    """SwinV2 block: window attention (shifted or not) and MLP, with post-normalisation."""

    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., pretrained_window_size=0, **kwargs):
        """Build the attention, MLP, normalisation and drop-path layers.

        Args:
            dim: channel width.
            num_heads: number of heads.
            window_size: window side.
            shift_size: cyclic shift in pixels (0 for no shift).
            mlp_ratio: hidden width of the MLP relative to dim.
            qkv_bias: use separate q and v biases.
            drop, attn_drop: dropout rates.
            drop_path: stochastic depth rate.
            pretrained_window_size: window size of the pretraining.
        """
        super().__init__(**kwargs)
        self.dim         = dim
        self.window_size = window_size
        self.shift_size  = shift_size

        self.norm1 = layers.LayerNormalization(epsilon=1e-5, name='norm1')
        self.attn  = WindowAttentionV2(
            dim=dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=pretrained_window_size, name='attn')
        self.drop_path_layer = DropPath(drop_path, name='drop_path') if drop_path > 0. else None
        self.norm2 = layers.LayerNormalization(epsilon=1e-5, name='norm2')

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = keras.Sequential([
            layers.Dense(mlp_hidden, activation='gelu', name='mlp_fc1'),
            layers.Dropout(drop),
            layers.Dense(dim, name='mlp_fc2'),
            layers.Dropout(drop),
        ], name='mlp')

    def _dp(self, x, training):
        """Apply drop path when enabled."""
        return self.drop_path_layer(x, training=training) if self.drop_path_layer else x
    def _get_attn_mask(self, H, W):
        """Build the shifted-window attention mask with TensorFlow operations (eager and graph mode).

        Args:
            H, W: padded feature map size.
        Returns:
            Tensor (num_windows, N, N) with 0 for allowed pairs and -100 otherwise.
        """
        if self.shift_size == 0:
            return None

        ws = self.window_size
        ss = self.shift_size

        h_regions = tf.concat([
            tf.zeros ([H - ws,  1], dtype=tf.float32),
            tf.ones  ([ws - ss, 1], dtype=tf.float32),
            tf.fill  ([ss,      1], 2.0),
        ], axis=0)  # (H, 1)

        w_regions = tf.concat([
            tf.zeros ([1, W - ws ], dtype=tf.float32),
            tf.ones  ([1, ws - ss], dtype=tf.float32),
            tf.fill  ([1, ss     ], 2.0),
        ], axis=1)  # (1, W)

        # Combine into unique region IDs 0-8 (matches original cnt=0..8 ordering)
        img_mask = h_regions * 3.0 + w_regions  # (H, W)
        img_mask = tf.reshape(img_mask, [1, H, W, 1])

        # Partition into windows and compute mask
        mask_windows = window_partition(img_mask, ws)  # (nW, ws, ws, 1)
        N            = ws * ws
        mask_windows = tf.reshape(mask_windows, [-1, N])  # (nW, N)

        attn_mask = (tf.expand_dims(mask_windows, 1)
                - tf.expand_dims(mask_windows, 2))  # (nW, N, N)
        attn_mask = tf.where(
            tf.not_equal(attn_mask, 0.0),
            tf.fill(tf.shape(attn_mask), -100.0),
            tf.zeros_like(attn_mask)
        )
        return attn_mask

    def call(self, x, H, W, training=False):
        """Apply the block.

        Args:
            x: tensor (B, H * W, C).
            H, W: feature map size.
            training: training flag.
        Returns:
            Tensor (B, H * W, C).
        """
        B = tf.shape(x)[0]
        C = self.dim
        shortcut = x

        x = tf.reshape(x, [B, H, W, C])

        # Pad to multiple of window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        x  = tf.pad(x, [[0, 0], [0, pad_b], [0, pad_r], [0, 0]])
        Hp = H + pad_b
        Wp = W + pad_r

        # Cyclic shift
        if self.shift_size > 0:
            x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])

        # Window partition
        x_windows = window_partition(x, self.window_size)  # (nW*B, ws, ws, C)
        N         = self.window_size * self.window_size
        x_windows = tf.reshape(x_windows, [-1, N, C])

        # Attention
        attn_mask = self._get_attn_mask(Hp, Wp)
        attn_out  = self.attn(x_windows, mask=attn_mask, training=training)

        # Merge windows
        attn_out = tf.reshape(attn_out, [-1, self.window_size, self.window_size, C])
        x        = window_reverse(attn_out, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(x, shift=[self.shift_size, self.shift_size], axis=[1, 2])

        # Remove padding and flatten back to sequence
        x = x[:, :H, :W, :]
        x = tf.reshape(x, [B, H * W, C])

        # Post-norm residual connections (mirrors PyTorch version)
        x = shortcut + self._dp(self.norm1(x), training)
        x = x        + self._dp(self.norm2(self.mlp(x, training=training)), training)
        return x


class RSTB(layers.Layer):
    """Residual Swin transformer block: SwinV2 layers followed by a 3x3 convolution and a residual connection."""

    def __init__(self, dim, depth, num_heads, window_size=8,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., pretrained_window_size=0, **kwargs):
        """Build the SwinV2 layers (alternating unshifted and shifted windows) and the convolution.

        Args:
            dim: channel width.
            depth: number of SwinV2 layers.
            num_heads: number of heads.
            window_size: window side.
            mlp_ratio, qkv_bias, drop, attn_drop: layer settings.
            drop_path: stochastic depth rate.
            pretrained_window_size: window size of the pretraining.
        """
        super().__init__(**kwargs)
        self.dim = dim

        if isinstance(drop_path, float):
            drop_path_rates = [drop_path] * depth
        else:
            drop_path_rates = list(drop_path)

        # Renamed from self.layers to avoid collision with Keras's .layers property
        self.swin_layers = [
            SwinTransformerV2Layer(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path_rates[i],
                pretrained_window_size=pretrained_window_size,
                name=f'swin_layer_{i}')
            for i in range(depth)
        ]
        self.norm = layers.LayerNormalization(epsilon=1e-5, name='norm')
        self.conv = layers.Conv2D(dim, 3, padding='same', name='conv')

    def call(self, x, H, W, training=False):
        """Apply the block.

        Args:
            x: tensor (B, H * W, C).
            H, W: feature map size.
            training: training flag.
        Returns:
            Tensor (B, H * W, C).
        """
        residual = x
        for layer in self.swin_layers:
            x = layer(x, H, W, training=training)
        x = self.norm(x)

        x = tf.reshape(x, [-1, H, W, self.dim])
        x = self.conv(x)
        x = tf.reshape(x, [-1, H * W, self.dim])
        return x + residual


class UpsampleX5(layers.Layer):
    """Upsampling by 5: 3x3 convolution to out_channels * 25 maps followed by depth_to_space."""

    def __init__(self, dim, out_channels=1, **kwargs):
        """Build the convolution.

        Args:
            dim: input channel width.
            out_channels: number of output channels.
        """
        super().__init__(**kwargs)
        self.out_channels = out_channels
        # Equivalent to nn.Conv2d(dim, out_channels*25, 3, 1, 1) in BCHW
        self.conv = layers.Conv2D(out_channels * 25, 3, padding='same', name='conv')

    def call(self, x):
        """Upsample a tensor (B, H, W, C) to (B, 5H, 5W, out_channels)."""
        x = self.conv(x)  # (B, H, W, out_channels*25)
        return tf.nn.depth_to_space(x, 5)  # (B, 5H, 5W, out_channels)


class Swin2SR(keras.Model):
    """Swin2SR for ERA5 to CERRA super-resolution by a factor 5.

    Shallow convolution, stack of RSTB blocks, convolution with a residual connection, then
    UpsampleX5. Inputs are padded (reflection) to a multiple of the window size.
    """

    def __init__(
        self,
        in_chans=1,
        out_chans=1,
        embed_dim=96,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=8,
        mlp_ratio=4.,
        upscale=5,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        pretrained_window_sizes=(0, 0, 0, 0, 0, 0),
        **kwargs
    ):
        """Build the layers.

        Args:
            in_chans, out_chans: number of input and output channels.
            embed_dim: channel width.
            depths: number of SwinV2 layers of each RSTB.
            num_heads: number of heads of each RSTB.
            window_size: attention window side.
            mlp_ratio: hidden width of the MLP relative to embed_dim.
            upscale: upscaling factor.
            drop_rate, attn_drop_rate: dropout rates.
            drop_path_rate: maximum stochastic depth rate.
            pretrained_window_sizes: window size of the pretraining for each RSTB.
        """
        super().__init__(**kwargs)
        self.window_size = window_size
        self.upscale     = upscale
        self.embed_dim   = embed_dim
        num_rstb         = len(depths)
        self.loss_tracker     = keras.metrics.Mean(name="train_loss")
        self.val_loss_tracker = keras.metrics.Mean(name="val_loss")

        # Shallow feature extraction - Conv2D is BHWC in TF
        self.conv_first = layers.Conv2D(embed_dim, 3, padding='same', name='conv_first')

        # Deep feature extraction (RSTB stack)
        total_depth = sum(depths)
        dpr = [float(x) for x in np.linspace(0, drop_path_rate, total_depth)]
        cur = 0
        self.rstb_layers = []
        for i in range(num_rstb):
            self.rstb_layers.append(
                RSTB(
                    dim=embed_dim, depth=depths[i], num_heads=num_heads[i],
                    window_size=window_size, mlp_ratio=mlp_ratio,
                    qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[cur: cur + depths[i]],
                    pretrained_window_size=pretrained_window_sizes[i],
                    name=f'rstb_{i}')
            )
            cur += depths[i]

        self.norm             = layers.LayerNormalization(epsilon=1e-5, name='norm')
        self.conv_after_body  = layers.Conv2D(embed_dim, 3, padding='same',
                                              name='conv_after_body')
        self.upsample         = UpsampleX5(embed_dim, out_channels=out_chans,
                                           name='upsample')

    @property
    def metrics(self):
        """Metrics reset by Keras at each epoch: the validation loss tracker."""
        # Keras resets these metrics at the end of each epoch
        return [self.val_loss_tracker]

    def _forward_bhwc(self, x, training=False):
        """Forward pass on a BHWC tensor.

        Args:
            x: tensor (B, H, W, in_chans).
            training: training flag.
        Returns:
            Tensor (B, upscale * H, upscale * W, out_chans).
        """
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]

        # Pad to multiple of window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x  = tf.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]], mode='REFLECT')
        Hp = H + pad_h
        Wp = W + pad_w

        # Shallow features (B, Hp, Wp, embed_dim)
        feat = self.conv_first(x)

        # Deep features via RSTB stack on flattened sequence (B, Hp*Wp, embed_dim)
        feat_seq = tf.reshape(feat, [-1, Hp * Wp, self.embed_dim])
        for rstb in self.rstb_layers:
            feat_seq = rstb(feat_seq, Hp, Wp, training=training)
        feat_seq = self.norm(feat_seq)

        # Back to spatial BHWC
        feat_deep = tf.reshape(feat_seq, [-1, Hp, Wp, self.embed_dim])
        feat_deep = self.conv_after_body(feat_deep)

        # Residual connection before upsampling
        feat_total = feat + feat_deep

        # Remove padding
        feat_total = feat_total[:, :H, :W, :]

        # Upsample x5
        return self.upsample(feat_total)

    def call(self, x, training=False):
        """Forward pass on a tensor (B, H, W, in_chans)."""
        return self._forward_bhwc(x, training=training)

    def summary_str(self):
        """Return a text summary with the number of total and trainable parameters."""
        total     = sum(tf.size(v).numpy() for v in self.variables)
        trainable = sum(tf.size(v).numpy() for v in self.trainable_variables)
        return (f"Swin2SR_TF  |  embed_dim={self.embed_dim}  "
                f"upscale=×{self.upscale}  "
                f"window={self.window_size}\n"
                f"  Total params    : {total:,}\n"
                f"  Trainable params: {trainable:,}")

    @tf.function(jit_compile=True)
    def train_step(self, data):
        """One training step with the MSE loss and gradient clipping at global norm 1.

        Args:
            data: batch (lr, hr).
        Returns:
            Dictionary with the running training loss.
        """
        lr, hr = data

        lr = tf.cast(lr, tf.float32)
        hr = tf.cast(hr, tf.float32)

        with tf.GradientTape() as tape:
            pred = self(lr, training=True)

            loss = tf.reduce_mean(tf.square(pred - hr))  # L2 loss

        grads = tape.gradient(loss, self.trainable_variables)

        # Clip the global gradient norm (not inside the optimizer)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update the loss tracker and return it for fit()
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @tf.function(jit_compile=True)
    def test_step(self, data):
        """One evaluation step; the validation loss is the mean absolute error.

        Args:
            data: batch (lr, hr).
        Returns:
            Dictionary with the running validation loss.
        """
        lr, hr = data
        lr = tf.cast(lr, tf.float32)
        hr = tf.cast(hr, tf.float32)

        pred = self(lr, training=False)
        loss = tf.reduce_mean(tf.abs(pred - hr))

        self.val_loss_tracker.update_state(loss)
        return {"loss": self.val_loss_tracker.result()}


def build_swin2sr(
    embed_dim=96,
    depths=(6, 6, 6, 6, 6, 6),
    num_heads=(6, 6, 6, 6, 6, 6),
    window_size=8,
    drop_path_rate=0.1,
) -> Swin2SR:
    """Build a single-channel Swin2SR for 64 x 80 to 320 x 400 downscaling.

    Args:
        embed_dim: channel width (96 light, 180 as in the paper).
        depths: number of SwinV2 layers of each RSTB.
        num_heads: number of heads of each RSTB.
        window_size: window side, must divide 64 and 80.
        drop_path_rate: stochastic depth rate.
    Returns:
        Swin2SR built with a dummy forward pass.
    """
    model = Swin2SR(
        in_chans=1, out_chans=1,
        embed_dim=embed_dim,
        depths=list(depths),
        num_heads=list(num_heads),
        window_size=window_size,
        mlp_ratio=4.,
        upscale=5,
        drop_path_rate=drop_path_rate,
        pretrained_window_sizes=tuple(0 for _ in depths),
    )
    # Build the model by running a dummy forward pass
    dummy = tf.zeros([1, 64, 80, 1], dtype=tf.float32)
    model(dummy, training=False)
    return model


def train_step(model, optimizer, lr_input, hr_input) -> float:
    """One training step with the mean absolute error and gradient clipping at norm 1.

    Args:
        model: Swin2SR model.
        optimizer: Keras optimizer.
        lr_input: LR batch (B, 64, 80, C) scaled to [0, 1].
        hr_input: HR batch (B, 320, 400, C) scaled to [0, 1].
    Returns:
        Loss value as a float.
    """
    lr = tf.cast(lr_input, tf.float32)
    hr = tf.cast(hr_input, tf.float32)

    with tf.GradientTape() as tape:
        pred = model(lr, training=True)
        loss = tf.reduce_mean(tf.abs(pred - hr))  # MAE loss

    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)  # gradient clipping
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return float(loss.numpy())


def infer(model, lr_input) -> np.ndarray:
    """Run the model on a batch.

    Args:
        model: Swin2SR model.
        lr_input: LR batch (B, 64, 80, C) scaled to [0, 1], numpy array or tensor.
    Returns:
        float32 numpy array (B, 320, 400, C) in the scaled range.
    """
    lr  = tf.cast(lr_input, tf.float32)
    out = model(lr, training=False)
    return out.numpy()


class Swin2SRGenerator:
    """Wrapper around the single-channel Swin2SR with predict and save."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (6, 6, 6, 6, 6, 6),
        num_heads: Tuple[int, ...] = (6, 6, 6, 6, 6, 6),
        window_size: int = 8,
    ):
        """Build the model and optionally load weights.

        Args:
            weights_path: weight file to load, or None.
            embed_dim, depths, num_heads, window_size: model hyperparameters.
        """
        self.model = build_swin2sr(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
        )
        if weights_path is not None:
            self.model.load_weights(weights_path)
            print(f"[Swin2SR_TF] Loaded weights from {weights_path}")
        print(self.model.summary_str())

    def predict(self, lr_input: np.ndarray) -> np.ndarray:
        """Predict HR fields from a LR batch (B, 64, 80, 1); returns (B, 320, 400, 1)."""
        return infer(self.model, lr_input)

    def save(self, path: str):
        """Save the weights to a file."""
        self.model.save_weights(path)
        print(f"[Swin2SR_TF] Weights saved to {path}")


def build_swin2sr_uv(
    in_chans=2,
    embed_dim=96,
    depths=(6, 6, 6, 6, 6, 6),
    num_heads=(6, 6, 6, 6, 6, 6),
    window_size=8,
    drop_path_rate=0.1,
) -> Swin2SR:
    """Build a Swin2SR for the u and v channels.

    Same as build_swin2sr with in_chans input channels and 2 output channels.

    Args:
        in_chans: number of input channels.
        embed_dim: channel width.
        depths: number of SwinV2 layers of each RSTB.
        num_heads: number of heads of each RSTB.
        window_size: window side.
        drop_path_rate: stochastic depth rate.
    Returns:
        Swin2SR mapping (B, 64, 80, in_chans) to (B, 320, 400, 2).
    """
    model = Swin2SR(
        in_chans=in_chans, out_chans=2,
        embed_dim=embed_dim,
        depths=list(depths),
        num_heads=list(num_heads),
        window_size=window_size,
        mlp_ratio=4.,
        upscale=5,
        drop_path_rate=drop_path_rate,
        pretrained_window_sizes=tuple(0 for _ in depths),
    )
    dummy = tf.zeros([1, 64, 80, in_chans], dtype=tf.float32)  # 2-channel dummy
    model(dummy, training=False)
    return model


class Swin2SRGenerator_UV:
    """Wrapper around the u and v Swin2SR with predict and save."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        embed_dim: int = 96,
        in_chans: int = 2,
        depths: Tuple[int, ...] = (6, 6, 6, 6, 6, 6),
        num_heads: Tuple[int, ...] = (6, 6, 6, 6, 6, 6),
        window_size: int = 8,
    ):
        """Build the model and optionally load weights.

        Args:
            weights_path: weight file to load, or None.
            embed_dim, in_chans, depths, num_heads, window_size: model hyperparameters.
        """
        self.model = build_swin2sr_uv(
            embed_dim=embed_dim,
            in_chans=in_chans,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
        )
        if weights_path is not None:
            self.model.load_weights(weights_path)
            print(f"[Swin2SR_UV_TF] Loaded weights from {weights_path}")
        print(self.model.summary_str())

    def predict(self, lr_input: np.ndarray) -> np.ndarray:
        """Predict HR fields from a LR batch (B, 64, 80, 2); returns (B, 320, 400, 2)."""
        return infer(self.model, lr_input)

    def save(self, path: str):
        """Save the weights to a file."""
        self.model.save_weights(path)
        print(f"[Swin2SR_UV_TF] Weights saved to {path}")


class Swin2SRSpectral(Swin2SR):
    """Swin2SR whose train_step and test_step use MSE plus an FFT log-amplitude loss.

    The layers and weights are identical to Swin2SR.
    """

    def __init__(
        self,
        *args,
        lambda_fft: float = 0.0,
        eps:        float = 1e-12,
        **kwargs,
    ):
        """Build the model and store the loss settings.

        Args:
            lambda_fft: weight of the spectral term (0 gives pure MSE).
            eps: constant added inside the logarithm.
        """
        super().__init__(*args, **kwargs)
        self.lambda_fft = float(lambda_fft)
        self.eps        = float(eps)

    def _fft_loss(self, pred: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        """Sum of squared differences of the log FFT amplitudes.

        Args:
            pred, target: tensors (B, H, W, C).
        Returns:
            Scalar tensor.
        """
        pred_c   = tf.cast(tf.transpose(pred,   [0, 3, 1, 2]), tf.float32)
        target_c = tf.cast(tf.transpose(target, [0, 3, 1, 2]), tf.float32)

        F_pred   = tf.signal.rfft2d(pred_c)  # (B, C, H, W//2+1) complex64
        F_target = tf.signal.rfft2d(target_c)

        log_amp_pred   = tf.math.log(tf.abs(F_pred)   + self.eps)
        log_amp_target = tf.math.log(tf.abs(F_target) + self.eps)

        return tf.reduce_sum(tf.square(log_amp_pred - log_amp_target))

    def _spectral_loss(self, pred: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        """MSE plus lambda_fft times the spectral term."""
        loss_mse = tf.reduce_mean(tf.square(pred - target))
        if self.lambda_fft > 0.0:
            return loss_mse + self.lambda_fft * self._fft_loss(pred, target)
        return loss_mse

    def train_step(self, data):
        """One training step with the spectral loss and gradient clipping at global norm 1.

        Args:
            data: batch (lr, hr).
        Returns:
            Dictionary with the running training loss.
        """
        lr, hr = data
        lr = tf.cast(lr, tf.float32)
        hr = tf.cast(hr, tf.float32)

        with tf.GradientTape() as tape:
            pred = self(lr, training=True)
            loss = self._spectral_loss(pred, hr)

        grads = tape.gradient(loss, self.trainable_variables)
        # Clip the global gradient norm (not inside the optimizer)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        """One evaluation step with the spectral loss.

        Args:
            data: batch (lr, hr).
        Returns:
            Dictionary with the running validation loss.
        """
        lr, hr = data
        lr = tf.cast(lr, tf.float32)
        hr = tf.cast(hr, tf.float32)

        pred = self(lr, training=False)
        loss = self._spectral_loss(pred, hr)

        self.val_loss_tracker.update_state(loss)
        return {"loss": self.val_loss_tracker.result()}

    def get_config(self):
        """Configuration including the loss settings."""
        config = super().get_config()
        config.update({"lambda_fft": self.lambda_fft, "eps": self.eps})
        return config


def build_swin2sr_uv_spectral(
    embed_dim:       int   = 96,
    depths:          tuple = (6, 6, 6, 6, 6, 6),
    num_heads:       tuple = (6, 6, 6, 6, 6, 6),
    window_size:     int   = 8,
    drop_path_rate:  float = 0.1,
    lambda_fft:      float = 0.0,
    eps:             float = 1e-12,
) -> Swin2SRSpectral:
    """Build a Swin2SRSpectral for the u and v channels.

    Args:
        embed_dim: channel width.
        depths: number of SwinV2 layers of each RSTB.
        num_heads: number of heads of each RSTB.
        window_size: window side.
        drop_path_rate: stochastic depth rate.
        lambda_fft: weight of the spectral term.
        eps: constant added inside the logarithm.
    Returns:
        Swin2SRSpectral mapping (B, 64, 80, 2) to (B, 320, 400, 2).
    """
    model = Swin2SRSpectral(
        in_chans  = 2,
        out_chans = 2,
        embed_dim = embed_dim,
        depths    = list(depths),
        num_heads = list(num_heads),
        window_size           = window_size,
        mlp_ratio             = 4.,
        upscale               = 5,
        drop_path_rate        = drop_path_rate,
        pretrained_window_sizes = tuple(0 for _ in depths),
        lambda_fft = lambda_fft,
        eps        = eps,
    )
    dummy = tf.zeros([1, 64, 80, 2], dtype=tf.float32)
    model(dummy, training=False)
    return model


class Swin2SRGenerator_UV_Spectral:
    """Wrapper around the u and v Swin2SRSpectral model with predict and save."""

    def __init__(
        self,
        weights_path: Optional[str]  = None,
        embed_dim:    int             = 96,
        depths:       Tuple[int, ...] = (6, 6, 6, 6, 6, 6),
        num_heads:    Tuple[int, ...] = (6, 6, 6, 6, 6, 6),
        window_size:  int             = 8,
        lambda_fft:   float           = 0.0,
        eps:          float           = 1e-12,
    ):
        """Build the model and optionally load weights.

        Args:
            weights_path: weight file to load, or None.
            embed_dim, depths, num_heads, window_size: model hyperparameters.
            lambda_fft: weight of the spectral term.
            eps: constant added inside the logarithm.
        """
        self.model = build_swin2sr_uv_spectral(
            embed_dim      = embed_dim,
            depths         = depths,
            num_heads      = num_heads,
            window_size    = window_size,
            lambda_fft     = lambda_fft,
            eps            = eps,
        )
        if weights_path is not None:
            self.model.load_weights(weights_path)
            print(f"[Swin2SR_UV_Spectral_TF] Loaded weights from {weights_path}")
        print(self.model.summary_str())

    def predict(self, lr_input: np.ndarray) -> np.ndarray:
        """Predict HR fields from a LR batch (B, 64, 80, 2); returns (B, 320, 400, 2)."""
        return infer(self.model, lr_input)

    def save(self, path: str):
        """Save the weights to a file."""
        self.model.save_weights(path)
        print(f"[Swin2SR_UV_Spectral_TF] Weights saved to {path}")


if __name__ == "__main__":
    print("\nBuilding Swin2SR_UV_TF (×5, two-channel)…")
    model_uv = build_swin2sr_uv()
    print(model_uv.summary_str())

    dummy_lr_uv = np.random.rand(2, 64, 80, 2).astype(np.float32)
    print(f"\nInput  shape : {dummy_lr_uv.shape}")
    out_uv = infer(model_uv, dummy_lr_uv)
    print(f"Output shape : {out_uv.shape}")
    assert out_uv.shape == (2, 320, 400, 2), f"Unexpected shape: {out_uv.shape}"
    print("✓  UV shape check passed.")
