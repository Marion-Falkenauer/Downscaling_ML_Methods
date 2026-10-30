"""U-Net building blocks for the conditional diffusion denoiser and the supervised post-processing network.
Author: Marion Falkenauer (with the help of Claude)
"""
import math

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.saving import register_keras_serializable

from setup import embedding_dims, embedding_max_frequency


@register_keras_serializable()
def sinusoidal_embedding(x):
    """Sinusoidal embedding of the noise variance.

    Args:
        x: tensor (B, 1, 1, 1) of noise variances.
    Returns:
        Tensor (B, 1, 1, embedding_dims) of sine and cosine features at
        log-spaced frequencies between 1 and embedding_max_frequency.
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
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def channel_attention(x, ratio=16):
    """CBAM channel attention.

    Args:
        x: feature tensor (B, H, W, C).
        ratio: channel reduction ratio of the shared MLP.
    Returns:
        Tensor (B, H, W, C), x rescaled per channel.
    """
    channel_axis = -1
    channel = x.shape[channel_axis]
    shared_layer_one = layers.Dense(
        channel // ratio, activation='relu', kernel_initializer='he_normal',
        use_bias=True, bias_initializer='zeros')
    shared_layer_two = layers.Dense(
        channel, activation='linear', kernel_initializer='he_normal',
        use_bias=True, bias_initializer='zeros')

    avg_pool = layers.GlobalAveragePooling2D()(x)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(x)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)
    return layers.Multiply()([x, cbam_feature])


def spatial_attention(x):
    """CBAM spatial attention.

    Args:
        x: feature tensor (B, H, W, C).
    Returns:
        Tensor (B, H, W, C), x rescaled per pixel.
    """
    avg_out = layers.Lambda(lambda x: keras.backend.mean(x, axis=3, keepdims=True))(x)
    max_out = layers.Lambda(lambda x: keras.backend.max(x, axis=3, keepdims=True))(x)
    concat = layers.Concatenate(axis=3)([avg_out, max_out])
    spatial_attention_feature = layers.Conv2D(
        1, kernel_size=7, padding="same", activation="sigmoid")(concat)
    return layers.Multiply()([x, spatial_attention_feature])


def ResidualBlockWithCBAM(width):
    """Residual block with CBAM channel and spatial attention.

    Args:
        width: number of output channels.
    Returns:
        Function applying the block to a tensor (B, H, W, C).
    """
    def apply(x):
        """Apply the residual block to a tensor (B, H, W, C)."""
        input_width = x.shape[3]
        residual = x if input_width == width else layers.Conv2D(width, kernel_size=1)(x)
        x = layers.LayerNormalization(axis=-1, center=True, scale=True)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same",
                          activation=keras.activations.swish)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = channel_attention(x)
        x = spatial_attention(x)
        x = layers.Add()([x, residual])
        return x
    return apply


def ResidualBlock(width):
    """Residual block: LayerNorm, Conv (swish), Conv, skip connection.

    Args:
        width: number of output channels.
    Returns:
        Function applying the block to a tensor (B, H, W, C).
    """
    def apply(x):
        """Apply the residual block to a tensor (B, H, W, C)."""
        input_width = x.shape[3]
        residual = x if input_width == width else layers.Conv2D(width, kernel_size=1)(x)
        x = layers.LayerNormalization(axis=-1, center=True, scale=True)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same",
                          activation=keras.activations.swish)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x
    return apply


def attention_block(x, g, width):
    """Additive gated spatial attention.

    Args:
        x: feature tensor (B, H, W, C).
        g: gating tensor (B, H, W, C').
        width: number of channels of the projections.
    Returns:
        Tensor (B, H, W, C), x multiplied by a single-channel attention map.
    """
    theta_x = layers.Conv2D(width, [1, 1], strides=[1, 1])(x)
    phi_g = layers.Conv2D(width, [1, 1], strides=[1, 1])(g)
    f = keras.activations.relu(theta_x + phi_g)
    f1 = keras.layers.Conv2D(1, [1, 1], strides=[1, 1])(f)
    f1 = keras.activations.sigmoid(f1)
    return keras.layers.Multiply()([x, f1])


def ResidualBlockWithAttention(width):
    """Residual block with gated spatial attention, used in the bottleneck.

    Args:
        width: number of output channels.
    Returns:
        Function applying the block to a tensor (B, H, W, C).
    """
    def apply(x):
        """Apply the residual block to a tensor (B, H, W, C)."""
        input_width = x.shape[3]
        residual = x if input_width == width else layers.Conv2D(width, kernel_size=1)(x)
        x = layers.LayerNormalization(axis=-1, center=True, scale=True)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same",
                          activation=keras.activations.swish)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = attention_block(x, residual, width)
        x = layers.Add()([x, residual])
        return x
    return apply


def DownBlock(width, block_depth):
    """Encoder block: residual blocks (skips stored) then 2x average pooling.

    Args:
        width: number of channels of the residual blocks.
        block_depth: number of residual blocks.
    Returns:
        Function taking [x, skips] and returning the pooled tensor.
    """
    def apply(x):
        """Apply the residual blocks and pooling to [x, skips]; the block outputs are appended to skips."""
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x
    return apply


def UpBlock(width, block_depth):
    """Decoder block: 2x bilinear upsampling then residual blocks on concatenated skips.

    Args:
        width: number of channels of the residual blocks.
        block_depth: number of residual blocks.
    Returns:
        Function taking [x, skips] and returning the upsampled tensor.
    """
    def apply(x):
        """Upsample [x, skips] and apply the residual blocks on the concatenated skips."""
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x
    return apply


def get_network(image_size_height, input_frames, output_frames,
                widths, block_depth, image_size_width):
    """Conditional denoising U-Net for the diffusion models.

    Args:
        image_size_height: HR latitude pixels.
        input_frames: number of LR conditioning channels.
        output_frames: number of HR target channels.
        widths: channel width at each encoder level.
        block_depth: residual blocks per level.
        image_size_width: HR longitude pixels.
    Returns:
        keras.Model taking [noisy_images (B, H, W, input_frames + output_frames),
        noise_variances (B, 1, 1, 1)] and returning (B, H, W, output_frames).
    """
    noisy_images = keras.Input(
        shape=(image_size_height, image_size_width, input_frames + output_frames))
    noise_variances = keras.Input(shape=(1, 1, 1))

    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.UpSampling2D(
        size=(image_size_height, image_size_width), interpolation="nearest")(e)

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlockWithAttention(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = layers.Conv2D(output_frames, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")


def get_post_network(image_size_height, input_frames, output_frames,
                     widths, block_depth, image_size_width):
    """Plain U-Net for supervised downscaling, without noise conditioning.

    Args:
        image_size_height: HR latitude pixels.
        input_frames: number of input channels.
        output_frames: number of output channels.
        widths: channel width at each encoder level.
        block_depth: residual blocks per level.
        image_size_width: HR longitude pixels.
    Returns:
        keras.Model mapping (B, H, W, input_frames) to (B, H, W, output_frames).
    """
    noisy_images = keras.Input(
        shape=(image_size_height, image_size_width, input_frames))

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = layers.Conv2D(output_frames, kernel_size=1,
                      kernel_initializer="zeros", dtype='float32')(x)

    return keras.Model([noisy_images], x, name="residual_unet")
