"""Hyperparameters shared by the U-Net, Swin2SR denoiser and diffusion scripts.

widths, block_depth: U-Net channel widths per level and residual blocks per level.
embedding_dims, embedding_max_frequency: sinusoidal noise-variance embedding.
min_signal_rate, max_signal_rate: bounds of the diffusion signal-rate schedule.
Author: Marion Falkenauer (with the help of Claude)
"""

widths = [64, 128, 256, 384]
block_depth = 3

embedding_dims = 64
embedding_max_frequency = 1000.0

min_signal_rate = 0.3
max_signal_rate = 0.95
