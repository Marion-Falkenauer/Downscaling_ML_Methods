"""Temporal sequence encoding of the ERA5 inputs and matching slicing of the CERRA targets.
Author: Marion Falkenauer (with the help of Claude)"""
import numpy as np


def sequence_encoding(arrays_LR, N_BEFORE, N_AFTER, keep_channels=True):
    """Stack neighbouring time steps of the low-resolution series.

    Args:
        arrays_LR: array (T, H, W, C) or (T, H, W).
        N_BEFORE: number of frames before the centre frame.
        N_AFTER: number of frames after the centre frame.
        keep_channels: kept for compatibility, not used.
    Returns:
        Array (T - N_BEFORE - N_AFTER, H, W, seq_len, C), or (T', H, W, seq_len)
        for 3D input, with seq_len = N_BEFORE + 1 + N_AFTER. When
        N_BEFORE == N_AFTER == 0 the sequence axis is removed.
    """
    arrays_LR = np.asarray(arrays_LR)
    if arrays_LR.ndim not in (3, 4):
        raise ValueError("arrays must have shape (T,H,W,C) or (T,H,W)")

    T = arrays_LR.shape[0]
    seq_len = N_BEFORE + 1 + N_AFTER

    offsets = np.arange(-N_BEFORE, N_AFTER + 1)
    print("Offsets:")
    print(offsets)
    if T < seq_len:
        raise ValueError("Not enough time steps to build a sequence with the requested N_BEFORE/N_AFTER")
    # Centre frames with a complete neighbourhood
    centers = np.arange(N_BEFORE, T - N_AFTER)
    idxs = centers[:, None] + offsets[None, :]
    if arrays_LR.ndim == 4:
        seq = arrays_LR[idxs]
        seq = np.transpose(seq, (0, 2, 3, 1, 4))
    else:
        seq = arrays_LR[idxs]
        seq = np.transpose(seq, (0, 2, 3, 1))
    if N_AFTER == N_BEFORE == 0:
        seq = np.squeeze(seq, axis=3)
    return seq


def HR_sequence_slicing(array_HR, n_before=0, n_after=0):
    """Drop the high-resolution frames that have no complete low-resolution sequence.

    Args:
        array_HR: array (T, ...) of high-resolution frames.
        n_before: number of frames before the centre frame.
        n_after: number of frames after the centre frame.
    Returns:
        Array (T - n_before - n_after, ...) aligned in time with the output of sequence_encoding.
    """
    if n_after > 0:
        array_HR = array_HR[n_before:-n_after, ...]
    elif n_after == 0:
        array_HR = array_HR[n_before:, ...]
    return array_HR
