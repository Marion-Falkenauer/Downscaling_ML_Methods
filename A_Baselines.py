#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A_Baselines.py
==============
Baseline evaluation pipeline for ERA5 → CERRA downscaling.

Two baselines are implemented, each saved in its own subfolder under
Results_ML/BASELINES/:

  bicubic/   — Bicubic interpolation via tf.image.resize(..., method='bicubic').
               Pure post-processing; no model weights or training.

  srcnn/     — Shallow CNN super-resolution (SRCNN) trained on the 2year
               training split.  Operates on bilinearly pre-upsampled LR input
               (same pipeline as A_UNet_V2.py).

Code structure
--------------
This script is a direct adaptation of A_UNet_V2.py.  All helper functions
(build_baseline_pipeline_from_prepared, plot_temporal_metrics, FFT helpers,
plot_comparison_wind_speed, plot_quiver_wind, plot_fft_analysis) are copied
verbatim from A_UNet_V2.py so that outputs are byte-for-byte comparable.

Binary training flag
--------------------
  TRAIN_SRCNN = True   → trains SRCNN from scratch and saves weights
  TRAIN_SRCNN = False  → loads weights from the existing checkpoint and
                         runs evaluation only (no backprop)

CSV outputs (per baseline, per case)
--------------------------------------
  test_metrics.csv          — scaled metrics (same schema as A_UNet_V2.py)
  test_metrics_unscaled.csv — physical-unit metrics (m/s)
"""

# Imports
import xarray as xr
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_XLA_FLAGS']          = '--tf_xla_auto_jit=2'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import csv
import gc
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
import pandas as pd

from scipy.fft    import fft2 as scipy_fft2, fftshift, fftfreq as scipy_fftfreq
from scipy.signal import windows as scipy_windows

from Pre_Processing_Functions import *
from ML_Methods_ERA5_CERRA import plot_comparison_transformer
from setup import *
from Tensor_Adaptation import *


# GPU config
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"Physical GPUs: {len(gpus)} — using {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU.")

# Plot settings  (identical to A_UNet_V2.py)
plt.rcParams.update({
    "font.size":       25,
    "axes.labelsize":  20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 20,
    "axes.titlesize":  20,
})

# Configuration
BASE_DIR = "/nfshomesnobackup/students/2025/falkenauer/Documents/RP_Codes"

PX_LAT_HR = 320
PX_LON_HR = 400
PX_LAT_LR = 64
PX_LON_LR = 80
RF        = 5
UV_COMP   = True

N_BEFORE = 0
N_AFTER  = 0
INPUT_FRAMES  = (N_BEFORE + 1 + N_AFTER) * (2 if UV_COMP else 1)   # 2
OUTPUT_FRAMES = 2 if UV_COMP else 1                                  # 2

# Multi-case runner
# Bicubic is evaluated on the shared test set (case-agnostic).
# SRCNN is trained on 2year and evaluated on the same test set.
# Additional cases listed here will be evaluated for both baselines.
CASES = ["2year"]

TRAIN_DURATION = {
    "1year":   2920,
    "2year":   5840,
    "4year":   8768,
    "5year":  14608,
    "10year": 29216,
    "20year": 58440,
}

OUTPUT_ROOT = BASE_DIR
SHARED_DIR  = os.path.join(OUTPUT_ROOT, "Data_Shared")

# SRCNN training flag
# True  → train SRCNN from scratch (weights saved to RESULTS_DIR)
# False → load existing checkpoint from RESULTS_DIR and evaluate only
TRAIN_SRCNN = True

# SRCNN training hyperparameters
SRCNN_EPOCHS               = 100
SRCNN_BATCH_SIZE           = 32
SRCNN_LEARNING_RATE        = 1e-5
SRCNN_EARLY_STOPPING_PAT   = 10
SRCNN_TRAIN_CASE           = "2year"   # training split to use

# FFT analysis
FFT_CUTOFF = 0.05
HR_H       = PX_LAT_HR
HR_W       = PX_LON_HR


# Timer  (identical to A_UNet_V2.py)
class SectionTimer:
    def __init__(self):
        self._starts = {}
        self.times   = {}

    def start(self, name):
        self._starts[name] = time.perf_counter()

    def stop(self, name):
        self.times[name] = time.perf_counter() - self._starts[name]


# tf.data pipeline
# Adapted from build_unet_pipeline_from_prepared in A_UNet_V2.py.
# Added lr_resize_method parameter so bicubic and bilinear baselines
# share the same code path.
def build_baseline_pipeline_from_prepared(
        case_dir,
        shared_dir,
        csv_path,
        batch_size      = 8,
        shuffle_buffer  = 3000,
        lr_resize_method = "bilinear",  # "bilinear" for SRCNN, "bicubic" for bicubic baseline
):
    # Build tf.data train / val / test datasets from pre-split memmaps.
    # lr_resize_method : str
    # Resize algorithm used to upsample LR → HR spatially.
    # "bilinear" (SRCNN) or "bicubic" (bicubic baseline).
    def _sidecar(npy_path):
        lines = open(npy_path.replace(".npy", "_shape.txt")).read().strip().splitlines()
        shape = tuple(int(s) for s in lines[0].split(","))
        dtype = np.dtype(lines[1]) if len(lines) > 1 else np.dtype("float32")
        return shape, dtype

    splits = {
        "train": (os.path.join(case_dir,   "train", "ERA5_train.npy"),
                  os.path.join(case_dir,   "train", "CERRA_train.npy")),
        "val":   (os.path.join(shared_dir, "val",   "ERA5_val.npy"),
                  os.path.join(shared_dir, "val",   "CERRA_val.npy")),
        "test":  (os.path.join(shared_dir, "test",  "ERA5_test.npy"),
                  os.path.join(shared_dir, "test",  "CERRA_test.npy")),
    }
    shapes = {}
    for split, (lr_path, hr_path) in splits.items():
        lr_sh, lr_dtype = _sidecar(lr_path)
        hr_sh, hr_dtype = _sidecar(hr_path)
        shapes[split] = {
            "lr": lr_sh, "hr": hr_sh,
            "lr_path": lr_path, "hr_path": hr_path,
            "lr_dtype": lr_dtype, "hr_dtype": hr_dtype,
        }

    print("\n  Tensor shape verification:")
    for split_name, info in shapes.items():
        lr_sh, hr_sh = info["lr"], info["hr"]
        if lr_sh[-1] != INPUT_FRAMES:
            raise ValueError(
                f"[{split_name}] LR channel mismatch: {lr_sh[-1]} != {INPUT_FRAMES}")
        if hr_sh[-1] != OUTPUT_FRAMES:
            raise ValueError(
                f"[{split_name}] HR channel mismatch: {hr_sh[-1]} != {OUTPUT_FRAMES}")
        if lr_sh[0] != hr_sh[0]:
            raise ValueError(
                f"[{split_name}] Timestep mismatch: LR={lr_sh[0]}, HR={hr_sh[0]}")
        print(f"    [{split_name}] OK — {lr_sh[0]} samples  "
              f"LR {lr_sh[1:]}  HR {hr_sh[1:]}")

    # Min/max from statistics CSV
    stats_df = pd.read_csv(csv_path, index_col=0)

    def _get_minmax(row_label, n_channels):
        comp_map = {0: ("min_u", "max_u"), 1: ("min_v", "max_v")}
        mins, maxs = [], []
        for c in range(n_channels):
            min_col, max_col = comp_map[c % 2]
            mins.append(tf.constant(float(stats_df.loc[row_label, min_col]),
                                    dtype=tf.float32))
            maxs.append(tf.constant(float(stats_df.loc[row_label, max_col]),
                                    dtype=tf.float32))
        return mins, maxs

    lr_mins, lr_maxs = _get_minmax("ERA5",  INPUT_FRAMES)
    hr_mins, hr_maxs = _get_minmax("CERRA", OUTPUT_FRAMES)

    def normalize_lr(x):
        return tf.stack(
            [(x[..., i] - lr_mins[i]) / (lr_maxs[i] - lr_mins[i] + 1e-8)
             for i in range(INPUT_FRAMES)], axis=-1)

    def normalize_hr(x):
        return tf.stack(
            [(x[..., i] - hr_mins[i]) / (hr_maxs[i] - hr_mins[i] + 1e-8)
             for i in range(OUTPUT_FRAMES)], axis=-1)

    def upsample_lr(x):
        return tf.image.resize(x, (PX_LAT_HR, PX_LON_HR),
                               method=lr_resize_method)

    def _make_split_ds(split, shuffle, repeat):
        info         = shapes[split]
        lr_sh, hr_sh = info["lr"], info["hr"]
        lr_mm = np.memmap(info["lr_path"], dtype=info["lr_dtype"],
                          mode="r", shape=lr_sh)
        hr_mm = np.memmap(info["hr_path"], dtype=info["hr_dtype"],
                          mode="r", shape=hr_sh)
        N_s = lr_sh[0]

        def _lr_gen():
            for i in range(N_s):
                yield lr_mm[i].astype(np.float32)

        def _hr_gen():
            for i in range(N_s):
                yield hr_mm[i].astype(np.float32)

        tf_lr = (
            tf.data.Dataset.from_generator(
                _lr_gen,
                output_signature=tf.TensorSpec(
                    shape=(lr_sh[1], lr_sh[2], lr_sh[3]), dtype=tf.float32))
            .map(normalize_lr, num_parallel_calls=tf.data.AUTOTUNE)
            .map(upsample_lr,  num_parallel_calls=tf.data.AUTOTUNE))
        tf_hr = (
            tf.data.Dataset.from_generator(
                _hr_gen,
                output_signature=tf.TensorSpec(
                    shape=(hr_sh[1], hr_sh[2], hr_sh[3]), dtype=tf.float32))
            .map(normalize_hr, num_parallel_calls=tf.data.AUTOTUNE))

        ds = tf.data.Dataset.zip((tf_lr, tf_hr))
        if shuffle:
            ds = ds.shuffle(min(shuffle_buffer, N_s),
                            reshuffle_each_iteration=True, seed=42)
        ds = ds.batch(batch_size, drop_remainder=True)
        if repeat:
            ds = ds.repeat()
        return ds.prefetch(tf.data.AUTOTUNE), N_s

    train_ds, n_train = _make_split_ds("train", shuffle=True,  repeat=True)
    val_ds,   n_val   = _make_split_ds("val",   shuffle=False, repeat=True)
    test_ds,  n_test  = _make_split_ds("test",  shuffle=False, repeat=False)

    return (
        {"train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds,
         "n_train": n_train, "n_val": n_val, "n_test": n_test},
        {"lr": (lr_mins, lr_maxs), "hr": (hr_mins, hr_maxs)},
    )


# SRCNN model
# Adapted from build_SRCNN in ML_Methods_ERA5_CERRA.py.
# The UpSampling2D layer is removed because the pipeline already
# upsamples LR → HR before the model sees the data.
# Architecture:  Conv(64,9×9,ReLU) → Conv(32,3×3,ReLU) → Conv(2,5×5,linear)
def build_srcnn_model(input_h=PX_LAT_HR, input_w=PX_LON_HR, n_ch=OUTPUT_FRAMES):
    """
    SRCNN operating on bilinearly pre-upsampled HR-resolution input.

    Input  : (batch, 320, 400, 2) — normalised [0, 1]
    Output : (batch, 320, 400, 2) — normalised [0, 1]
    """
    model = keras.Sequential([
        layers.Input(shape=(input_h, input_w, n_ch)),
        layers.Conv2D(64, (9, 9), activation='relu', padding='same',
                      name='srcnn_conv1'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      name='srcnn_conv2'),
        layers.Conv2D(n_ch, (5, 5), activation='linear', padding='same',
                      name='srcnn_output'),
    ], name='srcnn')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=SRCNN_LEARNING_RATE),
        loss='mse',
        metrics=['mse'],
        jit_compile=True,
    )
    model.summary()
    return model


# Temporal metrics  (identical to A_UNet_V2.py)
_YEAR_START      = "2024-01-01"
_YEAR_END        = "2024-12-31 21:00"
_FREQ_HOURS      = 3
_FULL_YEAR_TIMES = None


def _build_year_times():
    global _FULL_YEAR_TIMES
    if _FULL_YEAR_TIMES is None:
        _FULL_YEAR_TIMES = pd.date_range(
            _YEAR_START, _YEAR_END, freq=f"{_FREQ_HOURS}h")
    return _FULL_YEAR_TIMES


def plot_temporal_metrics(
        mse_per_sample, psnr_per_sample, ssim_per_sample,
        n_train, n_val, results_dir,
        uv_comp=True, rolling_window=8, data_is_ordered=True,
):
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker

    n_test     = len(mse_per_sample)
    all_times  = _build_year_times()
    offset     = n_train + n_val
    test_times = (all_times[0:n_test] if n_test <= len(all_times)
                  else pd.date_range(_YEAR_START, periods=n_test,
                                     freq=f"{_FREQ_HOURS}h"))

    def _rolling(arr):
        return (pd.Series(arr)
                .rolling(rolling_window, center=True, min_periods=1)
                .mean().values)

    mse_smooth  = _rolling(mse_per_sample)
    psnr_smooth = _rolling(psnr_per_sample)
    ssim_smooth = _rolling(ssim_per_sample)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={"hspace": 0.08})
    ch_label  = "U and V averaged channels" if uv_comp else "(u only)"
    raw_kw    = dict(color="#aec6e8", alpha=0.55, linewidth=0.6, zorder=1)
    smooth_kw = dict(color="#1f6fab", linewidth=1.8, zorder=3)
    mean_kw   = dict(color="black",  linewidth=1.2, linestyle="--", zorder=4)

    panels = [
        (mse_per_sample,  mse_smooth,
         f"Metrics over scaled quantities for {ch_label}", "MSE"),
        (psnr_per_sample, psnr_smooth, " ", "PSNR [dB]"),
        (ssim_per_sample, ssim_smooth, " ", "SSIM"),
    ]
    for ax, (raw, smooth, title, ylabel) in zip(axes, panels):
        mean_val = float(np.mean(raw))
        ax.plot(test_times, raw,    label="per-sample", **raw_kw)
        ax.plot(test_times, smooth,
                label=f"Rolling mean ({rolling_window} samples)", **smooth_kw)
        ax.axhline(mean_val, label=f"mean = {mean_val:.4f}", **mean_kw)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=13, pad=3)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.7)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        if not data_is_ordered:
            ax.text(0.01, 0.97,
                    "⚠  Temporal ordering not guaranteed (dataset was shuffled)",
                    transform=ax.transAxes, fontsize=9, color="#b5400a",
                    va="top", ha="left", style="italic")

    axes[-1].set_xlabel("Time (Year 2024 — 3-hourly samples)", fontsize=13)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
    fig.autofmt_xdate(rotation=35, ha="right")
    fig.suptitle(f"Per-sample test metrics over time  [n_test={n_test}]",
                 fontsize=14, y=1.01)
    plt.tight_layout()

    out_png = os.path.join(results_dir, "temporal_metrics.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved temporal metrics plot → {out_png}")

    pd.DataFrame({
        "datetime":     test_times,
        "sample_index": np.arange(offset, offset + n_test),
        "mse_scaled":   mse_per_sample,
        "psnr_db":      psnr_per_sample,
        "ssim":         ssim_per_sample,
        "mse_rolling":  mse_smooth,
        "psnr_rolling": psnr_smooth,
        "ssim_rolling": ssim_smooth,
    }).to_csv(os.path.join(results_dir, "temporal_metrics.csv"), index=False)
    print(f"Saved temporal metrics CSV → "
          f"{os.path.join(results_dir, 'temporal_metrics.csv')}")


# FFT spectral helpers  (identical to A_UNet_V2.py)
def _to_numpy(arr) -> np.ndarray:
    if isinstance(arr, tf.Tensor):
        return arr.numpy().astype(np.float32)
    return np.asarray(arr, dtype=np.float32)


def _uv_to_wspd_wdir(u: np.ndarray, v: np.ndarray):
    u64  = u.astype(np.float64)
    v64  = v.astype(np.float64)
    ws   = np.sqrt(u64 ** 2 + v64 ** 2)
    wdir = (270.0 - np.degrees(np.arctan2(v64, u64))) % 360.0
    return ws.astype(np.float32), wdir.astype(np.float32)


def _build_analysis_window(H: int, W: int) -> np.ndarray:
    w_h   = scipy_windows.hann(H, sym=True).astype(np.float32)
    w_w   = scipy_windows.hann(W, sym=True).astype(np.float32)
    win2d = np.outer(w_h, w_w)
    return (win2d / win2d.mean()).astype(np.float32)


def _compute_radial_grid(H: int, W: int):
    freq_h = fftshift(scipy_fftfreq(H)).astype(np.float32)
    freq_w = fftshift(scipy_fftfreq(W)).astype(np.float32)
    FH, FW = np.meshgrid(freq_h, freq_w, indexing="ij")
    return freq_h, freq_w, np.sqrt(FH ** 2 + FW ** 2).astype(np.float32)


def _windowed_power(field_2d: np.ndarray, window2d: np.ndarray) -> np.ndarray:
    xw = ((field_2d.astype(np.float64) - field_2d.mean())
          * window2d.astype(np.float64))
    return np.abs(fftshift(scipy_fft2(xw, norm="ortho"))) ** 2


def _radial_psd(field_2d: np.ndarray, window2d: np.ndarray,
                n_bins: int = 128, cutoff: float = None):
    H, W         = field_2d.shape
    _, _, radial = _compute_radial_grid(H, W)
    power        = _windowed_power(field_2d, window2d)
    if cutoff is not None:
        power = power * (radial > cutoff)
    r_edges = np.linspace(0.0, float(radial.max()), n_bins + 1)
    k_mid   = (0.5 * (r_edges[:-1] + r_edges[1:])).astype(np.float32)
    psd     = np.array([
        power[(radial >= r_edges[i]) & (radial < r_edges[i + 1])].mean()
        if ((radial >= r_edges[i]) & (radial < r_edges[i + 1])).any() else 0.0
        for i in range(n_bins)
    ], dtype=np.float32)
    return k_mid, psd


def plot_fft_analysis(results_dir, hr_pred, hr_true, lr_input,
                      cutoff=FFT_CUTOFF, n_bins=128, n_samples_avg=2920,
                      case=""):
    hr_pred  = _to_numpy(hr_pred)
    hr_true  = _to_numpy(hr_true)
    lr_input = _to_numpy(lr_input)
    HR_H_loc, HR_W_loc = hr_true.shape[1], hr_true.shape[2]

    if lr_input.shape[1] != HR_H_loc or lr_input.shape[2] != HR_W_loc:
        lr_input = tf.image.resize(
            lr_input, (HR_H_loc, HR_W_loc), method="bilinear"
        ).numpy().astype(np.float32)

    print(f"\n[plot_fft_analysis] case={case!r}  "
          f"n_use={min(n_samples_avg, hr_pred.shape[0])}")

    win2d      = _build_analysis_window(HR_H_loc, HR_W_loc)
    comp_names = ["U (100 m)", "V (100 m)"]
    freq_h_grid, freq_w_grid, _ = _compute_radial_grid(HR_H_loc, HR_W_loc)
    n_use = min(n_samples_avg, hr_pred.shape[0])

    single, avg = {}, {}
    for ch in range(2):
        k, gt_s    = _radial_psd(hr_true[0,  :, :, ch], win2d, n_bins=n_bins)
        _,  pred_s = _radial_psd(hr_pred[0,  :, :, ch], win2d, n_bins=n_bins)
        _,  lr_s   = _radial_psd(lr_input[0, :, :, ch], win2d, n_bins=n_bins)
        single[ch] = (k, gt_s, pred_s, lr_s)

        gt_stack   = np.stack([_radial_psd(hr_true[i,  :, :, ch], win2d, n_bins)[1]
                                for i in range(n_use)])
        pred_stack = np.stack([_radial_psd(hr_pred[i,  :, :, ch], win2d, n_bins)[1]
                                for i in range(n_use)])
        lr_stack   = np.stack([_radial_psd(lr_input[i, :, :, ch], win2d, n_bins)[1]
                                for i in range(n_use)])
        avg[ch] = (k, gt_stack.mean(0), pred_stack.mean(0), lr_stack.mean(0))

    # Save CSVs
    pd.DataFrame({
        "k_mid_cy_per_px": single[0][0],
        "gt_psd_u":   single[0][1], "pred_psd_u": single[0][2],
        "lr_psd_u":   single[0][3],
        "gt_psd_v":   single[1][1], "pred_psd_v": single[1][2],
        "lr_psd_v":   single[1][3],
    }).to_csv(os.path.join(results_dir, "sample_fft_spectra.csv"),
              index=False, float_format="%.8e")

    pd.DataFrame({
        "k_mid_cy_per_px": avg[0][0],
        "gt_psd_u":   avg[0][1], "pred_psd_u": avg[0][2],
        "lr_psd_u":   avg[0][3],
        "gt_psd_v":   avg[1][1], "pred_psd_v": avg[1][2],
        "lr_psd_v":   avg[1][3],
    }).to_csv(os.path.join(results_dir, "average_fft_spectra.csv"),
              index=False, float_format="%.8e")

    def _binned_figure(use_avg, save_name):
        fig, axes = (plt.subplots(2, 1, figsize=(9, 13)) if use_avg
                     else plt.subplots(2, 2, figsize=(18, 13)))
        tag_str = "Averaged" if use_avg else "Single-sample"
        fig.suptitle(f"{tag_str} — Binned Isotropic PSD — {case}\n"
                     f"HP cutoff = {cutoff:.3f} cy/px")
        for ch, comp in enumerate(comp_names):
            ax_psd = axes[ch] if use_avg else axes[ch, 0]
            k_ax, gt_psd, pred_psd, lr_psd = (avg if use_avg else single)[ch]
            ax_psd.semilogy(k_ax, gt_psd,   "k-",  lw=2.5,
                            label="CERRA (GT)", zorder=5)
            ax_psd.semilogy(k_ax, pred_psd, "b-",  lw=1.8, label="Prediction")
            ax_psd.semilogy(k_ax, lr_psd,   "g-",  lw=1.5, label="ERA5 bilinear")
            ax_psd.axvline(cutoff, color="red", ls="--", lw=1.5,
                           label=f"HP cutoff ({cutoff:.3f})")
            ax_psd.set_xlabel("k  (cycles / HR pixel)")
            ax_psd.set_ylabel(f"PSD  [{comp}]²")
            ax_psd.set_title(f"Binned Isotropic PSD — {comp}")
            ax_psd.legend(fontsize=13, loc="upper right")
            if not use_avg:
                ax_diff = axes[ch, 1]
                ax_diff.semilogy(k_ax, np.abs(pred_psd - gt_psd) + 1e-20,
                                 "b-",  lw=1.8, label="|Prediction − GT|")
                ax_diff.semilogy(k_ax, np.abs(lr_psd  - gt_psd) + 1e-20,
                                 "g--", lw=1.5, label="|ERA5 bilinear − GT|")
                ax_diff.axvline(cutoff, color="red", ls="--", lw=1.5)
                ax_diff.set_xlabel("k  (cycles / HR pixel)")
                ax_diff.set_ylabel("|ΔPsd|  (pred − GT)")
                ax_diff.set_title(f"|PSD difference| — {comp}")
                ax_diff.legend(fontsize=13, loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, save_name), dpi=300,
                    bbox_inches="tight")
        plt.close()
        print(f"  Saved → {os.path.join(results_dir, save_name)}")

    _binned_figure(use_avg=False, save_name="sample_fft_binned.pdf")
    _binned_figure(use_avg=True,  save_name="average_fft_binned.pdf")

    extent = [float(freq_w_grid.min()), float(freq_w_grid.max()),
              float(freq_h_grid.min()), float(freq_h_grid.max())]
    theta  = np.linspace(0, 2 * np.pi, 360)
    cx, cy = cutoff * np.cos(theta), cutoff * np.sin(theta)

    fig_2d, axes_2d = plt.subplots(2, 3, figsize=(4.2 * 3, 5.5 * 2),
                                    squeeze=False, constrained_layout=True)
    fig_2d.suptitle(
        f"2D log10-Power Spectrum — {case}\n"
        f"Averaged over {n_use} samples  |  "
        f"Cyan circle = HP cutoff ({cutoff:.3f} cy/px)", fontsize=18)

    for ch, comp in enumerate(comp_names):
        gt_pow_mean   = np.stack([_windowed_power(hr_true[i,  :, :, ch], win2d)
                                   for i in range(n_use)]).mean(axis=0)
        pred_pow_mean = np.stack([_windowed_power(hr_pred[i,  :, :, ch], win2d)
                                   for i in range(n_use)]).mean(axis=0)
        lr_pow_mean   = np.stack([_windowed_power(lr_input[i, :, :, ch], win2d)
                                   for i in range(n_use)]).mean(axis=0)
        log_gt   = np.log10(gt_pow_mean   + 1e-20)
        log_pred = np.log10(pred_pow_mean + 1e-20)
        log_lr   = np.log10(lr_pow_mean   + 1e-20)
        vmin = np.percentile(np.stack([log_gt, log_pred, log_lr]), 1)
        vmax = np.percentile(np.stack([log_gt, log_pred, log_lr]), 99)
        for col, (title, lp) in enumerate([
            ("CERRA (GT)", log_gt), ("Prediction", log_pred),
            ("ERA5 bilinear", log_lr),
        ]):
            ax = axes_2d[ch, col]
            im = ax.imshow(lp, origin="lower", extent=extent, aspect="auto",
                           cmap="inferno", vmin=vmin, vmax=vmax,
                           interpolation="nearest")
            ax.plot(cx, cy, color="cyan", lw=1.2, ls="--")
            ax.axvline(0, color="white", lw=0.6, ls=":", alpha=0.7)
            ax.axhline(0, color="white", lw=0.6, ls=":", alpha=0.7)
            ax.set_title(f"{title} — {comp}")
            ax.set_xlabel("k_x  (cy/px)")
            if col == 0:
                ax.set_ylabel("k_y  (cy/px)")
        fig_2d.colorbar(im, ax=axes_2d[ch, :].tolist(),
                        shrink=0.80, pad=0.02, aspect=30).set_label("log10(Power)")

    plt.savefig(os.path.join(results_dir, "sample_fft_2d.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close(fig_2d)
    print(f"  Saved → {os.path.join(results_dir, 'sample_fft_2d.pdf')}")
    print("[plot_fft_analysis] All figures and CSV files saved.")


# Wind-speed comparison figure  (identical to A_UNet_V2.py)
def plot_comparison_wind_speed(lr, pred, gt, lat_hr, lon_hr, lat_lr, lon_lr,
                               index=0, results_dir=".", case=""):
    def _pick(arr):
        arr = _to_numpy(arr)
        return arr[index] if arr.ndim == 4 else arr

    lr_s, pred_s, gt_s = _pick(lr), _pick(pred), _pick(gt)
    ws_lr,   _ = _uv_to_wspd_wdir(lr_s[..., 0],   lr_s[..., 1])
    ws_pred, _ = _uv_to_wspd_wdir(pred_s[..., 0], pred_s[..., 1])
    ws_gt,   _ = _uv_to_wspd_wdir(gt_s[..., 0],   gt_s[..., 1])
    err_abs  = np.abs(ws_pred - ws_gt)
    err_sign = ws_pred - ws_gt
    ext_hr = [float(lon_hr.min()), float(lon_hr.max()),
              float(lat_hr.min()), float(lat_hr.max())]
    ext_lr = [float(lon_lr.min()), float(lon_lr.max()),
              float(lat_lr.min()), float(lat_lr.max())]
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        USE_CARTOPY = True; proj = ccrs.PlateCarree()
    except ImportError:
        USE_CARTOPY = False; proj = None
    subplot_kw = {"projection": proj} if USE_CARTOPY else {}
    ws_vmax = float(max(ws_gt.max(), ws_pred.max(), ws_lr.max()))
    ws_vmin = float(min(ws_gt.min(), ws_pred.min(), ws_lr.min()))
    err_max = float(max(err_abs.max(), 0.01))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), subplot_kw=subplot_kw,
                             gridspec_kw={"hspace": 0.35, "wspace": 0.12})
    tag = f"  [{case}]" if case else ""
    fig.suptitle(f"Wind Speed Comparison — sample {index}{tag}", fontsize=16)

    def _imshow(ax, data, title, ext, cmap, vmin, vmax):
        im = ax.imshow(data, origin="upper", extent=ext, aspect="auto",
                       cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        if USE_CARTOPY:
            ax.add_feature(cfeature.BORDERS, linewidth=0.4)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.set_title(title, fontsize=12)
        return im

    _imshow(axes[0, 0], ws_lr,   "ERA5 bilinear (LR)",    ext_lr, "viridis", ws_vmin, ws_vmax)
    _imshow(axes[0, 1], ws_pred, "Prediction",             ext_hr, "viridis", ws_vmin, ws_vmax)
    im_ws  = _imshow(axes[0, 2], ws_gt,   "CERRA ground truth",   ext_hr, "viridis", ws_vmin, ws_vmax)
    im_err = _imshow(axes[1, 0], err_abs,  "|Pred − GT|  [m/s]", ext_hr, "Reds",   0.0, err_max)
    _imshow(axes[1, 1], err_sign, "Pred − GT  [m/s]",             ext_hr, "RdBu_r", -err_max, err_max)
    axes[1, 2].axis("off")
    fig.colorbar(im_ws, ax=axes[0, :].tolist(), orientation="vertical",
                 shrink=0.85, pad=0.02, aspect=30).set_label("Wind speed  [m/s]", fontsize=11)
    fig.colorbar(im_err, ax=axes[1, 2], orientation="vertical",
                 shrink=0.85, fraction=0.35).set_label("|Error|  [m/s]", fontsize=10)
    out = os.path.join(results_dir, f"comparison_windspeed_sample{index}.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved wind-speed comparison → {out}")


# Quiver wind plot  (identical to A_UNet_V2.py)
def plot_quiver_wind(pred, gt, lat_hr, lon_hr, index=0, results_dir=".",
                     quiver_stride=12, case=""):
    def _pick(arr):
        arr = _to_numpy(arr)
        return arr[index] if arr.ndim == 4 else arr

    pred_s, gt_s = _pick(pred), _pick(gt)
    ws_pred, _ = _uv_to_wspd_wdir(pred_s[..., 0], pred_s[..., 1])
    ws_gt,   _ = _uv_to_wspd_wdir(gt_s[..., 0],   gt_s[..., 1])
    lons = np.linspace(float(lon_hr.min()), float(lon_hr.max()), gt_s.shape[1])
    lats = np.linspace(float(lat_hr.min()), float(lat_hr.max()), gt_s.shape[0])
    LON, LAT = np.meshgrid(lons, lats)
    sy = sx = quiver_stride
    q_lon = LON[::sy, ::sx]; q_lat = LAT[::sy, ::sx]
    u_pred_q = pred_s[::sy, ::sx, 0]; v_pred_q = pred_s[::sy, ::sx, 1]
    u_gt_q   = gt_s[::sy,   ::sx, 0]; v_gt_q   = gt_s[::sy,   ::sx, 1]
    eps = 1e-6
    spd_pred_q = np.sqrt(u_pred_q**2 + v_pred_q**2) + eps
    spd_gt_q   = np.sqrt(u_gt_q**2   + v_gt_q**2)   + eps
    u_pred_n = u_pred_q / spd_pred_q; v_pred_n = v_pred_q / spd_pred_q
    u_gt_n   = u_gt_q   / spd_gt_q;   v_gt_n   = v_gt_q   / spd_gt_q
    ws_vmax = float(max(ws_gt.max(), ws_pred.max()))
    ext_hr  = [float(lon_hr.min()), float(lon_hr.max()),
               float(lat_hr.min()), float(lat_hr.max())]
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        USE_CARTOPY = True; proj = ccrs.PlateCarree()
    except ImportError:
        USE_CARTOPY = False; proj = None
    subplot_kw = {"projection": proj} if USE_CARTOPY else {}
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), subplot_kw=subplot_kw,
                             gridspec_kw={"wspace": 0.08})
    tag = f"  [{case}]" if case else ""
    fig.suptitle(f"Wind Speed & Direction — sample {index}{tag}\n"
                 f"Colour = wind speed [m/s]   |   Arrows = direction "
                 f"(normalised, stride = {quiver_stride} px)", fontsize=14)

    def _panel(ax, ws, u_n, v_n, title):
        im = ax.imshow(ws, origin="upper", extent=ext_hr, aspect="auto",
                       cmap="viridis", vmin=0.0, vmax=ws_vmax,
                       interpolation="nearest")
        quiver_kw = dict(color="white", scale=35, width=0.003,
                         headwidth=3.5, headlength=4, alpha=0.85)
        if USE_CARTOPY:
            ax.quiver(q_lon, q_lat, u_n, v_n, transform=proj, **quiver_kw)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        else:
            ax.quiver(q_lon, q_lat, u_n, v_n, **quiver_kw)
        ax.set_title(title, fontsize=13)
        return im

    im = _panel(axes[0], ws_pred, u_pred_n, v_pred_n, "Prediction")
    _panel(axes[1], ws_gt, u_gt_n, v_gt_n, "CERRA ground truth")
    fig.colorbar(im, ax=axes.tolist(), orientation="vertical",
                 shrink=0.85, pad=0.02, aspect=30).set_label(
                     "Wind speed  [m/s]", fontsize=12)
    out = os.path.join(results_dir, f"quiver_wind_sample{index}.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved quiver wind plot → {out}")


# Shared evaluation loop
# (used by both bicubic and SRCNN runners)
@tf.function(jit_compile=True)
def _compute_metrics(y_true, y_pred):
    error           = y_pred - y_true
    mse_per_sample  = tf.reduce_mean(tf.square(error), axis=[1, 2, 3])
    mse             = tf.reduce_mean(mse_per_sample)
    psnr_per_sample = (10.0 * tf.math.log(1.0 / (mse_per_sample + 1e-10))
                       / tf.math.log(10.0))
    psnr            = tf.reduce_mean(psnr_per_sample)
    ssim_u = tf.image.ssim(y_true[..., 0:1], y_pred[..., 0:1],
                           max_val=1.0, filter_size=7)
    ssim_v = tf.image.ssim(y_true[..., 1:2], y_pred[..., 1:2],
                           max_val=1.0, filter_size=7)
    ssim            = tf.reduce_mean((ssim_u + ssim_v) * 0.5)
    ssim_per_sample = (ssim_u + ssim_v) * 0.5
    return mse, psnr, ssim, mse_per_sample, psnr_per_sample, ssim_per_sample


def _evaluate_and_save(
    model_fn,          # callable: lr_batch → hr_pred  (tf.Tensor)
    datasets,
    minmax_stats,
    results_dir,
    baseline_name,     # "bicubic" or "srcnn"
    case,
    run_start_str,
    timer,
    total_start,
    n_train,
    n_val,
    lat_HR, lon_HR, lat_lr, lon_lr,
    best_val_loss = float("inf"),
    epochs_trained = 0,
):
    """
    Runs the test loop, computes metrics (scaled + unscaled), saves all
    standard outputs (plots, CSVs) mirroring A_UNet_V2.py section F–H.
    """
    hr_mins_np = np.array([float(v) for v in minmax_stats["hr"][0]],
                           dtype=np.float32)
    hr_maxs_np = np.array([float(v) for v in minmax_stats["hr"][1]],
                           dtype=np.float32)
    lr_mins_np = np.array([float(v) for v in minmax_stats["lr"][0]],
                           dtype=np.float32)
    lr_maxs_np = np.array([float(v) for v in minmax_stats["lr"][1]],
                           dtype=np.float32)
    hr_mins_bc = hr_mins_np.reshape(1, 1, 1, -1)
    hr_maxs_bc = hr_maxs_np.reshape(1, 1, 1, -1)
    lr_mins_bc = lr_mins_np.reshape(1, 1, 1, -1)
    lr_maxs_bc = lr_maxs_np.reshape(1, 1, 1, -1)

    # F. Test-set evaluation
    timer.start("sampling")
    print(f"\nComputing metrics on test dataset [{case}] — {baseline_name}:")

    mse_all, psnr_all, ssim_all          = [], [], []
    mse_per_sample_all                   = []
    psnr_per_sample_all                  = []
    ssim_per_sample_all                  = []
    hr_pred_list, hr_true_list, lr_list  = [], [], []

    for lr_batch, hr_batch in datasets["test_ds"]:
        hr_pred  = tf.cast(model_fn(lr_batch), tf.float32)
        y_true   = tf.cast(hr_batch, tf.float32)
        mse, psnr, ssim, mse_s, psnr_s, ssim_s = _compute_metrics(y_true, hr_pred)
        mse_all.append(mse);  psnr_all.append(psnr);  ssim_all.append(ssim)
        mse_per_sample_all.append(mse_s)
        psnr_per_sample_all.append(psnr_s)
        ssim_per_sample_all.append(ssim_s)
        hr_pred_list.append(hr_pred.numpy())
        hr_true_list.append(y_true.numpy())
        lr_list.append(lr_batch.numpy())

    mse_mean        = tf.reduce_mean(mse_all).numpy()
    psnr_mean       = tf.reduce_mean(psnr_all).numpy()
    ssim_mean_sc    = tf.reduce_mean(ssim_all).numpy()
    mse_per_sample  = tf.concat(mse_per_sample_all,  axis=0).numpy()
    psnr_per_sample = tf.concat(psnr_per_sample_all, axis=0).numpy()
    ssim_per_sample = tf.concat(ssim_per_sample_all, axis=0).numpy()

    print(f"\n--- SCALED METRICS [{case}] [{baseline_name}] ---")
    print(f"  MSE  : {mse_mean:.6f}")
    print(f"  PSNR : {psnr_mean:.3f} dB")
    print(f"  SSIM : {ssim_mean_sc:.4f}")

    hr_pred_all  = np.concatenate(hr_pred_list, axis=0)
    hr_true_all  = np.concatenate(hr_true_list, axis=0)
    lr_input_all = np.concatenate(lr_list,      axis=0)

    hr_pred_unscaled  = hr_pred_all  * (hr_maxs_bc - hr_mins_bc) + hr_mins_bc
    hr_batch_unscaled = hr_true_all  * (hr_maxs_bc - hr_mins_bc) + hr_mins_bc
    lr_batch_unscaled = lr_input_all * (lr_maxs_bc - lr_mins_bc) + lr_mins_bc

    mse_unscaled     = float(np.mean((hr_pred_unscaled - hr_batch_unscaled) ** 2))
    max_val_unscaled = float(hr_batch_unscaled.max())
    psnr_unscaled    = 10.0 * np.log10(
        max_val_unscaled ** 2 / (mse_unscaled + 1e-12))
    n_ssim = min(200, hr_batch_unscaled.shape[0])
    ssim_unscaled = float(tf.reduce_mean(tf.image.ssim(
        tf.constant(hr_batch_unscaled[:n_ssim]),
        tf.constant(hr_pred_unscaled[:n_ssim]),
        max_val=max_val_unscaled, filter_size=7,
    )).numpy())

    print(f"\n--- UNSCALED METRICS [{case}] [{baseline_name}] ---")
    print(f"  MSE  (m/s)²: {mse_unscaled:.6f}")
    print(f"  PSNR (dB)  : {psnr_unscaled:.3f}")
    print(f"  SSIM       : {ssim_unscaled:.4f}")

    timer.stop("sampling")

    # Save prediction arrays
    n_save = 200
    np.savez_compressed(
        os.path.join(results_dir, f"pred_arrays_{case}.npz"),
        hr_pred_unscaled = hr_pred_unscaled[:n_save],
        hr_true_unscaled = hr_batch_unscaled[:n_save],
        lr_unscaled      = lr_batch_unscaled[:n_save],
    )
    print(f"Saved prediction arrays → {results_dir}  (first {n_save} samples)")

    def _np(x):
        return x.numpy() if isinstance(x, tf.Tensor) else np.asarray(x)

    # Plots
    dummy_ssim = np.zeros((hr_batch_unscaled.shape[1],
                           hr_batch_unscaled.shape[2]), dtype=np.float32)
    plot_comparison_transformer(
        _np(lr_batch_unscaled), _np(hr_pred_unscaled), _np(hr_batch_unscaled),
        ssim_mean_u=dummy_ssim, ssim_std_u=dummy_ssim,
        ssim_mean_v=dummy_ssim, ssim_std_v=dummy_ssim,
        lat_hr=lat_HR, lon_hr=lon_HR, lat_lr=lat_lr, lon_lr=lon_lr,
        index=0, results_dir=results_dir,
    )

    plot_temporal_metrics(
        mse_per_sample  = mse_per_sample.astype(np.float32),
        psnr_per_sample = psnr_per_sample.astype(np.float32),
        ssim_per_sample = ssim_per_sample.astype(np.float32),
        n_train = n_train, n_val = n_val,
        results_dir = results_dir,
        uv_comp = True, rolling_window = 8, data_is_ordered = True,
    )

    plot_fft_analysis(
        results_dir   = results_dir,
        hr_pred       = _np(hr_pred_unscaled),
        hr_true       = _np(hr_batch_unscaled),
        lr_input      = _np(lr_batch_unscaled),
        cutoff        = FFT_CUTOFF,
        n_bins        = 128,
        n_samples_avg = 2920,
        case          = case,
    )

    plot_comparison_wind_speed(
        lr=_np(lr_batch_unscaled), pred=_np(hr_pred_unscaled),
        gt=_np(hr_batch_unscaled),
        lat_hr=lat_HR, lon_hr=lon_HR, lat_lr=lat_lr, lon_lr=lon_lr,
        index=0, results_dir=results_dir, case=case,
    )

    plot_quiver_wind(
        pred=_np(hr_pred_unscaled), gt=_np(hr_batch_unscaled),
        lat_hr=lat_HR, lon_hr=lon_HR,
        index=0, results_dir=results_dir, quiver_stride=12, case=case,
    )

    # G. Scaled metrics CSV  (same schema as A_UNet_V2.py)
    metrics_path = os.path.join(results_dir, "test_metrics.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "run_time", "case", "baseline",
                "val_loss_mse_scaled", "test_mse_scaled",
                "test_psnr", "test_ssim",
                "input_frames", "output_frames", "epochs"])
    with open(metrics_path, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(), case, baseline_name,
            round(best_val_loss, 6),
            round(float(mse_mean), 6),
            round(float(psnr_mean), 6),
            round(float(ssim_mean_sc), 4),
            INPUT_FRAMES, OUTPUT_FRAMES, epochs_trained])
    print(f"Saved scaled metrics → {metrics_path}")

    # G2. Unscaled metrics CSV
    unscaled_path = os.path.join(results_dir, "test_metrics_unscaled.csv")
    if not os.path.exists(unscaled_path):
        with open(unscaled_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "run_time", "case", "baseline",
                "test_mse_unscaled_m2s2",
                "test_psnr_unscaled_db",
                "test_ssim_unscaled",
                "n_ssim_samples"])
    with open(unscaled_path, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(), case, baseline_name,
            round(mse_unscaled,  6),
            round(psnr_unscaled, 4),
            round(ssim_unscaled, 4),
            n_ssim])
    print(f"Saved unscaled metrics → {unscaled_path}")

    # H. Timing CSV
    total_time  = time.perf_counter() - total_start
    csv_file    = os.path.join(results_dir, "computation_time_runs.csv")
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["run_start_time", "case", "baseline",
                             "preprocessing_s", "training_s",
                             "sampling_s", "total_s"])
        writer.writerow([
            run_start_str, case, baseline_name,
            f"{timer.times.get('preprocessing', 0.0):.4f}",
            f"{timer.times.get('training',      0.0):.4f}",
            f"{timer.times.get('sampling',      0.0):.4f}",
            f"{total_time:.4f}"])
    print(f"[{case}/{baseline_name}] Total elapsed: {total_time:.1f} s")


# Runner A — Bicubic interpolation
def run_bicubic(CASE, run_start_str, timer, total_start):
    print(f"\n{'='*70}")
    print(f"  Bicubic baseline — case: {CASE}")
    print(f"{'='*70}\n")

    RESULTS_DIR = os.path.join(BASE_DIR,
                               f"Results_ML/BASELINES/bicubic/{CASE}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with tf.device('/device:GPU:0'):

        # Preprocessing
        timer.start("preprocessing")
        case_dir = os.path.join(OUTPUT_ROOT, f"Data_{CASE}")
        coords_dir = os.path.join(SHARED_DIR, "coords")
        lat_HR = np.loadtxt(os.path.join(coords_dir, "lat_HR.npy"))
        lon_HR = np.loadtxt(os.path.join(coords_dir, "lon_HR.npy"))
        lat_lr = np.loadtxt(os.path.join(coords_dir, "lat_LR.npy"))
        lon_lr = np.loadtxt(os.path.join(coords_dir, "lon_LR.npy"))

        # Build pipeline with BICUBIC resize for the LR input
        datasets, minmax_stats = build_baseline_pipeline_from_prepared(
            case_dir         = case_dir,
            shared_dir       = SHARED_DIR,
            csv_path         = os.path.join(case_dir, f"statistics_{CASE}.csv"),
            batch_size       = SRCNN_BATCH_SIZE,
            shuffle_buffer   = 1000,
            lr_resize_method = "bicubic",
        )
        timer.stop("preprocessing")

        # No training: bicubic is a deterministic algorithm
        timer.start("training")
        timer.stop("training")

        # The "model" for bicubic is identity: the LR batch has already
        # been upsampled with bicubic by the pipeline, so the prediction
        # is simply the pipeline output itself.
        def bicubic_fn(lr_batch):
            return tf.cast(lr_batch, tf.float32)

        n_train = datasets["n_train"]
        n_val   = datasets["n_val"]

        _evaluate_and_save(
            model_fn       = bicubic_fn,
            datasets       = datasets,
            minmax_stats   = minmax_stats,
            results_dir    = RESULTS_DIR,
            baseline_name  = "bicubic",
            case           = CASE,
            run_start_str  = run_start_str,
            timer          = timer,
            total_start    = total_start,
            n_train        = n_train,
            n_val          = n_val,
            lat_HR         = lat_HR,
            lon_HR         = lon_HR,
            lat_lr         = lat_lr,
            lon_lr         = lon_lr,
            best_val_loss  = float("inf"),
            epochs_trained = 0,
        )

    gc.collect()
    tf.keras.backend.clear_session()
    print(f"[{CASE}/bicubic] GPU memory released.\n")


# Runner B — SRCNN
def run_srcnn(CASE, run_start_str, timer, total_start):
    print(f"\n{'='*70}")
    print(f"  SRCNN baseline — case: {CASE}  "
          f"({'train' if TRAIN_SRCNN else 'eval only'})")
    print(f"{'='*70}\n")

    RESULTS_DIR = os.path.join(BASE_DIR,
                               f"Results_ML/BASELINES/srcnn/{CASE}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_weights_path = os.path.join(RESULTS_DIR,
                                      "SRCNN_best_weights.weights.h5")

    with tf.device('/device:GPU:0'):

        # Preprocessing
        timer.start("preprocessing")
        case_dir = os.path.join(OUTPUT_ROOT, f"Data_{CASE}")
        coords_dir = os.path.join(SHARED_DIR, "coords")
        lat_HR = np.loadtxt(os.path.join(coords_dir, "lat_HR.npy"))
        lon_HR = np.loadtxt(os.path.join(coords_dir, "lon_HR.npy"))
        lat_lr = np.loadtxt(os.path.join(coords_dir, "lat_LR.npy"))
        lon_lr = np.loadtxt(os.path.join(coords_dir, "lon_LR.npy"))

        # SRCNN uses bilinear pre-upsampling (same as UNet pipeline)
        datasets, minmax_stats = build_baseline_pipeline_from_prepared(
            case_dir         = case_dir,
            shared_dir       = SHARED_DIR,
            csv_path         = os.path.join(case_dir, f"statistics_{CASE}.csv"),
            batch_size       = SRCNN_BATCH_SIZE,
            shuffle_buffer   = 3000,
            lr_resize_method = "bilinear",
        )
        timer.stop("preprocessing")

        n_train = datasets["n_train"]
        n_val   = datasets["n_val"]

        # Build model
        srcnn_model = build_srcnn_model()
        best_val_loss  = float("inf")
        epochs_trained = 0

        # Training (conditional)
        timer.start("training")
        if TRAIN_SRCNN:
            steps_per_epoch  = TRAIN_DURATION.get(SRCNN_TRAIN_CASE, n_train) \
                               // SRCNN_BATCH_SIZE
            validation_steps = max(1, n_val // SRCNN_BATCH_SIZE - 1)

            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    filepath          = model_weights_path,
                    monitor           = "val_loss",
                    save_best_only    = True,
                    save_weights_only = True,
                    verbose           = 1),
                keras.callbacks.EarlyStopping(
                    monitor              = "val_loss",
                    patience             = SRCNN_EARLY_STOPPING_PAT,
                    min_delta            = 1e-7,
                    mode                 = "min",
                    restore_best_weights = True),
            ]
            print(f"Training SRCNN — steps_per_epoch={steps_per_epoch}, "
                  f"validation_steps={validation_steps}, "
                  f"max_epochs={SRCNN_EPOCHS}")

            history = srcnn_model.fit(
                datasets["train_ds"],
                validation_data  = datasets["val_ds"],
                epochs           = SRCNN_EPOCHS,
                steps_per_epoch  = steps_per_epoch,
                validation_steps = validation_steps,
                verbose          = 1,
                callbacks        = callbacks,
            )
            best_val_loss  = float(min(history.history["val_loss"]))
            epochs_trained = len(history.history["val_loss"])

            # Loss curves
            best_epoch = int(np.argmin(history.history["val_loss"]))
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(history.history["loss"],     color="steelblue",
                    linewidth=1.5, label="train MSE loss")
            ax.plot(history.history["val_loss"], color="tomato",
                    linewidth=1.5, linestyle="--", label="val MSE loss")
            ax.axvline(best_epoch, color="black", linewidth=0.9, linestyle=":",
                       label=f"best epoch {best_epoch + 1}  "
                       f"(val={history.history['val_loss'][best_epoch]:.6f})")
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("MSE loss", fontsize=12)
            ax.legend(fontsize=10, framealpha=0.7)
            ax.set_title(f"SRCNN training curves — {CASE}", fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, "loss_history.pdf"),
                        dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Best val_loss = {best_val_loss:.6f}")
            np.save(os.path.join(RESULTS_DIR, "training_history.npy"),
                    history.history)

        else:
            # Evaluation-only: load existing checkpoint
            if not os.path.exists(model_weights_path):
                raise FileNotFoundError(
                    f"TRAIN_SRCNN=False but no checkpoint found:\n"
                    f"  {model_weights_path}\n"
                    f"Run with TRAIN_SRCNN=True first.")
            print(f"TRAIN_SRCNN=False — loading weights from:\n"
                  f"  {model_weights_path}")
            srcnn_model.load_weights(model_weights_path)

        timer.stop("training")

        # Evaluation
        @tf.function(jit_compile=True)
        def srcnn_fn(lr_batch):
            return srcnn_model(lr_batch, training=False)

        _evaluate_and_save(
            model_fn       = srcnn_fn,
            datasets       = datasets,
            minmax_stats   = minmax_stats,
            results_dir    = RESULTS_DIR,
            baseline_name  = "srcnn",
            case           = CASE,
            run_start_str  = run_start_str,
            timer          = timer,
            total_start    = total_start,
            n_train        = n_train,
            n_val          = n_val,
            lat_HR         = lat_HR,
            lon_HR         = lon_HR,
            lat_lr         = lat_lr,
            lon_lr         = lon_lr,
            best_val_loss  = best_val_loss,
            epochs_trained = epochs_trained,
        )

    del srcnn_model, datasets, minmax_stats
    gc.collect()
    tf.keras.backend.clear_session()
    print(f"[{CASE}/srcnn] GPU memory released.\n")


# Main
def main():
    if not CASES:
        raise ValueError("CASES is empty — add at least one case to run.")
    valid_cases = set(TRAIN_DURATION.keys())
    for c in CASES:
        if c not in valid_cases:
            raise ValueError(
                f"Unknown case '{c}'. Valid: {sorted(valid_cases)}")

    run_start     = datetime.now()
    run_start_str = run_start.isoformat(timespec="seconds")
    timer         = SectionTimer()
    total_start   = time.perf_counter()

    print(f"Baselines run — {len(CASES)} case(s): {CASES}")
    print(f"TRAIN_SRCNN = {TRAIN_SRCNN}")
    print(f"Started at: {run_start_str}\n")
    print("Output structure:")
    print(f"  Results_ML/BASELINES/bicubic/{{CASE}}/")
    print(f"  Results_ML/BASELINES/srcnn/{{CASE}}/\n")

    for i, CASE in enumerate(CASES, start=1):
        print(f"\n[{i}/{len(CASES)}] Running case: {CASE}")

        run_bicubic(CASE, run_start_str, timer, total_start)
        run_srcnn(CASE, run_start_str, timer, total_start)

    total_elapsed = time.perf_counter() - total_start
    print(f"\nAll cases completed: {CASES}")
    print(f"Total wall-clock time: {total_elapsed:.1f} s")
    print("Done.")


if __name__ == "__main__":
    main()
