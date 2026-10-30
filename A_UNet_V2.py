"""U-Net supervised downscaling of ERA5 to CERRA 100 m wind (evaluation and multi-case runner).

The U-Net (denoising_unet_edit.get_post_network) maps the LR input bilinearly upsampled to the HR
grid, (320, 400, INPUT_FRAMES), to the HR field (320, 400, 2). Data are the memmaps written by
prepare_training_data.py (folders Data_<case> and Data_Shared). Weights are loaded from
Results_ML/UNet/<case>/UNet_best_weights.weights.h5 (plain or spectral-loss checkpoints).

Outputs per case, in Results_ML/UNet_SSIM_Maps/<case>: spatial SSIM and metric maps
(plot_spatial_metric_maps) and computation_time_runs.csv; with SAVE_POSTPROCESSING_OUTPUTS
also pred_arrays_<case>.npz, comparison maps, temporal metrics, spectra (FFT), wind speed and
quiver figures and test_metrics.csv.

Configuration (module constants):
    CASES: training-set sizes run in sequence, keys of TRAIN_DURATION
        ("1year", "2year", "4year", "5year", "10year", "20year").
    TRAINING: True trains the U-Net (EPOCHS, EARLY_STOPPING_PATIENCE), False evaluates the
        loaded weights only.
    CONVERTING_DATA: True converts the raw GRIB/NetCDF files (debug run),
        False reads the prepared memmaps.
    PLOT_ONLY: True skips training and evaluation and redraws the spectral figures from
        the saved pred_arrays_<case>.npz.
    SAVE_POSTPROCESSING_OUTPUTS: True saves the arrays, figures and test_metrics.csv listed above.
    PRINT_UNSCALED_METRICS: True prints MSE, PSNR and SSIM in physical units.
    UV_COMP: True uses the u and v components, False only u.
    N_BEFORE, N_AFTER: time steps before and after the current one in the LR input.
    BATCH_SIZE, learning_rate, weight_decay: optimisation settings.
    FFT_CUTOFF: high-pass cutoff of the spectral figures, in cycles per pixel.
    BASE_DIR, SHARED_DIR: root folder and folder with the validation and test memmaps.

Paths BASE_DIR, PRETRAINED_BASE and RESULTS_DIR must be adapted to the local file system.
Author: Marion Falkenauer (with the help of Claude)
"""

import xarray as xr
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_XLA_FLAGS']         = '--tf_xla_auto_jit=2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import csv
import gc
import time
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import pandas as pd

from scipy.fft    import fft2 as scipy_fft2, fftshift, fftfreq as scipy_fftfreq
from scipy.signal import windows as scipy_windows

from ML_Methods_ERA5_CERRA import plot_comparison_transformer

from setup import widths, block_depth
from denoising_unet_edit import get_post_network

from plot_spatial_metric_maps import plot_spatial_metric_maps
plt.rcParams.update({
    "font.size":       25,
    "axes.labelsize":  20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 20,
    "axes.titlesize":  20,
})

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Physical GPUs: {len(gpus)}")
        print(f"Using GPU: {gpus[0].name}")
        print(f"Logical GPUs: {logical_gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU.")

BASE_DIR = "set to the base directory"
ERA5_files       = os.path.join(BASE_DIR, "Data/ERA5_Single_Year")
grib_files_ERA5  = sorted(glob.glob(os.path.join(ERA5_files,  "*.grib")))
print(f"Found {len(grib_files_ERA5)} ERA5 files.")

CERRA_files      = os.path.join(BASE_DIR, "Data/CERRA")
grib_files_CERRA = sorted(glob.glob(os.path.join(CERRA_files, "*.nc")))
print(f"Found {len(grib_files_CERRA)} CERRA files.")


PX_LAT_HR     = 320
PX_LON_HR     = 400
PX_LAT_LR     = 64
PX_LON_LR     = 80

RF              = 5
TRAINING        = False
CONVERTING_DATA = False
UV_COMP         = True

N_BEFORE = 0
N_AFTER  = 0

INPUT_FRAMES  = (N_BEFORE + 1 + N_AFTER) * (2 if UV_COMP else 1)
OUTPUT_FRAMES = 2 if UV_COMP else 1

PLOT_ONLY = False
SAVE_POSTPROCESSING_OUTPUTS = False
PRINT_UNSCALED_METRICS = False


PRETRAINED_WEIGHTS_PATH = "to specify"

CASES = ["1year", "2year", "4year"]

OUTPUT_ROOT = BASE_DIR
SHARED_DIR  = os.path.join(OUTPUT_ROOT, "Data_Shared")

TRAIN_DURATION = {
    "1year":   2920,
    "2year":   5840,
    "4year":   8768,
}

LON_COORDS,       LAT_COORDS       = [-5, 14.75], [44, 59.75]
LON_COORDS_CERRA, LAT_COORDS_CERRA = [-5, 14.95], [44, 59.95]
TIME_RANGE = ["2024-01-01", "2024-01-30"] # Try for a month for debugging purposes, to be changed 

BATCH_SIZE              = 2
learning_rate           = 1e-5
weight_decay            = 1e-6
EPOCHS                  = 0
EARLY_STOPPING_PATIENCE = 10

wind_vars = ["100u", "100v"]

FFT_CUTOFF = 0.05
HR_H       = PX_LAT_HR
HR_W       = PX_LON_HR


def domain_crop(df, lat_min, lat_max, lon_min, lon_max,variable_name="long"):
    """Crop a dataset to a latitude/longitude window.

    Args:
        df: xarray Dataset or DataArray.
        lat_min, lat_max: southern and northern bounds.
        lon_min, lon_max: western and eastern bounds.
        variable_name: "long" for dimensions latitude/longitude (latitude stored decreasing),
            "short" for dimensions lat/lon (latitude stored increasing).
    Returns:
        The cropped object.
    """

    print("Cropping to domain of interest...")
    if variable_name=="long":
        df_filtered = df.sel(
        latitude=slice(lat_max,lat_min),
        longitude=slice(lon_min,lon_max)
        )
    elif variable_name=="short":
        df_filtered = df.sel(
        lat=slice(lat_min,lat_max),
        lon=slice(lon_min,lon_max)
        )
    return df_filtered


def time_crop(df, start_time, end_time,time_var='time'):
    """Crop a dataset to a time range.

    Args:
        df: xarray Dataset or DataArray.
        start_time, end_time: bounds of the range.
        time_var: "time" (ERA5) or "valid_time" (CERRA).
    Returns:
        The cropped object.
    """
    if time_var=='time':
        df_cropped = df.sel(time=slice(start_time, end_time))
    elif time_var=='valid_time':
        df_cropped = df.sel(valid_time=slice(start_time, end_time))
    return df_cropped


def wind_speed_to_uv(xarray):
    """Convert wind speed and direction into u100 and v100.

    Args:
        xarray: Dataset with wind_speed and wind_direction (degrees, direction the wind blows from).
    Returns:
        Dataset with u100 and v100 in place of wind_speed and wind_direction.
    """
    print("Converting wind speed into u/v components")
    # Convert direction to radians
    wd_rad = np.deg2rad(xarray['wind_direction'])
    # Meteorological convention: direction the wind blows from
    xarray['u100'] = -xarray['wind_speed'] * np.sin(wd_rad)
    xarray['v100'] = -xarray['wind_speed'] * np.cos(wd_rad)
    xarray = xarray.drop_vars(['wind_speed', 'wind_direction'])
    return xarray


def plot_comparison_HR_LR(ds_HR,ds_LR,var="u100",time_idx=5,titles=("High resolution", "Low resolution"),cmap="viridis",extent=[-15, 35, 30, 70],savepath=None):
    """Plot a HR and a LR field side by side on maps with a shared colour scale.

    Args:
        ds_HR, ds_LR: datasets containing var.
        var: variable name.
        time_idx: time index to plot.
        titles: titles of the two panels.
        cmap: colormap name.
        extent: [lon_min, lon_max, lat_min, lat_max], or None.
        savepath: file to save the figure to (the figure is then closed).
    """

    # Compute common color limits (Dask-safe)
    vmin = min(
        ds_HR[var].isel(time=time_idx).min().compute().item(),
        ds_LR[var].isel(time=time_idx).min().compute().item()
    )
    vmax = max(
        ds_HR[var].isel(time=time_idx).max().compute().item(),
        ds_LR[var].isel(time=time_idx).max().compute().item()
    )

    # Create Cartopy axes
    fig, axes = plt.subplots(1, 2,figsize=(12, 4),subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True
    )

    # Plot HR
    p1 = ds_HR[var].isel(time=time_idx).plot(
        ax=axes[0],
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        add_colorbar=False
    )
    axes[0].set_title(titles[0])

    # Plot LR
    ds_LR[var].isel(time=time_idx).plot(
        ax=axes[1],
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        add_colorbar=False
    )
    axes[1].set_title(titles[1])

    # Add map features
    for ax in axes:
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, edgecolor="black", zorder=0)
        ax.add_feature(cfeature.OCEAN, zorder=0)

        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Shared colorbar
    cbar = fig.colorbar(p1, ax=axes, orientation="vertical", shrink=0.9)
    cbar.set_label(var)

    # Save or show
    if savepath is not None:
        plt.savefig(savepath, dpi=300)
        plt.close(fig)


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


def load_era5_variables(files, variables, level_type=None, file_type="cfgrib"):
    """Load GRIB or NetCDF files and concatenate them along time.

    Args:
        files: path or list of paths.
        variables: GRIB short names to read (cfgrib only).
        level_type: GRIB typeOfLevel filter (cfgrib only).
        file_type: "cfgrib", "netcdf4" or "h5netcdf".
    Returns:
        xarray.Dataset sorted by time.
    """
    if isinstance(files, str):
        files = [files]
    filter_keys    = {"shortName": variables}
    if level_type is not None:
        filter_keys["typeOfLevel"] = level_type
    backend_kwargs = {"filter_by_keys": filter_keys}
    datasets = []
    for f in files:
        print(f"Loading {f} ...")
        if file_type == "cfgrib":
            ds = xr.open_dataset(f, engine="cfgrib",
                                 backend_kwargs=backend_kwargs,
                                 chunks={"time": 100})
        elif file_type in ["netcdf4", "h5netcdf"]:
            ds = xr.open_dataset(f, engine=file_type, chunks={"time": 100})
        else:
            raise ValueError(f"Unsupported file_type: {file_type!r}")
        datasets.append(ds)
    return xr.concat(datasets, dim="time").sortby("time")


class SectionTimer:
    """Wall-clock timer for named sections of a run."""

    def __init__(self):
        """Create empty dictionaries of start times and durations."""
        self._starts = {}
        self.times   = {}

    def start(self, name):
        """Start the timer of a section.

        Args:
            name: section label.
        """
        self._starts[name] = time.perf_counter()

    def stop(self, name):
        """Stop a section and store its duration in seconds in self.times.

        Args:
            name: section label given to start.
        """
        self.times[name] = time.perf_counter() - self._starts[name]


def build_unet_pipeline(
        lr_mmap_path, hr_mmap_path, lr_shape, hr_shape,
        batch_size=BATCH_SIZE, split_ratios=(0.70, 0.15, 0.15),
        shuffle_buffer=3000,
):
    """Build train / val / test datasets from raw float32 memmap files.

    Min/max scaling constants are computed on the training split only. LR samples are scaled and
    bilinearly upsampled to the HR grid.

    Args:
        lr_mmap_path, hr_mmap_path: memmap files of the LR and HR arrays.
        lr_shape, hr_shape: shapes (N, H, W, C) of the two arrays.
        batch_size: batch size.
        split_ratios: (train, val, test) fractions.
        shuffle_buffer: shuffle buffer size.
    Returns:
        (datasets, minmax_stats). datasets has train_ds, val_ds, test_ds (batches of (lr_up, hr)
        scaled to [0, 1]) and n_train, n_val, n_test. minmax_stats is
        {"lr": (mins, maxs), "hr": (mins, maxs)} with one tf scalar per channel.
    """
    print("Opening memory-mapped files (read-only)...")
    lr_mmap = np.memmap(lr_mmap_path, dtype="float32", mode="r", shape=lr_shape)
    hr_mmap = np.memmap(hr_mmap_path, dtype="float32", mode="r", shape=hr_shape)
    N                  = lr_shape[0]
    H_lr, W_lr, lr_ch = lr_shape[1], lr_shape[2], lr_shape[3]
    H_hr, W_hr, hr_ch = hr_shape[1], hr_shape[2], hr_shape[3]
    print(f"  LR memmap : {lr_shape}  HR memmap : {hr_shape}")
    assert lr_shape[0] == hr_shape[0]

    def lr_generator():
        """Yield LR samples from the memmap."""
        for i in range(N): yield lr_mmap[i].astype(np.float32)

    def hr_generator():
        """Yield HR samples from the memmap."""
        for i in range(N): yield hr_mmap[i].astype(np.float32)

    tf_ds_lr = tf.data.Dataset.from_generator(
        lr_generator,
        output_signature=tf.TensorSpec(shape=(H_lr, W_lr, lr_ch), dtype=tf.float32))
    tf_ds_hr = tf.data.Dataset.from_generator(
        hr_generator,
        output_signature=tf.TensorSpec(shape=(H_hr, W_hr, hr_ch), dtype=tf.float32))

    n_train = int(split_ratios[0] * N)
    n_val   = int(split_ratios[1] * N)
    n_test  = N - n_train - n_val

    def compute_channel_minmax(dataset, n_channels):
        """Compute the minimum and maximum of each channel of a dataset.

        Args:
            dataset: tf.data.Dataset of (H, W, C) samples.
            n_channels: number of channels.
        Returns:
            (mins, maxs), lists of tf scalars.
        """
        mins, maxs = [], []
        for c in range(n_channels):
            min_c = tf.constant( 1e9, dtype=tf.float32)
            max_c = tf.constant(-1e9, dtype=tf.float32)
            for sample in dataset:
                ch_vals = tf.cast(sample[..., c], tf.float32)
                min_c = tf.minimum(min_c, tf.reduce_min(ch_vals))
                max_c = tf.maximum(max_c, tf.reduce_max(ch_vals))
            mins.append(min_c); maxs.append(max_c)
        return mins, maxs

    lr_mins, lr_maxs = compute_channel_minmax(tf_ds_lr.take(n_train), lr_ch)
    hr_mins, hr_maxs = compute_channel_minmax(tf_ds_hr.take(n_train), hr_ch)

    def normalize_lr(x):
        """Min-max scale a LR sample channel by channel."""
        return tf.stack([(x[..., i] - lr_mins[i]) / (lr_maxs[i] - lr_mins[i] + 1e-8)
                         for i in range(lr_ch)], axis=-1)

    def normalize_hr(x):
        """Min-max scale a HR sample channel by channel."""
        return tf.stack([(x[..., i] - hr_mins[i]) / (hr_maxs[i] - hr_mins[i] + 1e-8)
                         for i in range(hr_ch)], axis=-1)

    tf_ds_lr_up = (tf_ds_lr.map(normalize_lr, num_parallel_calls=tf.data.AUTOTUNE)
                            .map(lambda x: tf.image.resize(x, (H_hr, W_hr), method="bilinear"),
                                 num_parallel_calls=tf.data.AUTOTUNE))
    tf_ds_hr_s  = tf_ds_hr.map(normalize_hr, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = tf.data.Dataset.zip((tf_ds_lr_up, tf_ds_hr_s))
    dataset = dataset.shuffle(min(shuffle_buffer, N), reshuffle_each_iteration=True, seed=42)

    train_ds = (dataset.take(n_train)
                .batch(batch_size, drop_remainder=True).repeat()
                .prefetch(tf.data.AUTOTUNE))
    val_ds   = (dataset.skip(n_train).take(n_val)
                .batch(batch_size, drop_remainder=True)
                .prefetch(tf.data.AUTOTUNE))
    test_ds  = (dataset.skip(n_train + n_val).take(n_test)
                .batch(batch_size, drop_remainder=True)
                .prefetch(tf.data.AUTOTUNE))
    gc.collect()

    return ({"train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds,
             "n_train": n_train, "n_val": n_val, "n_test": n_test},
            {"lr": (lr_mins, lr_maxs), "hr": (hr_mins, hr_maxs)})


def build_unet_pipeline_from_prepared(
        case_dir, shared_dir, csv_path,
        batch_size=BATCH_SIZE, shuffle_buffer=3000,
):
    """Build train / val / test datasets from the memmaps of prepare_training_data.py.

    Args:
        case_dir: folder Data_<case> containing train/.
        shared_dir: folder Data_Shared containing val/ and test/.
        csv_path: statistics CSV of the training case (min/max of ERA5 and CERRA).
        batch_size: batch size.
        shuffle_buffer: shuffle buffer of the training dataset.
    Returns:
        (datasets, minmax_stats), with the same structure as build_unet_pipeline.
    """
    def _sidecar(npy_path):
        """Read the shape and dtype from the sidecar file of a memmap."""
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
        shapes[split] = {"lr": lr_sh, "hr": hr_sh,
                        "lr_path": lr_path, "hr_path": hr_path,
                        "lr_dtype": lr_dtype, "hr_dtype": hr_dtype}

    expected_lr_ch = INPUT_FRAMES
    expected_hr_ch = OUTPUT_FRAMES

    print("\n  Tensor shape verification:")
    for split_name, info in shapes.items():
        lr_sh, hr_sh = info["lr"], info["hr"]
        print(f"    {split_name:5s}  LR {lr_sh}  HR {hr_sh}")
        if lr_sh[-1] != expected_lr_ch:
            raise ValueError(
                f"[{split_name}] LR channel mismatch: {lr_sh[-1]} != {expected_lr_ch}")
        if hr_sh[-1] != expected_hr_ch:
            raise ValueError(
                f"[{split_name}] HR channel mismatch: {hr_sh[-1]} != {expected_hr_ch}")
        if lr_sh[0] != hr_sh[0]:
            raise ValueError(
                f"[{split_name}] Timestep mismatch: LR={lr_sh[0]}, HR={hr_sh[0]}")
        print(f"    [{split_name}] OK — {lr_sh[0]} samples")

    stats_df = pd.read_csv(csv_path, index_col=0)

    def _get_minmax(row_label, n_channels):
        """Return the min and max constants of a dataset row of the statistics CSV for each channel.

        Args:
            row_label: "ERA5" or "CERRA".
            n_channels: number of channels (u and v alternate).
        Returns:
            (mins, maxs), lists of tf scalars.
        """
        comp_map = {0: ("min_u", "max_u"), 1: ("min_v", "max_v")}
        mins, maxs = [], []
        for c in range(n_channels):
            min_col, max_col = comp_map[c % 2]
            mins.append(tf.constant(float(stats_df.loc[row_label, min_col]),
                                    dtype=tf.float32))
            maxs.append(tf.constant(float(stats_df.loc[row_label, max_col]),
                                    dtype=tf.float32))
        return mins, maxs

    lr_mins, lr_maxs = _get_minmax("ERA5",  expected_lr_ch)
    hr_mins, hr_maxs = _get_minmax("CERRA", expected_hr_ch)

    def normalize_lr(x):
        """Min-max scale a LR sample channel by channel."""
        return tf.stack(
            [(x[..., i] - lr_mins[i]) / (lr_maxs[i] - lr_mins[i] + 1e-8)
             for i in range(expected_lr_ch)], axis=-1)

    def normalize_hr(x):
        """Min-max scale a HR sample channel by channel."""
        return tf.stack(
            [(x[..., i] - hr_mins[i]) / (hr_maxs[i] - hr_mins[i] + 1e-8)
             for i in range(expected_hr_ch)], axis=-1)

    def upsample_lr(x):
        """Bilinearly resize a LR sample to the HR grid."""
        return tf.image.resize(x, (PX_LAT_HR, PX_LON_HR), method="bilinear")

    def _make_split_ds(split, shuffle, repeat):
        """Build the dataset of one split.

        Args:
            split: "train", "val" or "test".
            shuffle: shuffle with shuffle_buffer.
            repeat: repeat indefinitely.
        Returns:
            (dataset, number of samples).
        """
        info         = shapes[split]
        lr_sh, hr_sh = info["lr"], info["hr"]
        lr_mm = np.memmap(info["lr_path"], dtype=info["lr_dtype"], mode="r", shape=lr_sh)
        hr_mm = np.memmap(info["hr_path"], dtype=info["hr_dtype"], mode="r", shape=hr_sh)
        N_s                  = lr_sh[0]
        H_lr, W_lr, lr_ch_s = lr_sh[1], lr_sh[2], lr_sh[3]
        H_hr, W_hr, hr_ch_s = hr_sh[1], hr_sh[2], hr_sh[3]

        def _lr_gen():
            """Yield LR samples."""
            for i in range(N_s): yield lr_mm[i].astype(np.float32)

        def _hr_gen():
            """Yield HR samples."""
            for i in range(N_s): yield hr_mm[i].astype(np.float32)

        tf_lr = (tf.data.Dataset.from_generator(
                    _lr_gen,
                    output_signature=tf.TensorSpec(
                        shape=(H_lr, W_lr, lr_ch_s), dtype=tf.float32))
                 .map(normalize_lr, num_parallel_calls=tf.data.AUTOTUNE)
                 .map(upsample_lr,  num_parallel_calls=tf.data.AUTOTUNE))
        tf_hr = (tf.data.Dataset.from_generator(
                    _hr_gen,
                    output_signature=tf.TensorSpec(
                        shape=(H_hr, W_hr, hr_ch_s), dtype=tf.float32))
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

    return ({"train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds,
             "n_train": n_train, "n_val": n_val, "n_test": n_test},
            {"lr": (lr_mins, lr_maxs), "hr": (hr_mins, hr_maxs)})


_YEAR_START      = "2024-01-01"
_YEAR_END        = "2024-12-31 21:00"
_FREQ_HOURS      = 3
_FULL_YEAR_TIMES = None


def _build_year_times():
    """Return the 3-hourly time index of the year 2024, built once and cached."""
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
    """Plot per-sample MSE, PSNR and SSIM over time and save the series.

    Writes temporal_metrics.pdf and temporal_metrics.csv in results_dir.

    Args:
        mse_per_sample, psnr_per_sample, ssim_per_sample: arrays (N_test,).
        n_train, n_val: sizes of the training and validation sets, used as sample index offset.
        results_dir: output directory.
        uv_comp: True when u and v are averaged.
        rolling_window: window of the rolling mean in samples.
        data_is_ordered: False adds a warning that the test set is shuffled.
    """
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker

    n_test     = len(mse_per_sample)
    all_times  = _build_year_times()
    offset     = n_train + n_val
    test_times = (all_times[0:n_test]
                  if n_test <= len(all_times)
                  else pd.date_range(_YEAR_START, periods=n_test,
                                     freq=f"{_FREQ_HOURS}h"))

    def _rolling(arr):
        """Centred rolling mean of an array."""
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
        ax.plot(test_times, raw,
                label="per-sample", **raw_kw)
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
    print(f"Saved temporal metrics CSV  → "
          f"{os.path.join(results_dir, 'temporal_metrics.csv')}")


def _to_numpy(arr) -> np.ndarray:
    """Convert a tensor or array-like to a float32 numpy array."""
    if isinstance(arr, tf.Tensor):
        return arr.numpy().astype(np.float32)
    return np.asarray(arr, dtype=np.float32)


def _uv_to_wspd_wdir(u: np.ndarray, v: np.ndarray):
    """Convert wind components to speed and meteorological direction.

    Args:
        u, v: zonal and meridional components in m/s.
    Returns:
        (ws, wdir) float32 arrays: speed in m/s and direction in degrees the wind blows from.
    """
    u64  = u.astype(np.float64)
    v64  = v.astype(np.float64)
    ws   = np.sqrt(u64 ** 2 + v64 ** 2)
    wdir = (270.0 - np.degrees(np.arctan2(v64, u64))) % 360.0
    return ws.astype(np.float32), wdir.astype(np.float32)


def _build_analysis_window(H: int, W: int) -> np.ndarray:
    """Build a 2-D Hann window normalised by its mean.

    Args:
        H, W: window size.
    Returns:
        float32 array (H, W).
    """
    w_h   = scipy_windows.hann(H, sym=True).astype(np.float32)
    w_w   = scipy_windows.hann(W, sym=True).astype(np.float32)
    win2d = np.outer(w_h, w_w)
    return (win2d / win2d.mean()).astype(np.float32)


def _compute_radial_grid(H: int, W: int):
    """Frequency axes and radial wavenumber of an (H, W) field.

    Args:
        H, W: field size.
    Returns:
        (freq_h, freq_w, radial): shifted frequencies in cycles per pixel, (H,) and (W,),
        and the radial wavenumber grid (H, W).
    """
    freq_h = fftshift(scipy_fftfreq(H)).astype(np.float32)
    freq_w = fftshift(scipy_fftfreq(W)).astype(np.float32)
    FH, FW = np.meshgrid(freq_h, freq_w, indexing="ij")
    return freq_h, freq_w, np.sqrt(FH ** 2 + FW ** 2).astype(np.float32)


def _windowed_power(field_2d: np.ndarray, window2d: np.ndarray) -> np.ndarray:
    """Windowed 2-D power spectrum.

    Args:
        field_2d: field (H, W).
        window2d: window (H, W).
    Returns:
        Shifted power spectrum (H, W) of the mean-removed windowed field.
    """
    xw = ((field_2d.astype(np.float64) - field_2d.mean())
          * window2d.astype(np.float64))
    return np.abs(fftshift(scipy_fft2(xw, norm="ortho"))) ** 2


def _radial_psd(field_2d: np.ndarray, window2d: np.ndarray,
                n_bins: int = 128, cutoff: float = None):
    """Azimuthally averaged power spectrum.

    Args:
        field_2d: field (H, W).
        window2d: window (H, W).
        n_bins: number of radial bins.
        cutoff: if given, power at wavenumbers up to cutoff is set to zero.
    Returns:
        (k_mid, psd): bin centres in cycles per pixel and mean power per bin, both of length n_bins.
    """
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


def plot_comparison_wind_speed(
    lr:          np.ndarray,
    pred:        np.ndarray,
    gt:          np.ndarray,
    lat_hr:      np.ndarray,
    lon_hr:      np.ndarray,
    lat_lr:      np.ndarray,
    lon_lr:      np.ndarray,
    index:       int = 0,
    results_dir: str = ".",
    case:        str = "",
):
    """Save a wind-speed comparison figure of one sample.

    Top row: LR (bilinear), prediction and ground truth. Bottom row: absolute and signed error.
    Writes comparison_windspeed_sample{index}.pdf in results_dir.

    Args:
        lr, pred, gt: unscaled arrays (B, H, W, 2) or (H, W, 2); channels are u and v in m/s.
        lat_hr, lon_hr, lat_lr, lon_lr: coordinate vectors of the HR and LR grids.
        index: sample index when a batch dimension is present.
        results_dir: output directory.
        case: label used in the title.
    """
    # Extract single sample
    def _pick(arr):
        """Return the sample at index, or the array itself when it has no batch dimension."""
        arr = _to_numpy(arr)
        return arr[index] if arr.ndim == 4 else arr

    lr_s, pred_s, gt_s = _pick(lr), _pick(pred), _pick(gt)

    # Compute wind speed (scalar magnitude) from U/V
    ws_lr,   _ = _uv_to_wspd_wdir(lr_s[..., 0],   lr_s[..., 1])
    ws_pred, _ = _uv_to_wspd_wdir(pred_s[..., 0], pred_s[..., 1])
    ws_gt,   _ = _uv_to_wspd_wdir(gt_s[..., 0],   gt_s[..., 1])

    # Compute error fields
    err_abs  = np.abs(ws_pred - ws_gt)
    err_sign = ws_pred - ws_gt

    # Geographic extents
    ext_hr = [float(lon_hr.min()), float(lon_hr.max()),
              float(lat_hr.min()), float(lat_hr.max())]
    ext_lr = [float(lon_lr.min()), float(lon_lr.max()),
              float(lat_lr.min()), float(lat_lr.max())]

    # Optional Cartopy geographic axes
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        USE_CARTOPY = True
        proj = ccrs.PlateCarree()
    except ImportError:
        USE_CARTOPY = False
        proj = None

    subplot_kw = {"projection": proj} if USE_CARTOPY else {}

    # Shared colour scales
    ws_vmax = float(max(ws_gt.max(), ws_pred.max(), ws_lr.max()))
    err_max = float(max(err_abs.max(), 0.01))
    ws_vmin = float(min(ws_gt.min(), ws_pred.min(), ws_lr.min()))

    # Build figure
    fig, axes = plt.subplots(
        2, 3, figsize=(18, 10),
        subplot_kw=subplot_kw,
        gridspec_kw={"hspace": 0.35, "wspace": 0.12},
    )
    tag = f"  [{case}]" if case else ""
    fig.suptitle(f"Wind Speed Comparison — sample {index}{tag}", fontsize=16)

    def _imshow(ax, data, title, ext, cmap, vmin, vmax):
        """Draw one map panel and return the image handle."""
        im = ax.imshow(data, origin="upper", extent=ext, aspect="auto",
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        if USE_CARTOPY:
            ax.add_feature(cfeature.BORDERS, linewidth=0.4)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.set_title(title, fontsize=12)
        return im

    # Row 0: LR | Pred | GT
    _imshow(axes[0, 0], ws_lr,  "ERA5 bilinear (LR)", ext_lr, "viridis",ws_vmin, ws_vmax)
    _imshow(axes[0, 1], ws_pred,"Prediction",          ext_hr, "viridis", ws_vmin, ws_vmax)
    im_ws = _imshow(axes[0, 2], ws_gt, "CERRA ground truth", ext_hr, "viridis", ws_vmin, ws_vmax)

    # Row 1: absolute error | signed error | (blank - used for colour bar)
    im_err = _imshow(axes[1, 0], err_abs,  "|Pred − GT|  [m/s]", ext_hr,
                     "Reds",   0.0,     err_max)
    im_sgn = _imshow(axes[1, 1], err_sign, "Pred − GT  [m/s]",   ext_hr,
                     "RdBu_r", -err_max, err_max)
    axes[1, 2].axis("off")

    # Colour bars
    cbar_ws = fig.colorbar(im_ws, ax=axes[0, :].tolist(),
                           orientation="vertical", shrink=0.85,
                           pad=0.02, aspect=30)
    cbar_ws.set_label("Wind speed  [m/s]", fontsize=11)

    fig.colorbar(im_err, ax=axes[1, 2],
                 orientation="vertical", shrink=0.85,
                 fraction=0.35).set_label("|Error|  [m/s]", fontsize=10)

    out = os.path.join(results_dir,
                       f"comparison_windspeed_sample{index}.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved wind-speed comparison → {out}")


def plot_quiver_wind(
    pred:          np.ndarray,
    gt:            np.ndarray,
    lat_hr:        np.ndarray,
    lon_hr:        np.ndarray,
    index:         int = 0,
    results_dir:   str = ".",
    quiver_stride: int = 12,
    case:          str = "",
):
    """Save a quiver figure of one sample: wind speed as colour, direction as unit-length arrows.

    Writes quiver_wind_sample{index}.pdf in results_dir, with ground truth and prediction panels.

    Args:
        pred, gt: unscaled arrays (B, H, W, 2) or (H, W, 2); channels are u and v in m/s.
        lat_hr, lon_hr: coordinate vectors of the HR grid.
        index: sample index when a batch dimension is present.
        results_dir: output directory.
        quiver_stride: pixel spacing between arrows.
        case: label used in the title.
    """
    # Extract single sample
    def _pick(arr):
        """Return the sample at index, or the array itself when it has no batch dimension."""
        arr = _to_numpy(arr)
        return arr[index] if arr.ndim == 4 else arr

    pred_s, gt_s = _pick(pred), _pick(gt)
    H, W = gt_s.shape[:2]

    # Wind speed from U/V
    ws_pred, _ = _uv_to_wspd_wdir(pred_s[..., 0], pred_s[..., 1])
    ws_gt,   _ = _uv_to_wspd_wdir(gt_s[..., 0],   gt_s[..., 1])

    # Build coordinate grids and subsampled arrow arrays
    lons = np.linspace(float(lon_hr.min()), float(lon_hr.max()), W)
    lats = np.linspace(float(lat_hr.min()), float(lat_hr.max()), H)
    LON, LAT = np.meshgrid(lons, lats)

    sy = sx = quiver_stride
    q_lon = LON[::sy, ::sx]
    q_lat = LAT[::sy, ::sx]

    u_pred_q = pred_s[::sy, ::sx, 0];  v_pred_q = pred_s[::sy, ::sx, 1]
    u_gt_q   = gt_s[::sy,   ::sx, 0];  v_gt_q   = gt_s[::sy,   ::sx, 1]

    # Normalise arrows to unit length (direction only)
    eps = 1e-6
    spd_pred_q = np.sqrt(u_pred_q ** 2 + v_pred_q ** 2) + eps
    spd_gt_q   = np.sqrt(u_gt_q   ** 2 + v_gt_q   ** 2) + eps
    u_pred_n, v_pred_n = u_pred_q / spd_pred_q, v_pred_q / spd_pred_q
    u_gt_n,   v_gt_n   = u_gt_q   / spd_gt_q,   v_gt_q   / spd_gt_q

    # Shared wind-speed colour scale
    ws_vmax = float(max(ws_gt.max(), ws_pred.max()))
    ext_hr  = [float(lon_hr.min()), float(lon_hr.max()),
               float(lat_hr.min()), float(lat_hr.max())]

    # Optional Cartopy
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        USE_CARTOPY = True
        proj = ccrs.PlateCarree()
    except ImportError:
        USE_CARTOPY = False
        proj = None

    subplot_kw = {"projection": proj} if USE_CARTOPY else {}

    # Build figure
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 7),
        subplot_kw=subplot_kw,
        gridspec_kw={"wspace": 0.08},
    )
    tag = f"  [{case}]" if case else ""
    fig.suptitle(
        f"Wind Speed & Direction — sample {index}{tag}\n"
        f"Colour = wind speed [m/s]   |   Arrows = direction "
        f"(normalised, stride = {quiver_stride} px)",
        fontsize=14,
    )

    def _panel(ax, ws, u_n, v_n, title):
        """Draw the wind speed map with direction arrows on an axis and return the image handle."""
        im = ax.imshow(ws, origin="upper", extent=ext_hr, aspect="auto",
                       cmap="viridis", vmin=0.0, vmax=ws_vmax,
                       interpolation="nearest")
        quiver_kw = dict(color="white", scale=35, width=0.003,
                         headwidth=3.5, headlength=4, alpha=0.85)
        if USE_CARTOPY:
            ax.quiver(q_lon, q_lat, u_n, v_n,
                      transform=proj, **quiver_kw)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        else:
            ax.quiver(q_lon, q_lat, u_n, v_n, **quiver_kw)
        ax.set_title(title, fontsize=13)
        return im

    im = _panel(axes[0], ws_pred, u_pred_n, v_pred_n, "Prediction")
    _panel(axes[1], ws_gt,   u_gt_n,   v_gt_n,   "CERRA ground truth")

    fig.colorbar(im, ax=axes.tolist(), orientation="vertical",
                 shrink=0.85, pad=0.02, aspect=30).set_label(
                     "Wind speed  [m/s]", fontsize=12)

    out = os.path.join(results_dir, f"quiver_wind_sample{index}.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved quiver wind plot → {out}")


def plot_fft_analysis(
    results_dir:   str,
    hr_pred:       np.ndarray,
    hr_true:       np.ndarray,
    lr_input:      np.ndarray,
    cutoff:        float = 0.05,
    n_bins:        int   = 128,
    n_samples_avg: int   = 2920,
    case:          str   = "",
):
    """Save spectral comparison figures and PSD tables of prediction, ground truth and LR input.

    Writes sample_fft_binned.pdf, average_fft_binned.pdf, sample_fft_2d.pdf,
    sample_fft_spectra.csv and average_fft_spectra.csv in results_dir. The CSV columns are
    k_mid_cy_per_px, gt_psd_u, pred_psd_u, lr_psd_u, gt_psd_v, pred_psd_v, lr_psd_v.

    Args:
        results_dir: output directory.
        hr_pred, hr_true: unscaled arrays (N, H, W, 2) of predictions and ground truth.
        lr_input: unscaled LR array (N, h, w, 2), bilinearly resized to the HR grid if needed.
        cutoff: high-pass cutoff in cycles per pixel drawn on the figures.
        n_bins: number of radial bins.
        n_samples_avg: maximum number of samples used for the averaged spectra.
        case: label used in the titles.
    """
    hr_pred  = _to_numpy(hr_pred)
    hr_true  = _to_numpy(hr_true)
    lr_input = _to_numpy(lr_input)
    HR_H, HR_W = hr_true.shape[1], hr_true.shape[2]

    # Upsample LR to HR resolution if needed
    if lr_input.shape[1] != HR_H or lr_input.shape[2] != HR_W:
        lr_input = tf.image.resize(
            lr_input, (HR_H, HR_W), method="bilinear"
        ).numpy().astype(np.float32)
        print(f"  [plot_fft_analysis] LR upsampled to {HR_H}×{HR_W}.")

    print(f"\n[plot_fft_analysis] case={case!r}  n_use={min(n_samples_avg, hr_pred.shape[0])}")

    win2d      = _build_analysis_window(HR_H, HR_W)
    comp_names = ["U (100 m)", "V (100 m)"]
    freq_h_grid, freq_w_grid, _ = _compute_radial_grid(HR_H, HR_W)
    n_use = min(n_samples_avg, hr_pred.shape[0])

    # Pre-compute the PSD of the first sample and of the sample average
    single, avg = {}, {}
    for ch in range(2):
        # 1-D binned PSD: single sample
        k, gt_s    = _radial_psd(hr_true[0,  :, :, ch], win2d, n_bins=n_bins)
        _,  pred_s = _radial_psd(hr_pred[0,  :, :, ch], win2d, n_bins=n_bins)
        _,  lr_s   = _radial_psd(lr_input[0, :, :, ch], win2d, n_bins=n_bins)
        single[ch] = (k, gt_s, pred_s, lr_s)

        # 1-D binned PSD: averaged over n_use samples
        gt_stack   = np.stack([_radial_psd(hr_true[i,  :, :, ch], win2d, n_bins)[1]
                                for i in range(n_use)])
        pred_stack = np.stack([_radial_psd(hr_pred[i,  :, :, ch], win2d, n_bins)[1]
                                for i in range(n_use)])
        lr_stack   = np.stack([_radial_psd(lr_input[i, :, :, ch], win2d, n_bins)[1]
                                for i in range(n_use)])
        avg[ch] = (k, gt_stack.mean(0), pred_stack.mean(0), lr_stack.mean(0))

    df_s = pd.DataFrame({
        "k_mid_cy_per_px": single[0][0],
        "gt_psd_u":   single[0][1], "pred_psd_u": single[0][2], "lr_psd_u": single[0][3],
        "gt_psd_v":   single[1][1], "pred_psd_v": single[1][2], "lr_psd_v": single[1][3],
    })
    csv_s = os.path.join(results_dir, "sample_fft_spectra.csv")
    df_s.to_csv(csv_s, index=False, float_format="%.8e")
    print(f"  Saved spectra CSV (single sample) → {csv_s}")

    df_a = pd.DataFrame({
        "k_mid_cy_per_px": avg[0][0],
        "gt_psd_u":   avg[0][1], "pred_psd_u": avg[0][2], "lr_psd_u": avg[0][3],
        "gt_psd_v":   avg[1][1], "pred_psd_v": avg[1][2], "lr_psd_v": avg[1][3],
    })
    csv_a = os.path.join(results_dir, "average_fft_spectra.csv")
    df_a.to_csv(csv_a, index=False, float_format="%.8e")
    print(f"  Saved spectra CSV (averaged n={n_use}) → {csv_a}")

    def _binned_figure(use_avg, save_name):
        """Save the binned PSD figure of the first sample (with PSD difference) or averaged over the samples."""
        fig, axes = (plt.subplots(2, 1, figsize=(9, 13)) if use_avg
                     else plt.subplots(2, 2, figsize=(18, 13)))
        tag_str = "Averaged" if use_avg else "Single-sample"
        fig.suptitle(f"{tag_str} — Binned Isotropic PSD — {case}\n"
                     f"HP cutoff = {cutoff:.3f} cy/px")

        for ch, comp in enumerate(comp_names):
            ax_psd = axes[ch] if use_avg else axes[ch, 0]
            k_ax, gt_psd, pred_psd, lr_psd = (avg if use_avg else single)[ch]

            ax_psd.semilogy(k_ax, gt_psd,   "k-",  lw=2.5, label="CERRA (GT)", zorder=5)
            ax_psd.semilogy(k_ax, pred_psd, "b-",  lw=1.8, label="Prediction")
            ax_psd.semilogy(k_ax, lr_psd,   "g-", lw=1.5, label="ERA5 bilinear")
            ax_psd.axvline(cutoff, color="red", ls="--", lw=1.5,
                           label=f"HP cutoff ({cutoff:.3f})")
            ax_psd.set_xlabel("k  (cycles / HR pixel)")
            ax_psd.set_ylabel(f"PSD  [{comp}]²")
            ax_psd.set_title(f"Binned Isotropic PSD — {comp}")
            ax_psd.legend(fontsize=13, loc="upper right")

            if not use_avg:
                ax_diff = axes[ch, 1]
                ax_diff.semilogy(k_ax, np.abs(pred_psd - gt_psd) + 1e-20,
                                 "b-", lw=1.8, label="|Prediction − GT|")
                ax_diff.semilogy(k_ax, np.abs(lr_psd - gt_psd) + 1e-20,
                                 "g--", lw=1.5, label="|ERA5 bilinear − GT|")
                ax_diff.axvline(cutoff, color="red", ls="--", lw=1.5)
                ax_diff.set_xlabel("k  (cycles / HR pixel)")
                ax_diff.set_ylabel("|ΔPsd|  (pred − GT)")
                ax_diff.set_title(f"|PSD difference| — {comp}")
                ax_diff.legend(fontsize=13, loc="upper right")

        plt.tight_layout()
        sp = os.path.join(results_dir, save_name)
        plt.savefig(sp, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {sp}")

    _binned_figure(use_avg=False, save_name="sample_fft_binned.pdf")
    _binned_figure(use_avg=True,  save_name="average_fft_binned.pdf")

    # Average the 2-D power in linear scale over the samples, then take log10
    extent = [float(freq_w_grid.min()), float(freq_w_grid.max()),
              float(freq_h_grid.min()), float(freq_h_grid.max())]
    theta  = np.linspace(0, 2 * np.pi, 360)
    cx, cy = cutoff * np.cos(theta), cutoff * np.sin(theta)

    n_cols  = 3
    fig_2d, axes_2d = plt.subplots(
        2, n_cols, figsize=(4.2 * n_cols, 5.5 * 2),
        squeeze=False, constrained_layout=True)
    fig_2d.suptitle(
        f"2D log10-Power Spectrum — {case}\n"
        f"Averaged over {n_use} samples  |  "
        f"Cyan circle = HP cutoff ({cutoff:.3f} cy/px)",
        fontsize=18)

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
            ("CERRA (GT)", log_gt),
            ("Prediction", log_pred),
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

    sp2d = os.path.join(results_dir, "sample_fft_2d.pdf")
    plt.savefig(sp2d, dpi=300, bbox_inches="tight")
    plt.close(fig_2d)
    print(f"  Saved → {sp2d}  [averaged over {n_use} samples]")
    print("[plot_fft_analysis] All figures and CSV files saved.")


def run_case(CASE, run_start_str, timer, total_start):
    """Run one training-set case end to end.

    Steps: optional raw data conversion or memmap checks, U-Net build with weight loading,
    datasets, optional training, test evaluation, spatial metric maps, optional post-processing
    outputs, timing CSV, then release of the GPU memory.

    Args:
        CASE: key of TRAIN_DURATION.
        run_start_str: start time of the run (ISO string) written in the timing CSV.
        timer: SectionTimer collecting preprocessing, training and sampling times.
        total_start: time.perf_counter value at the start of the run.
    """
    print(f"\n{'='*70}")
    print(f"  Starting case: {CASE}")
    print(f"{'='*70}\n")
    RESULTS_DIR = os.path.join(
        BASE_DIR,
        f"Results_ML/UNet_SSIM_Maps/{CASE}" if not CONVERTING_DATA else "Results_ML/UNet_Debug"
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    lat_ERA5_path  = os.path.join(RESULTS_DIR, "lat_LR.npy")
    lon_ERA5_path  = os.path.join(RESULTS_DIR, "lon_LR.npy")
    lat_CERRA_path = os.path.join(RESULTS_DIR, "lat_HR.npy")
    lon_CERRA_path = os.path.join(RESULTS_DIR, "lon_HR.npy")

    with tf.device('/device:GPU:0'):

        if PLOT_ONLY:
            pred_path = os.path.join(RESULTS_DIR, f"pred_arrays_{CASE}.npz")
            if not os.path.exists(pred_path):
                raise FileNotFoundError(
                    f"PLOT_ONLY=True but no saved arrays found at {pred_path}.\n"
                    f"Run with PLOT_ONLY=False first.")
            print(f"[PLOT_ONLY] Loading saved arrays from {pred_path}")
            arrs = np.load(pred_path)
            plot_fft_analysis(
                results_dir=RESULTS_DIR,
                hr_pred=arrs["hr_pred_unscaled"],
                hr_true=(arrs["hr_true_unscaled"] if "hr_true_unscaled" in arrs.files else arrs["hr_batch_unscaled"]),
                lr_input=(arrs["lr_unscaled"] if "lr_unscaled" in arrs.files else arrs["lr_batch_unscaled"]),
                cutoff=FFT_CUTOFF, n_bins=128, n_samples_avg=2920, case=CASE)
            return

        if CONVERTING_DATA:
            timer.start("preprocessing")
            print("Starting data extraction from ERA5 GRIB files...")
            try:
                ds_wind_ERA5  = load_era5_variables(
                    grib_files_ERA5,  wind_vars, file_type="cfgrib")
                ds_wind_CERRA = load_era5_variables(
                    grib_files_CERRA, wind_vars, file_type="h5netcdf")
            except OSError as e:
                print(f"Error loading files: {e}"); raise

            ds_wind_ERA5  = time_crop(
                ds_wind_ERA5,  TIME_RANGE[0], TIME_RANGE[1], time_var="time")
            ds_wind_CERRA = time_crop(
                ds_wind_CERRA, TIME_RANGE[0], TIME_RANGE[1], time_var="time")
            ds_wind_ERA5  = domain_crop(
                ds_wind_ERA5,
                LAT_COORDS[0], LAT_COORDS[1],
                LON_COORDS[0], LON_COORDS[1],
                variable_name="long")
            ds_wind_CERRA = domain_crop(
                ds_wind_CERRA,
                LAT_COORDS_CERRA[0], LAT_COORDS_CERRA[1],
                LON_COORDS_CERRA[0], LON_COORDS_CERRA[1],
                variable_name="short")

            ds_wind_ERA5  = ds_wind_ERA5.isel(
                time=ds_wind_ERA5["time"].dt.hour.isin(
                    [0, 3, 6, 9, 12, 15, 18, 21]))
            ds_wind_CERRA = wind_speed_to_uv(ds_wind_CERRA)
            ds_wind_CERRA = ds_wind_CERRA.sel(lat=slice(None, None, -1))

            image_path = os.path.join(RESULTS_DIR, "u100_CERRA_vs_ERA5.pdf")
            plot_comparison_HR_LR(
                ds_wind_CERRA, ds_wind_ERA5, var="u100",
                titles=("CERRA u100", "ERA5 u100"),
                extent=[LON_COORDS[0], LON_COORDS[1],
                        LAT_COORDS[0], LAT_COORDS[1]],
                savepath=image_path)

            u_era5  = ds_wind_ERA5["u100"].values.astype(np.float32)
            v_era5  = ds_wind_ERA5["v100"].values.astype(np.float32)
            u_cerra = ds_wind_CERRA["u100"].values.astype(np.float32)
            v_cerra = ds_wind_CERRA["v100"].values.astype(np.float32)

            arrays_np_lr  = np.stack([u_era5, v_era5],   axis=-1)
            arrays_np_hr  = np.stack([u_cerra, v_cerra], axis=-1)
            arrays_seq_lr = sequence_encoding(
                arrays_np_lr, N_BEFORE, N_AFTER, keep_channels=True)
            arrays_np_hr  = HR_sequence_slicing(arrays_np_hr, N_BEFORE, N_AFTER)

            if not UV_COMP:
                arrays_seq_lr = arrays_seq_lr[..., ::2]
                arrays_np_hr  = arrays_np_hr[...,  0:1]

            lat_HR = ds_wind_CERRA["lat"].values
            lon_HR = ds_wind_CERRA["lon"].values
            lat_lr = ds_wind_ERA5["latitude"].values
            lon_lr = ds_wind_ERA5["longitude"].values
            np.savetxt(lat_ERA5_path,  lat_lr)
            np.savetxt(lon_ERA5_path,  lon_lr)
            np.savetxt(lat_CERRA_path, lat_HR)
            np.savetxt(lon_CERRA_path, lon_HR)
            timer.stop("preprocessing")

            lr_mmap_path = os.path.join(RESULTS_DIR, "arrays_np_lr.dat")
            hr_mmap_path = os.path.join(RESULTS_DIR, "arrays_np_hr.dat")
            lr_mmap    = np.memmap(lr_mmap_path, dtype="float32", mode="w+",
                                   shape=arrays_seq_lr.shape)
            lr_mmap[:] = arrays_seq_lr[:]; lr_mmap.flush()
            np.save(os.path.join(RESULTS_DIR, "lr_shape.npy"), arrays_seq_lr.shape)
            hr_mmap    = np.memmap(hr_mmap_path, dtype="float32", mode="w+",
                                   shape=arrays_np_hr.shape)
            hr_mmap[:] = arrays_np_hr[:]; hr_mmap.flush()
            np.save(os.path.join(RESULTS_DIR, "hr_shape.npy"), arrays_np_hr.shape)
            lr_shape = arrays_seq_lr.shape
            hr_shape = arrays_np_hr.shape

            del arrays_np_lr, arrays_seq_lr, arrays_np_hr
            del ds_wind_ERA5, ds_wind_CERRA
            gc.collect()
            tf.keras.backend.clear_session()

        else:
            timer.start("preprocessing")
            case_dir = os.path.join(OUTPUT_ROOT, f"Data_{CASE}")
            for required in [
                os.path.join(case_dir,   "train", "ERA5_train.npy"),
                os.path.join(case_dir,   "train", "CERRA_train.npy"),
                os.path.join(SHARED_DIR, "val",   "ERA5_val.npy"),
                os.path.join(SHARED_DIR, "val",   "CERRA_val.npy"),
                os.path.join(SHARED_DIR, "test",  "ERA5_test.npy"),
                os.path.join(SHARED_DIR, "test",  "CERRA_test.npy"),
            ]:
                if not os.path.exists(required):
                    raise FileNotFoundError(
                        f"Prepared memmap not found: {required}")

            coords_dir = os.path.join(SHARED_DIR, "coords")
            lat_HR = np.loadtxt(os.path.join(coords_dir, "lat_HR.npy"))
            lon_HR = np.loadtxt(os.path.join(coords_dir, "lon_HR.npy"))
            lat_lr = np.loadtxt(os.path.join(coords_dir, "lat_LR.npy"))
            lon_lr = np.loadtxt(os.path.join(coords_dir, "lon_LR.npy"))
            timer.stop("preprocessing")

        if CONVERTING_DATA:
            lat_HR = np.loadtxt(lat_CERRA_path)
            lon_HR = np.loadtxt(lon_CERRA_path)
            lat_lr = np.loadtxt(lat_ERA5_path)
            lon_lr = np.loadtxt(lon_ERA5_path)

        # Build U-Net model + smart weight loading
        import h5py
        def _load_unet_weights(weights_path: str) -> tf.keras.Model:
            """Build a plain U-Net and load a checkpoint saved either from it or from the spectral-loss wrapper.

            Args:
                weights_path: .weights.h5 file.
            Returns:
                keras.Model with the loaded weights.
            """
            bare_unet = get_post_network(
                image_size_height = PX_LAT_HR,
                input_frames      = INPUT_FRAMES,
                output_frames     = OUTPUT_FRAMES,
                widths            = widths,
                block_depth       = block_depth,
                image_size_width  = PX_LON_HR,
            )
            dummy = tf.zeros([1, PX_LAT_HR, PX_LON_HR, INPUT_FRAMES], dtype=tf.float32)
            bare_unet(dummy, training=False)

            with h5py.File(weights_path, "r") as f:
                top_keys   = list(f.keys())
                layer_keys = list(f["layers"].keys()) if "layers" in f else []
                # Sub-models have a nested 'layers/' group; plain layers only have 'vars/'
                needs_shell = any(
                    isinstance(f[f"layers/{k}"], h5py.Group)
                    and "layers" in f[f"layers/{k}"]
                    for k in layer_keys
                ) if layer_keys else False

            print(f"  Checkpoint top-level keys : {top_keys}")
            print(f"  Checkpoint layer-level keys: {layer_keys}")
            print(f"  Wrapped checkpoint detected: {needs_shell}")

            if needs_shell:
                print("  → Loading via _SpectralShell")
                # Keras 3 keys sub-models by their .name property ("functional"),
                # not by the Python attribute name ("unet"). Both UNetSpectral and
                # _SpectralShell store the same Functional model, so the names match.
                class _SpectralShell(tf.keras.Model):
                    """Wrapper reproducing the layer layout of the spectral-loss model, used to read its checkpoints."""

                    def __init__(self, inner):
                        """Store the wrapped U-Net."""
                        super().__init__()
                        self.unet = inner
                    def call(self, x, training=False):
                        """Apply the wrapped U-Net."""
                        return self.unet(x, training=training)

                shell = _SpectralShell(bare_unet)
                shell(dummy, training=False)
                shell.load_weights(weights_path)
                print("  Weights loaded successfully.")
                return shell.unet
            else:
                print("  → Loading directly into bare UNet")
                bare_unet.load_weights(weights_path)
                print("  Weights loaded successfully.")
                return bare_unet

        PRETRAINED_BASE = os.path.join(BASE_DIR, "Results_ML/UNet")
        weights_to_load = os.path.join(PRETRAINED_BASE, CASE, "UNet_best_weights.weights.h5")


        if not os.path.exists(weights_to_load):
            raise FileNotFoundError(
                f"Weights file not found:\n  {weights_to_load}")

        print(f"Loading weights from:\n  {weights_to_load}")
        unet_model = _load_unet_weights(weights_to_load)

        unet_model.compile(
            optimizer   = keras.optimizers.AdamW(
                learning_rate=learning_rate, weight_decay=weight_decay),
            loss        = tf.keras.losses.MeanSquaredError(),
            metrics     = ["mse"],
            jit_compile = True,
        )
        unet_model.summary()
        model_weights_path = os.path.join(RESULTS_DIR, "UNet_best_weights.weights.h5")

        # Build tf.data pipeline
        if CONVERTING_DATA:
            lr_shape = tuple(np.load(os.path.join(RESULTS_DIR, "lr_shape.npy")))
            hr_shape = tuple(np.load(os.path.join(RESULTS_DIR, "hr_shape.npy")))
            datasets, minmax_stats = build_unet_pipeline(
                os.path.join(RESULTS_DIR, "arrays_np_lr.dat"),
                os.path.join(RESULTS_DIR, "arrays_np_hr.dat"),
                lr_shape, hr_shape,
                batch_size=BATCH_SIZE, shuffle_buffer=3000)
        else:
            case_dir = os.path.join(OUTPUT_ROOT, f"Data_{CASE}")

            datasets, minmax_stats = build_unet_pipeline_from_prepared(
                case_dir   = case_dir,
                shared_dir = SHARED_DIR,
                csv_path   = os.path.join(case_dir, f"statistics_{CASE}.csv"),
                batch_size = BATCH_SIZE,
                shuffle_buffer = 3000)

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

        n_train = datasets["n_train"]
        n_val   = datasets["n_val"]

        # Training
        best_val_loss = float("inf")
        timer.start("training")

        if TRAINING:
            steps_per_epoch  = int(TRAIN_DURATION.get(CASE, n_train) / BATCH_SIZE)
            validation_steps = max(1, n_val // BATCH_SIZE - 1)
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    filepath          = model_weights_path,
                    monitor           = "val_loss",
                    save_best_only    = True,
                    save_weights_only = True,
                    verbose           = 1),
                keras.callbacks.EarlyStopping(
                    monitor              = "val_loss",
                    patience             = EARLY_STOPPING_PATIENCE,
                    min_delta            = 1e-7,
                    mode                 = "min",
                    restore_best_weights = True),
            ]
            print(f"Start training — steps_per_epoch={steps_per_epoch}, "
                  f"validation_steps={validation_steps}, max_epochs={EPOCHS}")
            history = unet_model.fit(
                datasets["train_ds"],
                validation_data  = datasets["val_ds"],
                epochs           = EPOCHS,
                steps_per_epoch  = steps_per_epoch,
                validation_steps = validation_steps,
                verbose          = 1,
                callbacks        = callbacks)
            best_val_loss = min(history.history["val_loss"])

            best_epoch = int(np.argmin(history.history["val_loss"]))
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(history.history["loss"],     color="steelblue",
                    linewidth=1.5, label="train MSE loss")
            ax.plot(history.history["val_loss"], color="tomato",
                    linewidth=1.5, linestyle="--", label="val MSE loss")
            ax.axvline(best_epoch, color="black", linewidth=0.9,
                       linestyle=":",
                       label=f"best epoch {best_epoch + 1}  "
                       f"(val={history.history['val_loss'][best_epoch]:.6f})")
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("MSE loss", fontsize=12)
            ax.legend(fontsize=10, framealpha=0.7)
            ax.set_title(f"UNet training curves — {CASE}", fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, "loss_history.pdf"),
                        dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Best val_loss = {best_val_loss:.6f}")

        else:
            history = None

        timer.stop("training")

        # Evaluate on the test set

        print(f"\nComputing metrics on test dataset [{CASE}]:")

        print("XLA + vectorized evaluation enabled")

        @tf.function(jit_compile=True)
        def compute_metrics(y_true, y_pred):
            """Compute MSE, PSNR and SSIM of a batch scaled to [0, 1].

            Args:
                y_true, y_pred: tensors (B, H, W, 2).
            Returns:
                (mse, psnr, ssim, mse_per_sample, psnr_per_sample, ssim_per_sample).
            """
            error           = y_pred - y_true
            mse_per_sample  = tf.reduce_mean(tf.square(error), axis=[1, 2, 3])
            mse             = tf.reduce_mean(mse_per_sample)
            psnr_per_sample = 10.0 * tf.math.log(1.0 / (mse_per_sample + 1e-10)) / tf.math.log(10.0)
            psnr            = tf.reduce_mean(psnr_per_sample)
            ssim_u          = tf.image.ssim(y_true[..., 0:1], y_pred[..., 0:1], max_val=1.0, filter_size=7)
            ssim_v          = tf.image.ssim(y_true[..., 1:2], y_pred[..., 1:2], max_val=1.0, filter_size=7)
            ssim            = tf.reduce_mean((ssim_u + ssim_v) * 0.5)
            ssim_per_sample = (ssim_u + ssim_v) * 0.5
            return mse, psnr, ssim, mse_per_sample, psnr_per_sample, ssim_per_sample

        mse_all, psnr_all, ssim_all     = [], [], []
        mse_per_sample_all              = []
        psnr_per_sample_all             = []
        ssim_per_sample_all             = []
        hr_pred_list, hr_true_list, lr_input_list = [], [], []

        for lr_batch, hr_batch in datasets["test_ds"]:
            hr_pred  = tf.cast(unet_model(lr_batch, training=False), tf.float32)
            y_pred   = hr_pred
            y_true   = tf.cast(hr_batch, tf.float32)
            mse, psnr, ssim, mse_s, psnr_s, ssim_s = compute_metrics(y_true, y_pred)
            mse_all.append(mse);  psnr_all.append(psnr);  ssim_all.append(ssim)
            mse_per_sample_all.append(mse_s)
            psnr_per_sample_all.append(psnr_s)
            ssim_per_sample_all.append(ssim_s)
            # Move each batch to CPU memory so GPU memory does not accumulate
            hr_pred_list.append(y_pred.numpy())
            hr_true_list.append(y_true.numpy())
            lr_input_list.append(lr_batch.numpy())

        mse_mean       = tf.reduce_mean(mse_all).numpy()
        psnr_mean      = tf.reduce_mean(psnr_all).numpy()
        ssim_mean_scalar = tf.reduce_mean(ssim_all).numpy()
        mse_per_sample  = tf.concat(mse_per_sample_all,  axis=0).numpy()
        psnr_per_sample = tf.concat(psnr_per_sample_all, axis=0).numpy()
        ssim_per_sample = tf.concat(ssim_per_sample_all, axis=0).numpy()

        print(f"\n--- TEST METRICS (scaled [0,1]) ---  [{CASE}]")
        print(f"  Mean MSE  : {mse_mean:.6f}")
        print(f"  Mean PSNR : {psnr_mean:.3f} dB")
        print(f"  Mean SSIM : {ssim_mean_scalar:.4f}")

        # Concatenate on CPU - stays in RAM, no GPU allocation
        hr_pred_all  = np.concatenate(hr_pred_list,  axis=0)
        hr_true_all  = np.concatenate(hr_true_list,  axis=0)
        lr_input_all = np.concatenate(lr_input_list, axis=0)

        # Unscale the batches with broadcast-shaped min/max arrays
        hr_pred_unscaled  = hr_pred_all  * (hr_maxs_bc - hr_mins_bc) + hr_mins_bc
        hr_batch_unscaled = hr_true_all  * (hr_maxs_bc - hr_mins_bc) + hr_mins_bc
        lr_batch_unscaled = lr_input_all * (lr_maxs_bc - lr_mins_bc) + lr_mins_bc
        if PRINT_UNSCALED_METRICS:
            mse_unscaled     = float(np.mean((hr_pred_unscaled - hr_batch_unscaled) ** 2))
            max_val_unscaled = float(hr_batch_unscaled.max())
            psnr_unscaled    = 10.0 * np.log10(max_val_unscaled ** 2 / (mse_unscaled + 1e-12))
            n_ssim           = min(200, hr_batch_unscaled.shape[0])
            ssim_unscaled    = float(tf.reduce_mean(tf.image.ssim(
                tf.constant(hr_batch_unscaled[:n_ssim]),
                tf.constant(hr_pred_unscaled[:n_ssim]),
                max_val=max_val_unscaled, filter_size=7,
            )).numpy())

            print(f"\n--- UN-SCALED METRICS (SSIM over first {n_ssim} samples) ---")
            print(f"MSE  (unscaled): {mse_unscaled:.6f}")
            print(f"PSNR (unscaled): {psnr_unscaled:.3f} dB")
            print(f"SSIM (unscaled): {ssim_unscaled:.4f}")

        if SAVE_POSTPROCESSING_OUTPUTS:
            pred_save_path = os.path.join(RESULTS_DIR, f"pred_arrays_{CASE}.npz")
            n_save = 2
            pred_save_path = os.path.join(RESULTS_DIR, f"pred_arrays_{CASE}.npz")
            np.savez_compressed(pred_save_path,
                                hr_pred_unscaled = hr_pred_unscaled[:n_save],
                                hr_true_unscaled = hr_batch_unscaled[:n_save],
                                lr_unscaled      = lr_batch_unscaled[:n_save])
            print(f"Saved prediction arrays → {pred_save_path}  (first {n_save} samples)")

            def _np(x):
                """Convert a tensor to numpy."""
                return x.numpy() if isinstance(x, tf.Tensor) else np.asarray(x)

            # Placeholder SSIM maps required by plot_comparison_transformer
            dummy_ssim = np.zeros((hr_batch_unscaled.shape[1], hr_batch_unscaled.shape[2]),
                                  dtype=np.float32)
            plot_comparison_transformer(
                _np(lr_batch_unscaled), _np(hr_pred_unscaled), _np(hr_batch_unscaled),
                ssim_mean_u=dummy_ssim, ssim_std_u=dummy_ssim,
                ssim_mean_v=dummy_ssim, ssim_std_v=dummy_ssim,
                lat_hr=lat_HR, lon_hr=lon_HR, lat_lr=lat_lr, lon_lr=lon_lr,
                index=0, results_dir=RESULTS_DIR,
            )

            plot_temporal_metrics(
                mse_per_sample  = mse_per_sample.astype(np.float32),
                psnr_per_sample = psnr_per_sample.astype(np.float32),
                ssim_per_sample = ssim_per_sample.astype(np.float32),
                n_train         = n_train,
                n_val           = n_val,
                results_dir     = RESULTS_DIR,
                uv_comp         = UV_COMP,
                rolling_window  = 8,
                data_is_ordered = not CONVERTING_DATA,
            )

            plot_fft_analysis(
                results_dir   = RESULTS_DIR,
                hr_pred       = _np(hr_pred_unscaled),
                hr_true       = _np(hr_batch_unscaled),
                lr_input      = _np(lr_batch_unscaled),
                cutoff        = FFT_CUTOFF,
                n_bins        = 128,
                n_samples_avg = 2920,
                case          = CASE,
            )

            plot_comparison_wind_speed(
                lr          = _np(lr_batch_unscaled),
                pred        = _np(hr_pred_unscaled),
                gt          = _np(hr_batch_unscaled),
                lat_hr=lat_HR, lon_hr=lon_HR, lat_lr=lat_lr, lon_lr=lon_lr,
                index=0, results_dir=RESULTS_DIR, case=CASE,
            )

            plot_quiver_wind(
                pred          = _np(hr_pred_unscaled),
                gt            = _np(hr_batch_unscaled),
                lat_hr=lat_HR, lon_hr=lon_HR,
                index=0, results_dir=RESULTS_DIR, quiver_stride=12, case=CASE,
            )

            # Append the test metrics to test_metrics.csv
            metrics_path = os.path.join(RESULTS_DIR, "test_metrics.csv")
            if not os.path.exists(metrics_path):
                with open(metrics_path, "w", newline="") as f:
                    csv.writer(f).writerow([
                        "run_time", "case", "val_loss_mse_scaled",
                        "test_mse_scaled", "test_psnr", "test_ssim",
                        "input_frames", "output_frames", "epochs"])
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    datetime.now().isoformat(),
                    CASE,
                    round(best_val_loss, 6),
                    round(mse_mean, 6),
                    round(psnr_mean, 6),
                    round(ssim_mean_scalar, 4),
                    INPUT_FRAMES,
                    OUTPUT_FRAMES,
                    EPOCHS if TRAINING else 0])
            print(f"Saved test metrics → {metrics_path}")

        timer.start("sampling")

        timer.stop("sampling")

        plot_spatial_metric_maps(
            hr_pred_scaled   = hr_pred_all,
            hr_true_scaled   = hr_true_all,
            hr_true_unscaled = hr_batch_unscaled,
            lat_hr=lat_HR, lon_hr=lon_HR,
            results_dir=RESULTS_DIR, case=CASE,
            win_size=11, uv_comp=UV_COMP, save_npy=True, chunk_size=64,
        )
        # Append the run times to computation_time_runs.csv
        total_time  = time.perf_counter() - total_start
        csv_file    = os.path.join(RESULTS_DIR, "computation_time_runs.csv")
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["run_start_time", "case",
                                 "preprocessing_s", "training_s",
                                 "sampling_s", "total_s"])
            writer.writerow([
                run_start_str,
                CASE,
                f"{timer.times.get('preprocessing', 0.0):.4f}",
                f"{timer.times.get('training',      0.0):.4f}",
                f"{timer.times.get('sampling',      0.0):.4f}",
                f"{total_time:.4f}"])
        print(f"[{CASE}] Total elapsed time so far: {total_time:.1f} s")

    del unet_model, datasets, minmax_stats
    gc.collect()
    tf.keras.backend.clear_session()
    print(f"[{CASE}] GPU memory released. Ready for next case.\n")


def main():
    """Check CASES and run every case in sequence."""
    if not CASES:
        raise ValueError("CASES is empty — add at least one case to run.")
    valid_cases = set(TRAIN_DURATION.keys())
    for c in CASES:
        if c not in valid_cases:
            raise ValueError(
                f"Unknown case '{c}'. Valid options: {sorted(valid_cases)}")

    run_start     = datetime.now()
    run_start_str = run_start.isoformat(timespec="seconds")
    timer         = SectionTimer()
    total_start   = time.perf_counter()

    print(f"Multi-case UNet run — {len(CASES)} case(s): {CASES}")
    print(f"TRAINING     = {TRAINING}")
    print(f"PLOT_ONLY    = {PLOT_ONLY}")
    if not TRAINING and PRETRAINED_WEIGHTS_PATH is not None:
        print(f"Loading weights from: {PRETRAINED_WEIGHTS_PATH}")
    print(f"Started at: {run_start_str}\n")

    for i, CASE in enumerate(CASES, start=1):
        print(f"\n[{i}/{len(CASES)}] Running case: {CASE}")
        run_case(CASE, run_start_str, timer, total_start)

    total_elapsed = time.perf_counter() - total_start
    print(f"\nAll cases completed: {CASES}")
    print(f"Total wall-clock time: {total_elapsed:.1f} s")
    print("Done.")


if __name__ == "__main__":
    main()
