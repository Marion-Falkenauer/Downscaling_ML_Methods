"""Sweep of the spectral-loss weight lambda_fft for the U-Net downscaling model.

Loss: L = MSE(pred, target) + lambda_fft * sum over wavenumbers k > cutoff of
(log|FFT_w(pred)_k| - log|FFT_w(target)_k|)^2, where FFT_w is the FFT of the field multiplied
by a Hann window (to limit leakage) and only the high-frequency part is kept by a high-pass
mask. Phases are ignored, the MSE term drives the global reconstruction.

The network is get_post_network (U-Net of Merizzi et al.,
https://github.com/fmerizzi/ERA5-to-CERRA-via-Diffusion-Models) applied to the LR input
bilinearly upsampled to the HR grid. One model is trained per lambda_fft value between LAMB_MIN
and LAMB_MAX; data are streamed per batch from the memmaps of prepare_training_data.py.

Outputs, in Results_ML/UNet_Final_Loss/<case>: one folder per lambda_fft value (weights, loss
history, comparison figures), sweep_results.csv, sweep summary figure, spectral figures
(plot_fft_analysis) and computation_time_runs.csv.

Configuration (module constants):
    CASE: training-set size, key of TRAIN_DURATION.
    CONVERTING_DATA: True converts the raw GRIB/NetCDF files, False reads the prepared memmaps.
    PLOT_ONLY: True only draws the spectral figures from the saved weights.
    LAMB_MIN, LAMB_MAX, N_STEPS: range and number of lambda_fft values.
    EPOCHS, BATCH_SIZE, learning_rate, weight_decay, EARLY_STOPPING_PATIENCE: optimisation settings.
    FFT_CUTOFF: high-pass cutoff in cycles per pixel.
    UNET_WEIGHT_FILENAME: name of the weight file in each lambda_fft folder.

Author: Marion Falkenauer (with the help of Claude)
"""

import xarray as xr
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2" # To choose the GPU unit, to check in the Linux terminal which one is available
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import csv
import gc
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import time

from scipy.fft    import fft2 as scipy_fft2, fftshift, fftfreq as scipy_fftfreq
from scipy.fft    import fftfreq as scipy_fftfreq_full, rfftfreq as scipy_rfftfreq
from scipy.signal import windows as scipy_windows

from ML_Methods_ERA5_CERRA import plot_comparison_transformer

from setup import widths, block_depth
from denoising_unet_edit import get_post_network

plt.rcParams.update({
    "font.size":       25,
    "axes.labelsize":  20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 20,
    "axes.titlesize":  20,
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ERA5_files      = os.path.join(BASE_DIR, "Data/ERA5_Single_Year")
grib_files_ERA5 = sorted(glob.glob(os.path.join(ERA5_files, "*.grib")))
CERRA_files      = os.path.join(BASE_DIR, "Data/CERRA_Single_Year")
grib_files_CERRA = sorted(glob.glob(os.path.join(CERRA_files, "*.nc")))

PX_LAT_HR     = 320
PX_LON_HR     = 400
PX_LAT_LR     = 64
PX_LON_LR     = 80
RF              = 5

CONVERTING_DATA = False
PLOT_ONLY       = False

N_BEFORE = 0
N_AFTER  = 0

INPUT_FRAMES  = 2
OUTPUT_FRAMES = 2


CASE = "2year"

OUTPUT_ROOT = BASE_DIR
SHARED_DIR  = os.path.join(OUTPUT_ROOT, "Data_Shared")

TRAIN_DURATION = {"1year":2920,"2year":5840,"4year": 8768}

LON_COORDS,       LAT_COORDS       = [-5, 14.75], [44, 59.75]
LON_COORDS_CERRA, LAT_COORDS_CERRA = [-5, 14.95], [44, 59.95]
TIME_RANGE = ["2024-01-01", "2024-12-31"]


BATCH_SIZE              = 2
learning_rate           = 1e-5
weight_decay            = 1e-6
EARLY_STOPPING_PATIENCE = 10


LAMB_MIN = 7.5e-09
LAMB_MAX = 7.5e-09
N_STEPS= 1
EPOCHS= 50

LAMBDA_VALUES = list(np.linspace(LAMB_MIN, LAMB_MAX, N_STEPS))

HR_H = 320
HR_W = 400
LR_H= 64
LR_W = 80

FFT_CUTOFF=0.05

RESULTS_DIR=os.path.join(
    BASE_DIR,
    f"Results_ML/UNet_Final_Loss/{CASE}"
    if not CONVERTING_DATA
    else "Results_ML/UNet_loss_functions_debug"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

UNET_WEIGHT_FILENAME= "UNet_weights_Checkpoint.weights.h5"

wind_vars =["100u", "100v"]

lat_ERA5_path= os.path.join(RESULTS_DIR, "lat_LR.npy")
lon_ERA5_path= os.path.join(RESULTS_DIR, "lon_LR.npy")
lat_CERRA_path= os.path.join(RESULTS_DIR, "lat_HR.npy")
lon_CERRA_path= os.path.join(RESULTS_DIR, "lon_HR.npy")


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
        self.times = {}

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


run_start     = datetime.now()
run_start_str = run_start.isoformat(timespec="seconds")
timer         = SectionTimer()
total_start   = time.perf_counter()


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"TF using {len(gpus)} GPU(s): {[g.name for g in gpus]}")
else:
    print("No GPU detected. TF using CPU.")
print("TF logical devices:", tf.config.list_logical_devices())


class UNetSpectral(tf.keras.Model):
    """U-Net trained with MSE plus a Hann-windowed high-frequency log-spectral loss.

    Windowing, rfft2d, squared difference of the log amplitudes and a high-pass mask are all
    XLA-compatible; train_step and test_step are compiled with jit_compile.
    """

    _HR_H: int = 320
    _HR_W: int = 400

    def __init__(
        self,
        unet_model,
        lambda_fft:  float = 0.0,
        eps:         float = 1e-12,
        cutoff_freq: float = FFT_CUTOFF,
        **kwargs,
    ):
        """Store the U-Net, the loss settings, the Hann window and the high-pass mask.

        Args:
            unet_model: keras.Model mapping (B, H, W, C_in) to (B, H, W, C_out).
            lambda_fft: weight of the spectral term (0 gives pure MSE).
            eps: constant added inside the logarithm.
            cutoff_freq: high-pass cutoff in cycles per pixel.
        """
        super().__init__(**kwargs)
        self.unet        = unet_model
        self.lambda_fft  = float(lambda_fft)
        self.eps         = float(eps)
        self.cutoff_freq = float(cutoff_freq)

        self._hann_window   = UNetSpectral._build_hann_window(
            self._HR_H, self._HR_W)
        self._highpass_mask = UNetSpectral._build_highpass_mask(
            self._HR_H, self._HR_W, self.cutoff_freq)

        self.loss_tracker     = keras.metrics.Mean(name="loss")
        self.val_loss_tracker = keras.metrics.Mean(name="val_loss")

    @staticmethod
    def _build_hann_window(H: int, W: int) -> tf.Tensor:
        """Separable 2-D Hann window normalised by its mean.

        Args:
            H, W: field size.
        Returns:
            Constant tensor (1, H, W, 1).
        """
        w_h   = scipy_windows.hann(H, sym=True).astype(np.float32)
        w_w   = scipy_windows.hann(W, sym=True).astype(np.float32)
        win2d = np.outer(w_h, w_w)
        win2d = win2d / win2d.mean()
        return tf.constant(win2d[np.newaxis, :, :, np.newaxis], dtype=tf.float32)

    @staticmethod
    def _build_highpass_mask(H: int, W: int, cutoff: float) -> tf.Tensor:
        """High-pass radial mask for the output of rfft2d.

        Args:
            H, W: field size.
            cutoff: cutoff in cycles per pixel.
        Returns:
            Constant tensor (1, 1, H, W // 2 + 1), 1 where the wavenumber exceeds the cutoff.
        """
        freq_h = scipy_fftfreq_full(H).astype(np.float32)
        freq_w = scipy_rfftfreq(W).astype(np.float32)
        FH, FW = np.meshgrid(freq_h, freq_w, indexing='ij')
        mask   = (np.sqrt(FH**2 + FW**2) > cutoff).astype(np.float32)
        return tf.constant(mask[np.newaxis, np.newaxis, :, :], dtype=tf.float32)

    def call(self, x, training=False):
        """Apply the U-Net."""
        return self.unet(x, training=training)

    @property
    def metrics(self):
        """Metrics reset by Keras at each epoch: the validation loss tracker."""
        return [self.val_loss_tracker]

    def _fft_loss(self, pred: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        """High-pass log-amplitude spectral loss.

        Args:
            pred, target: tensors (B, H, W, C).
        Returns:
            Scalar tensor, sum over the high-frequency bins of the squared difference of log amplitudes.
        """
        # Hann window in spatial domain
        pred_w   = pred   * self._hann_window
        target_w = target * self._hann_window

        # Move channels first (BHWC to BCHW) for rfft2d
        pred_t   = tf.transpose(pred_w,   [0, 3, 1, 2])
        target_t = tf.transpose(target_w, [0, 3, 1, 2])

        # 2D real FFT, output shape [B, C, H, W//2+1]
        F_pred   = tf.signal.rfft2d(pred_t)
        F_target = tf.signal.rfft2d(target_t)

        # Log-amplitude squared difference
        log_amp_pred   = tf.math.log(tf.abs(F_pred)   + self.eps)
        log_amp_target = tf.math.log(tf.abs(F_target) + self.eps)
        diff_sq        = tf.square(log_amp_pred - log_amp_target)

        return tf.reduce_sum(diff_sq * self._highpass_mask)

    def _spectral_loss(self, pred: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        """MSE plus lambda_fft times the spectral term.

        Args:
            pred, target: tensors (B, H, W, C).
        Returns:
            Scalar tensor.
        """
        loss_mse = tf.reduce_mean(tf.square(pred - target))
        if self.lambda_fft > 0.0:
            return loss_mse + self.lambda_fft * self._fft_loss(pred, target)
        return loss_mse

    @tf.function(jit_compile=True)
    def train_step(self, data):
        """One training step with gradient clipping at norm 1.

        Args:
            data: batch (x, y).
        Returns:
            Dictionary with the running training loss.
        """
        x, y = data
        with tf.GradientTape() as tape:
            pred = self(x, training=True)
            loss = self._spectral_loss(pred, y)
        grads = tape.gradient(loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @tf.function(jit_compile=True)
    def test_step(self, data):
        """One evaluation step.

        Args:
            data: batch (x, y).
        Returns:
            Dictionary with the running validation loss.
        """
        x, y = data
        pred = self(x, training=False)
        loss = self._spectral_loss(pred, y)
        self.val_loss_tracker.update_state(loss)
        return {"loss": self.val_loss_tracker.result()}

    def get_config(self):
        """Configuration of the model including the loss settings."""
        return {
            **super().get_config(),
            "lambda_fft":  self.lambda_fft,
            "eps":         self.eps,
            "cutoff_freq": self.cutoff_freq,
        }


def build_unet_spectral(
    lambda_fft:  float = 0.0,
    eps:         float = 1e-12,
    cutoff_freq: float = FFT_CUTOFF,
) -> UNetSpectral:
    """Build a UNetSpectral model around get_post_network, with initialised weights.

    Args:
        lambda_fft: weight of the spectral term.
        eps: constant added inside the logarithm.
        cutoff_freq: high-pass cutoff in cycles per pixel.
    Returns:
        UNetSpectral, not yet compiled. widths and block_depth come from setup.py.
    """
    bare_unet = get_post_network(
        image_size_height = PX_LAT_HR,
        input_frames      = INPUT_FRAMES,
        output_frames     = OUTPUT_FRAMES,
        widths            = widths,
        block_depth       = block_depth,
        image_size_width  = PX_LON_HR,
    )
    model = UNetSpectral(
        unet_model  = bare_unet,
        lambda_fft  = lambda_fft,
        eps         = eps,
        cutoff_freq = cutoff_freq,
    )
    dummy = tf.zeros([1, PX_LAT_HR, PX_LON_HR, INPUT_FRAMES], dtype=tf.float32)
    model(dummy, training=False)
    return model


def _plot_sweep_summary(csv_path: str, results_dir: str,
                        x_col: str, sweep_label: str):
    """Plot PSNR, SSIM and MSE against the swept coefficient from the sweep CSV.

    Saves sweep_summary_<x_col>.png in results_dir.

    Args:
        csv_path: sweep CSV file.
        results_dir: output directory.
        x_col: CSV column of the swept coefficient.
        sweep_label: label of the x axis.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[_plot_sweep_summary] Could not read {csv_path}: {e}")
        return
    if x_col not in df.columns:
        print(f"[_plot_sweep_summary] Column '{x_col}' not found in {csv_path}.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [
        ("test_psnr", "PSNR [dB]",   "#1f77b4"),
        ("test_ssim", "SSIM [-]",         "#2ca02c"),
        ("test_mse",  "MSE (scaled) [-]", "#d62728"),
    ]
    for ax, (col, ylabel, color) in zip(axes, metrics):
        if col not in df.columns:
            continue
        ax.plot(df[x_col], df[col], "o-", color=color, linewidth=1.5, markersize=6)
        ax.set_xlabel(sweep_label)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs {sweep_label}")
        ax.grid(True, alpha=0.4)

    fig.suptitle(f"Parametric study — {sweep_label}", fontsize=20)
    plt.tight_layout()
    out = os.path.join(results_dir, f"sweep_summary_{x_col}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[_plot_sweep_summary] Saved sweep summary -> {out}")


def run_unet_fft_sweep(
    datasets:         dict,
    minmax_stats:     dict,
    results_dir:      str,
    case:             str,
    n_steps:          int        = N_STEPS,
    lambda_min:       float      = LAMB_MIN,
    lambda_max:       float      = LAMB_MAX,
    epochs:           int        = EPOCHS,
    lr_rate:          float      = learning_rate,
    batch_size:       int        = BATCH_SIZE,
    validation_steps: int        = 2807,
    lat_hr:           np.ndarray = None,
    lon_hr:           np.ndarray = None,
    lat_lr:           np.ndarray = None,
    lon_lr:           np.ndarray = None,
    resume:           bool       = True,
    skip_if_done:     bool       = False,
):
    """Train one UNetSpectral model per lambda_fft value and evaluate it on the test set.

    Each value has its own folder with the weights, the loss history, the comparison figures and
    a row in the sweep CSV. A summary figure is saved at the end.

    Args:
        datasets: dictionary with train_ds, val_ds, test_ds, n_train, n_val.
        minmax_stats: dictionary {"lr": (mins, maxs), "hr": (mins, maxs)} used to unscale the data.
        results_dir: output directory.
        case: training-set case, sets the steps per epoch.
        n_steps: number of lambda_fft values.
        lambda_min, lambda_max: range of lambda_fft.
        epochs: maximum number of epochs per value.
        lr_rate: AdamW learning rate.
        batch_size: batch size used for the steps per epoch.
        validation_steps: validation steps per epoch.
        lat_hr, lon_hr, lat_lr, lon_lr: coordinate vectors of the HR and LR grids for the figures.
        resume: load an existing checkpoint of the value before training.
        skip_if_done: skip training when a checkpoint exists (evaluation only).
    """
    lambda_values   = list(np.linspace(lambda_min, lambda_max, n_steps))
    n_train         = datasets["n_train"]
    steps_per_epoch = int(TRAIN_DURATION.get(case, n_train) / batch_size)

    hr_mins_np = np.array([float(v) for v in minmax_stats["hr"][0]], dtype=np.float32)
    hr_maxs_np = np.array([float(v) for v in minmax_stats["hr"][1]], dtype=np.float32)
    lr_mins_np = np.array([float(v) for v in minmax_stats["lr"][0]], dtype=np.float32)
    lr_maxs_np = np.array([float(v) for v in minmax_stats["lr"][1]], dtype=np.float32)
    hr_mins_bc = hr_mins_np.reshape(1, 1, 1, -1)
    hr_maxs_bc = hr_maxs_np.reshape(1, 1, 1, -1)
    lr_mins_bc = lr_mins_np.reshape(1, 1, 1, -1)
    lr_maxs_bc = lr_maxs_np.reshape(1, 1, 1, -1)

    sweep_csv = os.path.join(results_dir, "sweep_results.csv")
    if not os.path.exists(sweep_csv):
        with open(sweep_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "lambda_fft", "best_val_loss",
                "test_mse", "test_psnr", "test_ssim",
            ])

    for step, lam in enumerate(lambda_values):
        lam_str     = f"{lam:.12f}".replace(".", "p")
        lam_dir     = os.path.join(results_dir, "param_loss_function",
                                   f"fft_lambda_{lam_str}")
        os.makedirs(lam_dir, exist_ok=True)
        weight_path = os.path.join(lam_dir, UNET_WEIGHT_FILENAME)

        print(f"\n{'='*65}")
        print(f"  Sweep step {step+1}/{n_steps}  |  lambda_fft = {lam:.3e}")
        weights_existed = os.path.exists(weight_path)
        if weights_existed:
            if skip_if_done:
                print(f"  Weights found  →  skip_if_done=True : evaluation only")
            elif resume:
                print(f"  Weights found  →  resume=True : loading and continuing training")
            else:
                print(f"  Weights found  →  resume=False : retraining from scratch")
        else:
            print(f"  No weights found  →  training from scratch")
        print(f"{'='*65}\n")

        # Build model
        model = build_unet_spectral(lambda_fft=lam)
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=lr_rate,
                weight_decay=weight_decay,
            ),
        )

        # Load existing checkpoint when requested
        if weights_existed and (resume or skip_if_done):
            model.load_weights(weight_path)
            print(f"  Loaded weights from {weight_path}")

        # Training
        do_train = not (skip_if_done and weights_existed)

        if do_train:
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    filepath          = weight_path,
                    monitor           = "val_loss",
                    save_best_only    = True,
                    save_weights_only = True,
                    verbose           = 1,
                ),
                keras.callbacks.EarlyStopping(
                    monitor              = "val_loss",
                    patience             = EARLY_STOPPING_PATIENCE,
                    min_delta            = 1e-7,
                    mode                 = "min",
                    restore_best_weights = True,
                ),
            ]

            history = model.fit(
                datasets["train_ds"],
                validation_data  = datasets["val_ds"],
                epochs           = epochs,
                steps_per_epoch  = steps_per_epoch,
                validation_steps = validation_steps,
                verbose          = 1,
                callbacks        = callbacks,
            )

            best_val = min(history.history.get("val_loss", [float("inf")]))

            plt.figure(figsize=(8, 4))
            plt.plot(history.history["loss"],     label="train")
            plt.plot(history.history["val_loss"], label="val")
            plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
            plt.title(f"lambda_fft = {lam:.3e}")
            plt.tight_layout()
            plt.savefig(os.path.join(lam_dir, "loss_history.png"), dpi=150)
            plt.close()

        else:
            best_val = float("nan")
            print(f"  Training skipped (skip_if_done=True, weights already present).")

        # Evaluate on the test set
        from skimage.metrics import structural_similarity as sk_ssim

        mse_all, psnr_all, ssim_all = [], [], []
        ssim_map_u, ssim_map_v      = [], []

        for lr_b, hr_b in datasets["test_ds"]:
            pred    = tf.cast(model(lr_b, training=False), tf.float32)
            hr_true = tf.cast(hr_b, tf.float32)

            mse_b = float(tf.reduce_mean((pred - hr_true) ** 2).numpy())
            mse_all.append(mse_b)
            psnr_all.append(
                10.0 * float(tf.math.log(1.0 / (mse_b + 1e-10)).numpy())
                / float(tf.math.log(10.0).numpy()))

            ssim_u = tf.image.ssim(hr_true[..., 0:1], pred[..., 0:1], max_val=1.0)
            ssim_v = tf.image.ssim(hr_true[..., 1:2], pred[..., 1:2], max_val=1.0)
            ssim_all.append(float(((ssim_u + ssim_v) / 2.0).numpy().mean()))

            p_np = pred.numpy()
            t_np = hr_true.numpy()
            for b in range(p_np.shape[0]):
                _, mu = sk_ssim(t_np[b, ..., 0], p_np[b, ..., 0],
                                full=True, data_range=1.0, win_size=11)
                ssim_map_u.append(mu)
                _, mv = sk_ssim(t_np[b, ..., 1], p_np[b, ..., 1],
                                full=True, data_range=1.0, win_size=11)
                ssim_map_v.append(mv)

        mse_mean  = float(np.mean(mse_all))
        psnr_mean = float(np.mean(psnr_all))
        ssim_mean = float(np.mean(ssim_all))

        print(f"  lambda={lam:.5e}  MSE={mse_mean:.6f}  "
              f"PSNR={psnr_mean:.5f} dB  SSIM={ssim_mean:.5f}")

        if lat_hr is not None:
            pred_un    = p_np * (hr_maxs_bc - hr_mins_bc) + hr_mins_bc
            true_un    = t_np * (hr_maxs_bc - hr_mins_bc) + hr_mins_bc
            lr_un      = lr_b.numpy() * (lr_maxs_bc - lr_mins_bc) + lr_mins_bc
            lr_un_plot = lr_un[:, ::RF, ::RF, :] if lr_un.shape[1] == PX_LAT_HR \
                         else lr_un

            plot_comparison_transformer(
                lr_un_plot, pred_un, true_un,
                ssim_mean_u = np.stack(ssim_map_u).mean(axis=0),
                ssim_std_u  = np.stack(ssim_map_u).std(axis=0),
                ssim_mean_v = np.stack(ssim_map_v).mean(axis=0),
                ssim_std_v  = np.stack(ssim_map_v).std(axis=0),
                lat_hr=lat_hr, lon_hr=lon_hr,
                lat_lr=lat_lr, lon_lr=lon_lr,
                index=0, results_dir=lam_dir,
            )

        with open(sweep_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                f"{lam:.6e}", f"{best_val:.6f}",
                f"{mse_mean:.6f}", f"{psnr_mean:.4f}", f"{ssim_mean:.4f}",
            ])

        del model
        gc.collect()
        tf.keras.backend.clear_session()

    print(f"\nSweep complete. Results saved to {sweep_csv}")

    _plot_sweep_summary(
        csv_path    = sweep_csv,
        results_dir = results_dir,
        x_col       = r"$\lambda_{fft}$",
        sweep_label = r"$\lambda_{fft}$",
    )


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


def _compute_radial_grid(H: int, W: int) -> tuple:
    """Frequency axes and radial wavenumber of an (H, W) field.

    Args:
        H, W: field size.
    Returns:
        (freq_h, freq_w, radial): shifted frequencies in cycles per pixel, (H,) and (W,),
        and the radial wavenumber grid (H, W), up to about 0.707 in the corners.
    """
    freq_h = fftshift(scipy_fftfreq(H)).astype(np.float32)
    freq_w = fftshift(scipy_fftfreq(W)).astype(np.float32)
    FH, FW = np.meshgrid(freq_h, freq_w, indexing="ij")
    return freq_h, freq_w, np.sqrt(FH**2 + FW**2).astype(np.float32)


def _windowed_power(field_2d: np.ndarray, window2d: np.ndarray) -> np.ndarray:
    """Windowed 2-D power spectrum.

    Args:
        field_2d: field (H, W).
        window2d: window (H, W).
    Returns:
        Shifted power spectrum (H, W) of the mean-removed windowed field.
    """
    xw = (field_2d - field_2d.mean()).astype(np.float64) * window2d.astype(np.float64)
    F  = fftshift(scipy_fft2(xw, norm="ortho"))
    return np.abs(F) ** 2


def _scatter_psd(field_2d: np.ndarray, window2d: np.ndarray) -> tuple:
    """Unbinned power spectrum.

    Args:
        field_2d: field (H, W).
        window2d: window (H, W).
    Returns:
        (k, power) arrays of all wavenumbers sorted in increasing order.
    """
    H, W = field_2d.shape
    _, _, radial = _compute_radial_grid(H, W)
    power = _windowed_power(field_2d, window2d)
    idx   = np.argsort(radial.ravel())
    return radial.ravel()[idx], power.ravel()[idx]


def _radial_psd(
    field_2d: np.ndarray,
    window2d: np.ndarray,
    n_bins:   int   = 128,
    cutoff:   float = None,
) -> tuple:
    """Azimuthally averaged power spectrum.

    Args:
        field_2d: field (H, W).
        window2d: window (H, W).
        n_bins: number of radial bins.
        cutoff: if given, power at wavenumbers up to cutoff is set to zero.
    Returns:
        (k_mid, psd): bin centres in cycles per pixel and mean power per bin, both of length n_bins.
    """
    H, W = field_2d.shape
    freq_h, freq_w, radial = _compute_radial_grid(H, W)
    power = _windowed_power(field_2d, window2d)

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


def plot_fft_analysis(
    results_dir:     str,
    datasets:        dict,
    minmax_stats:    dict,
    build_model_fn,
    lambda_values:   list  = None,
    cutoff:          float = FFT_CUTOFF,
    n_bins:          int   = 128,
    n_samples_avg:   int   = 50,
    weight_filename: str   = UNET_WEIGHT_FILENAME,
):
    """Save spectral comparison figures of the models of the sweep.

    Writes sample_fft_binned.pdf, average_fft_binned.pdf and sample_fft_2d.pdf in results_dir, comparing
    ground truth, LR input and the prediction of the model of each lambda_fft value.

    Args:
        results_dir: output directory containing the model folders.
        datasets: dictionary with test_ds.
        minmax_stats: dictionary with hr_mins, hr_maxs, lr_mins, lr_maxs (lists of floats).
        build_model_fn: function taking lambda_fft and returning the model.
        lambda_values: lambda_fft values to load (default LAMBDA_VALUES).
        cutoff: high-pass cutoff drawn on the figures, in cycles per pixel.
        n_bins: number of radial bins.
        n_samples_avg: number of samples of the averaged spectra.
        weight_filename: weight file name inside each lambda_fft folder.
    """
    if lambda_values is None:
        lambda_values = LAMBDA_VALUES

    hr_mins = minmax_stats.get("hr_mins", [0.0, 0.0])
    hr_maxs = minmax_stats.get("hr_maxs", [1.0, 1.0])

    def _denorm_hr(arr_norm):
        """Unscale a normalised HR array with the HR min/max."""
        out = arr_norm.copy()
        for c in range(arr_norm.shape[-1]):
            out[..., c] = arr_norm[..., c] * (hr_maxs[c] - hr_mins[c]) + hr_mins[c]
        return out.astype(np.float32)

    win2d      = _build_analysis_window(HR_H, HR_W)
    comp_names = ["U (100 m)", "V (100 m)"]
    n_lam      = len(lambda_values)
    cmap       = plt.get_cmap("plasma")
    colors     = [cmap(i / max(n_lam - 1, 1)) for i in range(n_lam)]

    def _lbl(lam):
        """Legend label of a lambda_fft value."""
        return f"lambda={lam:.4e}" if lam > 0 else "lambda=0 (MSE only)"

    freq_h_grid, freq_w_grid, _ = _compute_radial_grid(HR_H, HR_W)

    # Load trained models
    print("Loading UNet models for FFT analysis...")
    models = {}
    for lam in lambda_values:
        lam_str = f"{lam:.12f}".replace(".", "p")
        wpath   = os.path.join(results_dir, "param_loss_function",
                               f"fft_lambda_{lam_str}", weight_filename)
        m = build_model_fn(lam)
        if os.path.exists(wpath):
            m.load_weights(wpath)
            print(f"  lambda={lam:.12e}  ->  weights loaded from {wpath}")
        else:
            print(f"  WARNING  lambda={lam:.12e}  ->  no weights at {wpath}")
        models[lam] = m

    # Collect test-set predictions
    print(f"Collecting {n_samples_avg} test batches...")
    gt_batches   = []
    pred_batches = {lam: [] for lam in lambda_values}

    for bi, (lr_b, hr_b) in enumerate(datasets["test_ds"]):
        if bi >= n_samples_avg:
            break
        gt_batches.append(_denorm_hr(hr_b.numpy()))
        for lam, m in models.items():
            pred_batches[lam].append(_denorm_hr(m(lr_b, training=False).numpy()))

    def _collect_binned(arr_list, ch):
        """Binned PSD of one channel for each array of a list."""
        return [
            _radial_psd(arr_list[bi][b, :, :, ch], win2d, n_bins=n_bins)[1]
            for bi in range(len(arr_list))
            for b  in range(arr_list[bi].shape[0])
        ]

    # Figures 1 & 2 - Binned isotropic PSD
    def _binned_figure(use_avg, save_name):
        """Save the binned PSD figure of the first sample (with PSD difference) or averaged over the samples."""
        if use_avg:
            fig, axes = plt.subplots(2, 1, figsize=(9, 13))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(18, 13))

        tag = "Averaged" if use_avg else "Single-sample"
        fig.suptitle(
            f"{tag} — Binned Isotropic PSD (UNet)\n"
            f"HP cutoff = {cutoff:.3f} cy/px",
        )
        k_ax = _radial_psd(gt_batches[0][0, :, :, 0], win2d, n_bins=n_bins)[0]

        for ch, comp in enumerate(comp_names):
            # For averaged: axes is 1D, so ax_psd only
            # For sample: axes is 2D, so both ax_psd and ax_diff
            if use_avg:
                ax_psd = axes[ch]
            else:
                ax_psd, ax_diff = axes[ch, 0], axes[ch, 1]

            if use_avg:
                gt_stack = np.stack(_collect_binned(gt_batches, ch))
                gt_mean  = gt_stack.mean(0)
                ax_psd.semilogy(k_ax, gt_mean, "k-", lw=2.5, label="GT", zorder=5)
                gt_ref = gt_mean
            else:
                gt_ref = _radial_psd(
                    gt_batches[0][0, :, :, ch], win2d, n_bins=n_bins)[1]
                ax_psd.semilogy(k_ax, gt_ref, "k-", lw=2.5, label="GT", zorder=5)

            for i, lam in enumerate(lambda_values):
                if use_avg:
                    pr_mean = np.stack(_collect_binned(pred_batches[lam], ch)).mean(0)
                    ax_psd.semilogy(k_ax, pr_mean, lw=1.8,
                                    color=colors[i], label=_lbl(lam))
                else:
                    pr = _radial_psd(
                        pred_batches[lam][0][0, :, :, ch], win2d, n_bins=n_bins)[1]
                    ax_psd.semilogy(k_ax, pr, lw=1.8,
                                    color=colors[i], label=_lbl(lam))
                    diff = pr - gt_ref
                    ax_diff.semilogy(k_ax, np.abs(diff) + 1e-20, lw=1.8,
                                     color=colors[i], label=_lbl(lam))

            # Decorate ax_psd
            ax_psd.axvline(cutoff,     color="red",       ls="--", lw=1.5,
                           label=f"HP cutoff ({cutoff:.3f})")
            ax_psd.set_xlabel("k  (cycles / HR pixel)")
            ax_psd.set_ylabel(f"PSD  [{comp}]^2")
            ax_psd.set_title(f"Binned Isotropic PSD — {comp}")
            ax_psd.grid(True, which="both", ls=":", alpha=0.35)
            ax_psd.legend(fontsize=13, loc="upper right")

            # Decorate ax_diff only for single-sample figure
            if not use_avg:
                ax_diff.axvline(cutoff,     color="red",       ls="--", lw=1.5,
                                label=f"HP cutoff ({cutoff:.3f})")
                ax_diff.set_xlabel("k  (cycles / HR pixel)")
                ax_diff.set_ylabel("|delta PSD|  (pred - GT)")
                ax_diff.set_title(f"|PSD difference| — {comp}")
                ax_diff.grid(True, which="both", ls=":", alpha=0.35)
                ax_diff.legend(fontsize=13, loc="upper right")

        plt.tight_layout()
        sp = os.path.join(results_dir, save_name)
        plt.savefig(sp, dpi=300, bbox_inches="tight")
        print(f"Saved -> {sp}")
        plt.close(fig)

    _binned_figure(use_avg=False, save_name="sample_fft_binned.png")
    _binned_figure(use_avg=True,  save_name="average_fft_binned.png")

    # Figure 3 - 2D log-power spectrum
    n_cols = 1 + n_lam
    fig_2d, axes_2d = plt.subplots(
        2, n_cols, figsize=(4.2 * n_cols, 5.5 * 2),
        squeeze=False, constrained_layout=True,
    )
    fig_2d.suptitle(
        f"2D log10-Power Spectrum (DC-centred) — UNet — first test sample\n"
        f"Cyan circle = HP cutoff ({cutoff:.3f} cy/px)",
        fontsize=20,
    )

    s_gt   = gt_batches[0][0]
    s_pred = {lam: pred_batches[lam][0][0] for lam in lambda_values}
    extent = [float(freq_w_grid.min()), float(freq_w_grid.max()),
              float(freq_h_grid.min()), float(freq_h_grid.max())]
    theta  = np.linspace(0, 2 * np.pi, 360)
    cx, cy = cutoff * np.cos(theta), cutoff * np.sin(theta)

    for ch, comp in enumerate(comp_names):
        log_pws = [np.log10(_windowed_power(s_gt[:, :, ch], win2d) + 1e-20)]
        for lam in lambda_values:
            log_pws.append(
                np.log10(_windowed_power(s_pred[lam][:, :, ch], win2d) + 1e-20))

        vmin = np.percentile(np.stack(log_pws), 1)
        vmax = np.percentile(np.stack(log_pws), 99)

        col_data = [("GT", log_pws[0])] + [
            (_lbl(lam), log_pws[j + 1]) for j, lam in enumerate(lambda_values)]

        for col, (title, lp) in enumerate(col_data):
            ax = axes_2d[ch, col]
            im = ax.imshow(lp, origin="lower", extent=extent, aspect="auto",
                           cmap="inferno", vmin=vmin, vmax=vmax,
                           interpolation="nearest")
            ax.plot(cx, cy, color="cyan", lw=1.2, ls="--")
            ax.axvline(0, color="white", lw=0.6, ls=":", alpha=0.7)
            ax.axhline(0, color="white", lw=0.6, ls=":", alpha=0.7)
            ax.set_title(f"{title}")
            ax.set_xlabel("k_x  (cy/px)")
            if col == 0:
                ax.set_ylabel("k_y  (cy/px)")

        cbar = fig_2d.colorbar(im, ax=axes_2d[ch, :].tolist(),
                               shrink=0.80, pad=0.02, aspect=30)
        cbar.set_label("log10(Power)")

    sp2d = os.path.join(results_dir, "sample_fft_2d.png")
    plt.savefig(sp2d, dpi=300, bbox_inches="tight")
    print(f"Saved -> {sp2d}")
    plt.close(fig_2d)

    print("[plot_fft_analysis] All three figures saved.")


def _load_plot_inputs(case_dir: str, shared_dir: str, batch_size: int) -> tuple:
    """Load the test data needed by PLOT_ONLY mode.

    LR is scaled and bilinearly upsampled to the HR grid, as in build_unet_pipeline_from_prepared.

    Args:
        case_dir: folder Data_<case>.
        shared_dir: folder Data_Shared.
        batch_size: batch size.
    Returns:
        (datasets, minmax_stats) with min/max values as flat lists.
    """
    csv_path = os.path.join(case_dir, f"statistics_{CASE}.csv")
    df       = pd.read_csv(csv_path)

    def _get_row(name):
        """Return the training row of the statistics CSV for a dataset name."""
        row = df[(df["dataset"] == name) & (df["split"] == "train")]
        if row.empty:
            raise ValueError(f"No 'train' row for dataset={name!r} in {csv_path}")
        return row.iloc[0]

    row_lr = _get_row("ERA5")
    row_hr = _get_row("CERRA")

    lr_mins_np = [float(row_lr["min_u"]), float(row_lr["min_v"])]
    lr_maxs_np = [float(row_lr["max_u"]), float(row_lr["max_v"])]
    hr_mins_np = [float(row_hr["min_u"]), float(row_hr["min_v"])]
    hr_maxs_np = [float(row_hr["max_u"]), float(row_hr["max_v"])]

    print(f"  LR min/max: {list(zip(lr_mins_np, lr_maxs_np))}")
    print(f"  HR min/max: {list(zip(hr_mins_np, hr_maxs_np))}")

    lr_mins    = [tf.constant(v, dtype=tf.float32) for v in lr_mins_np]
    lr_maxs    = [tf.constant(v, dtype=tf.float32) for v in lr_maxs_np]
    hr_mins_tf = [tf.constant(v, dtype=tf.float32) for v in hr_mins_np]
    hr_maxs_tf = [tf.constant(v, dtype=tf.float32) for v in hr_maxs_np]

    def _read_sidecar(npy_path):
        """Read the shape and dtype from the sidecar file of a memmap."""
        sidecar = npy_path.replace(".npy", "_shape.txt")
        lines   = open(sidecar).read().strip().splitlines()
        shape   = tuple(int(s) for s in lines[0].split(","))
        dtype   = np.dtype(lines[1]) if len(lines) > 1 else np.dtype("float32")
        return shape, dtype

    paths = {
        "lr_test": os.path.join(shared_dir, "test", "ERA5_test.npy"),
        "hr_test": os.path.join(shared_dir, "test", "CERRA_test.npy"),
    }
    mmaps, shapes = {}, {}
    for key, path in paths.items():
        shape, dtype   = _read_sidecar(path)
        mmaps[key]     = np.memmap(path, dtype=dtype, mode="r", shape=shape)
        shapes[key]    = shape
        print(f"  {key:12s}  shape={shape}  dtype={dtype}")

    n_test       = shapes["lr_test"][0]
    H_lr, W_lr   = shapes["lr_test"][1], shapes["lr_test"][2]
    H_hr, W_hr   = shapes["hr_test"][1], shapes["hr_test"][2]
    lr_ch, hr_ch = shapes["lr_test"][3], shapes["hr_test"][3]

    def normalize_lr(x):
        """Min-max scale a LR sample channel by channel."""
        return tf.stack(
            [(x[..., i] - lr_mins[i]) / (lr_maxs[i] - lr_mins[i])
             for i in range(lr_ch)], axis=-1)

    def normalize_hr(x):
        """Min-max scale a HR sample channel by channel."""
        return tf.stack(
            [(x[..., i] - hr_mins_tf[i]) / (hr_maxs_tf[i] - hr_mins_tf[i])
             for i in range(hr_ch)], axis=-1)

    def upsample_lr(x):
        """Bilinearly resize a LR sample to the HR grid."""
        return tf.image.resize(x, (H_hr, W_hr), method="bilinear")

    def lr_gen():
        """Yield LR samples."""
        for i in range(n_test):
            yield mmaps["lr_test"][i].astype(np.float32)

    def hr_gen():
        """Yield HR samples."""
        for i in range(n_test):
            yield mmaps["hr_test"][i].astype(np.float32)

    tf_lr = (tf.data.Dataset.from_generator(
                 lr_gen,
                 output_signature=tf.TensorSpec(
                     shape=(H_lr, W_lr, lr_ch), dtype=tf.float32))
             .map(normalize_lr, num_parallel_calls=tf.data.AUTOTUNE)
             .map(upsample_lr,  num_parallel_calls=tf.data.AUTOTUNE))

    tf_hr = (tf.data.Dataset.from_generator(
                 hr_gen,
                 output_signature=tf.TensorSpec(
                     shape=(H_hr, W_hr, hr_ch), dtype=tf.float32))
             .map(normalize_hr, num_parallel_calls=tf.data.AUTOTUNE))

    test_ds = (tf.data.Dataset.zip((tf_lr, tf_hr))
               .batch(batch_size, drop_remainder=True)
               .prefetch(tf.data.AUTOTUNE))

    return (
        {"test_ds": test_ds},
        {"hr_mins": hr_mins_np, "hr_maxs": hr_maxs_np,
         "lr_mins": lr_mins_np, "lr_maxs": lr_maxs_np},
    )


def build_unet_pipeline(
        lr_mmap_path,
        hr_mmap_path,
        lr_shape,
        hr_shape,
        batch_size     = BATCH_SIZE,
        split_ratios   = (0.70, 0.15, 0.15),
        shuffle_buffer = 3000,
):
    """Build train / val / test datasets from raw float32 memmap files (CONVERTING_DATA=True).

    Min/max scaling constants are computed on the training split only. LR samples are scaled and
    bilinearly upsampled to the HR grid.

    Args:
        lr_mmap_path, hr_mmap_path: memmap files of the LR and HR arrays.
        lr_shape, hr_shape: shapes (N, H, W, C) of the two arrays.
        batch_size: batch size.
        split_ratios: (train, val, test) fractions.
        shuffle_buffer: shuffle buffer size.
    Returns:
        (datasets, minmax_stats). datasets has train_ds, val_ds, test_ds and n_train, n_val, n_test.
        minmax_stats is {"lr": (mins, maxs), "hr": (mins, maxs)} with one tf scalar per channel.
    """
    lr_mmap = np.memmap(lr_mmap_path, dtype="float32", mode="r", shape=lr_shape)
    hr_mmap = np.memmap(hr_mmap_path, dtype="float32", mode="r", shape=hr_shape)

    N                  = lr_shape[0]
    H_lr, W_lr, lr_ch = lr_shape[1], lr_shape[2], lr_shape[3]
    H_hr, W_hr, hr_ch = hr_shape[1], hr_shape[2], hr_shape[3]

    def lr_generator():
        """Yield LR samples from the memmap."""
        for i in range(N):
            yield lr_mmap[i].astype(np.float32)

    def hr_generator():
        """Yield HR samples from the memmap."""
        for i in range(N):
            yield hr_mmap[i].astype(np.float32)

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
                min_c   = tf.minimum(min_c, tf.reduce_min(ch_vals))
                max_c   = tf.maximum(max_c, tf.reduce_max(ch_vals))
            mins.append(min_c); maxs.append(max_c)
        return mins, maxs

    lr_mins, lr_maxs = compute_channel_minmax(tf_ds_lr.take(n_train), lr_ch)
    hr_mins, hr_maxs = compute_channel_minmax(tf_ds_hr.take(n_train), hr_ch)

    def normalize_lr(x):
        """Min-max scale a LR sample channel by channel."""
        return tf.stack(
            [(x[..., i] - lr_mins[i]) / (lr_maxs[i] - lr_mins[i] + 1e-8)
             for i in range(lr_ch)], axis=-1)

    def normalize_hr(x):
        """Min-max scale a HR sample channel by channel."""
        return tf.stack(
            [(x[..., i] - hr_mins[i]) / (hr_maxs[i] - hr_mins[i] + 1e-8)
             for i in range(hr_ch)], axis=-1)

    def upsample_lr(x):
        """Bilinearly resize a LR sample to the HR grid."""
        return tf.image.resize(x, (H_hr, W_hr), method="bilinear")

    tf_ds_lr_up = (tf_ds_lr
                   .map(normalize_lr, num_parallel_calls=tf.data.AUTOTUNE)
                   .map(upsample_lr,  num_parallel_calls=tf.data.AUTOTUNE))
    tf_ds_hr_sc = tf_ds_hr.map(normalize_hr, num_parallel_calls=tf.data.AUTOTUNE)
    dataset     = (tf.data.Dataset.zip((tf_ds_lr_up, tf_ds_hr_sc))
                   .shuffle(min(shuffle_buffer, N), seed=42))

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
    return (
        {"train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds,
         "n_train": n_train, "n_val": n_val, "n_test": n_test},
        {"lr": (lr_mins, lr_maxs), "hr": (hr_mins, hr_maxs)},
    )


def build_unet_pipeline_from_prepared(
        case_dir,
        shared_dir,
        csv_path,
        batch_size     = BATCH_SIZE,
        shuffle_buffer = 3000,
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
        txt   = npy_path.replace(".npy", "_shape.txt")
        lines = open(txt).read().strip().splitlines()
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
        lr_sh, lr_dt = _sidecar(lr_path)
        hr_sh, _     = _sidecar(hr_path)
        shapes[split] = {"lr": lr_sh, "hr": hr_sh, "dtype": lr_dt,
                         "lr_path": lr_path, "hr_path": hr_path}
        print(f"  [{split}] LR {lr_sh}  HR {hr_sh}")

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
            mc, xc = comp_map[c % 2]
            mins.append(tf.constant(float(stats_df.loc[row_label, mc]),
                                    dtype=tf.float32))
            maxs.append(tf.constant(float(stats_df.loc[row_label, xc]),
                                    dtype=tf.float32))
        return mins, maxs

    lr_mins, lr_maxs = _get_minmax("ERA5",  INPUT_FRAMES)
    hr_mins, hr_maxs = _get_minmax("CERRA", OUTPUT_FRAMES)

    def normalize_lr(x):
        """Min-max scale a LR sample channel by channel."""
        return tf.stack(
            [(x[..., i] - lr_mins[i]) / (lr_maxs[i] - lr_mins[i] + 1e-8)
             for i in range(INPUT_FRAMES)], axis=-1)

    def normalize_hr(x):
        """Min-max scale a HR sample channel by channel."""
        return tf.stack(
            [(x[..., i] - hr_mins[i]) / (hr_maxs[i] - hr_mins[i] + 1e-8)
             for i in range(OUTPUT_FRAMES)], axis=-1)

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
        info               = shapes[split]
        lr_sh, hr_sh, dtype = info["lr"], info["hr"], info["dtype"]
        lr_mm   = np.memmap(info["lr_path"], dtype=dtype, mode="r", shape=lr_sh)
        hr_mm   = np.memmap(info["hr_path"], dtype=dtype, mode="r", shape=hr_sh)
        N_split = lr_sh[0]
        H_lr, W_lr, lc = lr_sh[1], lr_sh[2], lr_sh[3]
        H_hr, W_hr, hc = hr_sh[1], hr_sh[2], hr_sh[3]

        def _lr_gen():
            """Yield LR samples."""
            for i in range(N_split):
                yield lr_mm[i].astype(np.float32)

        def _hr_gen():
            """Yield HR samples."""
            for i in range(N_split):
                yield hr_mm[i].astype(np.float32)

        tf_lr = (tf.data.Dataset.from_generator(
                     _lr_gen,
                     output_signature=tf.TensorSpec(
                         shape=(H_lr, W_lr, lc), dtype=tf.float32))
                 .map(normalize_lr, num_parallel_calls=tf.data.AUTOTUNE)
                 .map(upsample_lr,  num_parallel_calls=tf.data.AUTOTUNE))

        tf_hr = (tf.data.Dataset.from_generator(
                     _hr_gen,
                     output_signature=tf.TensorSpec(
                         shape=(H_hr, W_hr, hc), dtype=tf.float32))
                 .map(normalize_hr, num_parallel_calls=tf.data.AUTOTUNE))

        ds = tf.data.Dataset.zip((tf_lr, tf_hr))
        if shuffle:
            ds = ds.shuffle(min(shuffle_buffer, N_split), seed=42)
        ds = ds.batch(batch_size, drop_remainder=True)
        if repeat:
            ds = ds.repeat()
        return ds.prefetch(tf.data.AUTOTUNE), N_split

    train_ds, n_train = _make_split_ds("train", shuffle=True,  repeat=True)
    val_ds,   n_val   = _make_split_ds("val",   shuffle=False, repeat=True)
    test_ds,  n_test  = _make_split_ds("test",  shuffle=False, repeat=False)

    print(f"  train={n_train}  val={n_val}  test={n_test}")

    return (
        {"train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds,
         "n_train": n_train, "n_val": n_val, "n_test": n_test},
        {"lr": (lr_mins, lr_maxs), "hr": (hr_mins, hr_maxs)},
    )


def main():
    """Run the lambda_fft sweep for CASE: data preparation or checks, datasets, sweep, spectral figures and timing CSV (or only the figures with PLOT_ONLY)."""

    with tf.device('/device:GPU:0'):

        # PLOT_ONLY shortcut - skip all preprocessing and training
        if PLOT_ONLY:
            print("PLOT_ONLY mode — loading test data and trained UNet weights...")
            case_dir               = os.path.join(OUTPUT_ROOT, f"Data_{CASE}")
            datasets, minmax_stats = _load_plot_inputs(
                case_dir=case_dir, shared_dir=SHARED_DIR, batch_size=BATCH_SIZE)
            plot_fft_analysis(
                results_dir     = RESULTS_DIR,
                datasets        = datasets,
                minmax_stats    = minmax_stats,
                build_model_fn  = lambda lam: build_unet_spectral(lambda_fft=lam),
                lambda_values   = LAMBDA_VALUES,
                cutoff          = FFT_CUTOFF,
                n_bins          = 128,
                n_samples_avg   = 2000,
                weight_filename = UNET_WEIGHT_FILENAME,
            )
            return

        # Pre-processing
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

            ds_wind_ERA5  = time_crop(ds_wind_ERA5,  TIME_RANGE[0], TIME_RANGE[1],
                                      time_var="time")
            ds_wind_CERRA = time_crop(ds_wind_CERRA, TIME_RANGE[0], TIME_RANGE[1],
                                      time_var="time")
            ds_wind_ERA5  = domain_crop(ds_wind_ERA5,
                                        LAT_COORDS[0], LAT_COORDS[1],
                                        LON_COORDS[0], LON_COORDS[1],
                                        variable_name="long")
            ds_wind_CERRA = domain_crop(ds_wind_CERRA,
                                        LAT_COORDS_CERRA[0], LAT_COORDS_CERRA[1],
                                        LON_COORDS_CERRA[0], LON_COORDS_CERRA[1],
                                        variable_name="short")

            mask         = ds_wind_ERA5["time"].dt.hour.isin([0, 3, 6, 9, 12, 15, 18, 21])
            ds_wind_ERA5 = ds_wind_ERA5.isel(time=mask)
            ds_wind_CERRA = wind_speed_to_uv(ds_wind_CERRA)
            ds_wind_CERRA = ds_wind_CERRA.sel(lat=slice(None, None, -1))

            u_era5  = ds_wind_ERA5["u100"].values.astype(np.float32)
            v_era5  = ds_wind_ERA5["v100"].values.astype(np.float32)
            u_cerra = ds_wind_CERRA["u100"].values.astype(np.float32)
            v_cerra = ds_wind_CERRA["v100"].values.astype(np.float32)

            arrays_np_lr  = np.stack([u_era5,  v_era5],  axis=-1)
            arrays_np_hr  = np.stack([u_cerra, v_cerra], axis=-1)
            arrays_seq_lr = sequence_encoding(arrays_np_lr, N_BEFORE, N_AFTER,
                                              keep_channels=True)
            arrays_np_hr  = HR_sequence_slicing(arrays_np_hr, N_BEFORE, N_AFTER)

            lat_HR = ds_wind_CERRA["lat"].values
            lon_HR = ds_wind_CERRA["lon"].values
            lat_lr = ds_wind_ERA5["latitude"].values
            lon_lr = ds_wind_ERA5["longitude"].values

            np.savetxt(lat_ERA5_path, lat_lr);  np.savetxt(lon_ERA5_path, lon_lr)
            np.savetxt(lat_CERRA_path, lat_HR); np.savetxt(lon_CERRA_path, lon_HR)

            lr_mmap_path = os.path.join(RESULTS_DIR, "arrays_np_lr.dat")
            hr_mmap_path = os.path.join(RESULTS_DIR, "arrays_np_hr.dat")

            lr_mm    = np.memmap(lr_mmap_path, dtype="float32", mode="w+",
                                 shape=arrays_seq_lr.shape)
            lr_mm[:] = arrays_seq_lr[:]; lr_mm.flush()
            np.save(os.path.join(RESULTS_DIR, "lr_shape.npy"), arrays_seq_lr.shape)

            hr_mm    = np.memmap(hr_mmap_path, dtype="float32", mode="w+",
                                 shape=arrays_np_hr.shape)
            hr_mm[:] = arrays_np_hr[:]; hr_mm.flush()
            np.save(os.path.join(RESULTS_DIR, "hr_shape.npy"), arrays_np_hr.shape)

            del arrays_np_lr, arrays_seq_lr, arrays_np_hr, ds_wind_ERA5, ds_wind_CERRA
            gc.collect(); tf.keras.backend.clear_session()
            timer.stop("preprocessing")
            print("Pre-processing finished.")

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
                        f"Prepared memmap not found: {required}\n"
                        f"Run prepare_training_data.py with CASES=['{CASE}'] first.")

            coords_dir = os.path.join(SHARED_DIR, "coords")
            lat_HR = np.loadtxt(os.path.join(coords_dir, "lat_HR.npy"))
            lon_HR = np.loadtxt(os.path.join(coords_dir, "lon_HR.npy"))
            lat_lr = np.loadtxt(os.path.join(coords_dir, "lat_LR.npy"))
            lon_lr = np.loadtxt(os.path.join(coords_dir, "lon_LR.npy"))
            timer.stop("preprocessing")
            print("Shape verification will run inside build_unet_pipeline_from_prepared.")

        # Load coordinates (CONVERTING_DATA=True path only)
        if CONVERTING_DATA:
            lat_HR = np.loadtxt(lat_CERRA_path)
            lon_HR = np.loadtxt(lon_CERRA_path)
            lat_lr = np.loadtxt(lat_ERA5_path)
            lon_lr = np.loadtxt(lon_ERA5_path)

        # Data pipeline
        if CONVERTING_DATA:
            lr_shape = tuple(np.load(os.path.join(RESULTS_DIR, "lr_shape.npy")))
            hr_shape = tuple(np.load(os.path.join(RESULTS_DIR, "hr_shape.npy")))
            print(f"LR shape: {lr_shape}   HR shape: {hr_shape}")
            datasets, minmax_stats = build_unet_pipeline(
                lr_mmap_path   = os.path.join(RESULTS_DIR, "arrays_np_lr.dat"),
                hr_mmap_path   = os.path.join(RESULTS_DIR, "arrays_np_hr.dat"),
                lr_shape       = lr_shape,
                hr_shape       = hr_shape,
                batch_size     = BATCH_SIZE,
                shuffle_buffer = 3000,
            )
        else:
            csv_path = os.path.join(case_dir, f"statistics_{CASE}.csv")
            datasets, minmax_stats = build_unet_pipeline_from_prepared(
                case_dir       = case_dir,
                shared_dir     = SHARED_DIR,
                csv_path       = csv_path,
                batch_size     = BATCH_SIZE,
                shuffle_buffer = 3000,
            )

        count = sum(1 for _ in datasets["test_ds"])
        print(f"Number of test batches: {count}")

        # Parametric sweep
        timer.start("training")
        print("Starting UNet FFT lambda sweep...")

        run_unet_fft_sweep(
            datasets         = datasets,
            minmax_stats     = minmax_stats,
            results_dir      = RESULTS_DIR,
            case             = CASE,
            n_steps          = N_STEPS,
            lambda_min       = LAMB_MIN,
            lambda_max       = LAMB_MAX,
            epochs           = EPOCHS,
            lr_rate          = learning_rate,
            batch_size       = BATCH_SIZE,
            validation_steps = max(1, datasets["n_val"] // BATCH_SIZE - 1),
            lat_hr           = lat_HR,
            lon_hr           = lon_HR,
            lat_lr           = lat_lr,
            lon_lr           = lon_lr,
            resume           = False,
            skip_if_done     = False,
        )

        timer.stop("training")

        # Spectral analysis plots (post-sweep)
        flat_minmax = {
            "hr_mins": [float(v) for v in minmax_stats["hr"][0]],
            "hr_maxs": [float(v) for v in minmax_stats["hr"][1]],
            "lr_mins": [float(v) for v in minmax_stats["lr"][0]],
            "lr_maxs": [float(v) for v in minmax_stats["lr"][1]],
        }

        print("\nGenerating spectral analysis plots...")
        plot_fft_analysis(
            results_dir     = RESULTS_DIR,
            datasets        = datasets,
            minmax_stats    = flat_minmax,
            build_model_fn  = lambda lam: build_unet_spectral(lambda_fft=lam),
            lambda_values   = LAMBDA_VALUES,
            cutoff          = FFT_CUTOFF,
            n_bins          = 128,
            n_samples_avg   = 50,
            weight_filename = UNET_WEIGHT_FILENAME,
        )

        # Append the run times to computation_time_runs.csv
        total_time  = time.perf_counter() - total_start
        csv_file    = os.path.join(RESULTS_DIR, "computation_time_runs.csv")
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "run_start_time", "case",
                    "preprocessing_seconds", "training_seconds",
                    "sampling_seconds", "total_seconds",
                ])
            writer.writerow([
                run_start_str, CASE,
                f"{timer.times.get('preprocessing', 0.0):.4f}",
                f"{timer.times.get('training',      0.0):.4f}",
                f"{timer.times.get('sampling',      0.0):.4f}",
                f"{total_time:.4f}",
            ])


if __name__ == "__main__":
    main()
