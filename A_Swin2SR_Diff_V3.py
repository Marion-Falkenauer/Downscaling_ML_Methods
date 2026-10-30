"""Diffusion model with a Swin2SR denoiser for ERA5 to CERRA wind fields downscaling.

The denoiser (A_Swin2SR_denoiser.get_swin2_denoising_network) receives the two ERA5 channels
(u, v) bilinearly upsampled to the CERRA grid together with the noisy CERRA channels (prescribed noise level), and
predicts the added noise. Training and test data are the memmaps written by
prepare_training_data.py: LR (N, 64, 80, 2) and HR (N, 320, 400, 2).

Outputs per case : training curves, comparison maps,
temporal metrics, spectra (FFT), wind speed and quiver figures, pred_arrays_<case>.npz,
test_metrics.csv and computation_time_runs.csv.

Configuration (module constants):
    CASES: training-set sizes run in sequence, keys of TRAIN_DURATION
        ("1year", "2year", "4year", "5year", "10year", "20year").
    CONVERTING_DATA: True converts the raw GRIB/NetCDF files (debug run, no training),
        False reads the prepared memmaps.
    PLOT_ONLY: True skips training and evaluation and redraws the spectral figures from
        the saved pred_arrays_<case>.npz.
    MIXED_PRECISION: True enables the mixed_bfloat16 policy.
    EPOCHS: maximum number of training epochs (0 evaluates the loaded weights only).
    BATCH_SIZE (the Swin2SR denoiser needs more memory than the U-Net), learning_rate, weight_decay, ema:
        optimisation settings.
    DIFFUSION_STEPS_CHOICE: number of reverse diffusion steps at inference (no need to go over 50 with the type of network/implementation chosen).
    SWIN_CONFIG: Swin2SR hyperparameters. Default embed_dim=180, depths=(8,)*6 (about 20 M parameters, compared configuration with UNet);
        embed_dim=96, depths=(6,)*6 gives about 12 M and embed_dim=48, depths=(2,)*6 of about 5 M parameters (to be ran on smaller GPUs for debugging purposes).
        with SWIN_LARGE_CONFIG and SWIN_SMALL_CONFIG from A_Swin2SR_denoiser are ready-made alternatives (available at the end of the code A-Swin2SR_denoiser.py).
    N_BEFORE, N_AFTER: time steps before and after the current one in the LR input (base case, N_BEFORE = 0 and N_AFTER = 0).
    FFT_CUTOFF: high-pass cutoff of the spectral figures, in cycles per pixel.
    BASE_DIR, SHARED_DIR: root folder and folder with the validation and test memmaps.

Paths BASE_DIR and PRETRAINED_BASE (pretrained weights loaded before training or evaluation, set inside run_case) must be adapted to the local file system.

Author: Marion Falkenauer (with the help of Claude)
"""

import xarray as xr
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Variable to change the GPU to the one chosen on the machine, depending on local availability
os.environ['TF_XLA_FLAGS']          = '--tf_xla_auto_jit=2'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import csv
import gc
import time
import tensorflow as tf
from tensorflow import keras
from keras import layers
from datetime import datetime

from scipy.fft    import fft2 as scipy_fft2, fftshift, fftfreq as scipy_fftfreq
from scipy.signal import windows as scipy_windows

from ML_Methods_ERA5_CERRA import (
    plot_comparison_diffusion,
    ml_diffusion_pipeline_streamed,
)
from setup import min_signal_rate, max_signal_rate

from A_Swin2SR_denoiser import get_swin2_denoising_network, SWIN_LARGE_CONFIG, SWIN_SMALL_CONFIG


plt.rcParams.update({
    "font.size":       20,
    "axes.labelsize":  20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 20,
    "axes.titlesize":  20,
})


BASE_DIR = "path to working directory"
ERA5_files      = os.path.join(BASE_DIR, "Data/ERA5")
grib_files_ERA5 = sorted(glob.glob(os.path.join(ERA5_files, "*.grib")))
print(f"Found {len(grib_files_ERA5)} ERA5 files.")

CERRA_files      = os.path.join(BASE_DIR, "Data/CERRA")
grib_files_CERRA = sorted(glob.glob(os.path.join(CERRA_files, "*.nc")))
print(f"Found {len(grib_files_CERRA)} CERRA files.")


CONVERTING_DATA = False
UV_COMP         = True
PLOT_ONLY       = False
MIXED_PRECISION = False

RF        = 5
PX_LAT_HR = 320
PX_LON_HR = 400
PX_LAT_LR = 64
PX_LON_LR = 80

N_BEFORE = 0
N_AFTER  = 0

CASES = ["1year","2year","4year"]

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

LON_COORDS,       LAT_COORDS       = [-5, 14.75],  [44, 59.75]
LON_COORDS_CERRA, LAT_COORDS_CERRA = [-5, 14.95],  [44, 59.95]
TIME_RANGE = ["2020-01-01", "2020-12-31"]


EPOCHS               = 10
BATCH_SIZE           = 1
DIFFUSION_STEPS_CHOICE = 20
ema                  = 0.999
learning_rate        = 1e-5
weight_decay         = 1e-6

wind_vars = ["100u", "100v"]

FFT_CUTOFF = 0.05
HR_H       = PX_LAT_HR
HR_W       = PX_LON_HR

SWIN_CONFIG = dict(
    embed_dim=180,depths=(8, 8, 8, 8, 8, 8),num_heads=(6, 6, 6, 6, 6, 6),
    window_size = 8,
)


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


def sequence_encoding(arrays_LR, N_BEFORE, N_AFTER, keep_channels=True):
    """Stack neighbouring time steps of the low-resolution series to get tensors augmented in the time direction (to try to capture advection).

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

    T = arrays_LR.shape[0] # Total number of samples available
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
        seq = np.squeeze(seq, axis=3) # Format the tensor back to the prescribed shape
    return seq


def HR_sequence_slicing(array_HR, n_before=0, n_after=0):
    """Drop the high-resolution frames that have no complete low-resolution sequence.

    Args:
        array_HR: array (T, ...) of high-resolution frames.
        n_before: number of frames before the centre frame.
        n_after: number of frames after the centre frame.
    Returns:
        Array (T - n_before - n_after, ...) aligned in time with the output of sequence_encoding to avoid mismatch between the datasets.
    """
    if n_after > 0:
        array_HR = array_HR[n_before:-n_after, ...]
    elif n_after == 0:
        array_HR = array_HR[n_before:, ...]
    return array_HR


def domain_crop(df, lat_min, lat_max, lon_min, lon_max,variable_name="long"):
    """Crop a dataset to a latitude/longitude window.

    Args:
        df: xarray Dataset or DataArray.
        lat_min, lat_max: southern and northern bounds.
        lon_min, lon_max: western and eastern bounds.
        variable_name: "long" for dimensions latitude/longitude (latitude stored decreasing),
            "short" for dimensions lat/lon (latitude stored increasing) (depends on the data source (ERA5/CERRA)).
    Returns:
        The cropped dataset.
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
            raise ValueError("Unsupported file_type")
        datasets.append(ds)
    return xr.concat(datasets, dim="time").sortby("time")


def _normalizer_stats_from_raw(raw_min, raw_max, raw_mean=None, raw_var=None):
    """Convert raw statistics to the mean and variance of the min-max scaled data.

    Args:
        raw_min, raw_max: minimum and maximum of the channel.
        raw_mean, raw_var: mean and variance of the channel, optional.
    Returns:
        (mean_norm, var_norm). Without raw_mean the mean is -raw_min / scale, without raw_var the variance is 1/36.
    """
    scale = float(raw_max) - float(raw_min)
    if scale <= 0:
        raise ValueError(f"raw_max ({raw_max}) must be > raw_min ({raw_min}).")
    mean_norm = (
        (float(raw_mean) - float(raw_min)) / scale
        if raw_mean is not None
        else (-float(raw_min)) / scale
    )
    var_norm = (
        float(raw_var) / (scale ** 2)
        if raw_var is not None
        else 1.0 / 36.0
    )
    return float(mean_norm), float(var_norm)


def _read_case_statistics(csv_path, n_lr_ch=2, n_hr_ch=2, split="train"):
    """Read per-channel statistics from a statistics CSV.

    Args:
        csv_path: CSV file, either with columns dataset, split, min_u, max_u, ... or with columns type, channel, min, max.
        n_lr_ch, n_hr_ch: number of LR and HR channels to read.
        split: split row used with the first layout.
    Returns:
        {"lr": stats, "hr": stats}, each with arrays mins, maxs, means, vars (None when absent).
    """
    df = pd.read_csv(csv_path)
    if {'dataset', 'split', 'min_u', 'max_u'}.issubset(df.columns):
        df_split = df[df['split'] == split]
        if df_split.empty:
            raise ValueError(f"No rows with split='{split}' in {csv_path}.")
        comp_cols = {
            'u': ('min_u', 'max_u', 'mean_u', 'var_u', 'std_u'),
            'v': ('min_v', 'max_v', 'mean_v', 'var_v', 'std_v'),
        }
        ch_order = ['u', 'v']
        result = {}
        for dataset_label, key, n_ch in [
            ('ERA5',  'lr', n_lr_ch),
            ('CERRA', 'hr', n_hr_ch),
        ]:
            row = df_split[df_split['dataset'] == dataset_label]
            if row.empty:
                raise ValueError(f"No '{dataset_label}' row with split='{split}'.")
            row = row.iloc[0]
            mins, maxs, means, vars_ = [], [], [], []
            for comp in ch_order[:n_ch]:
                min_col, max_col, mean_col, var_col, std_col = comp_cols[comp]
                mins.append(float(row[min_col]))
                maxs.append(float(row[max_col]))
                means.append(float(row[mean_col]) if mean_col in df.columns else None)
                if var_col in df.columns:
                    vars_.append(float(row[var_col]))
                elif std_col in df.columns:
                    vars_.append(float(row[std_col]) ** 2)
                else:
                    vars_.append(None)
            means_arr = (np.array(means, dtype=np.float64)
                         if all(m is not None for m in means) else None)
            vars_arr  = (np.array(vars_, dtype=np.float64)
                         if all(v is not None for v in vars_) else None)
            result[key] = {
                'mins':  np.array(mins, dtype=np.float64),
                'maxs':  np.array(maxs, dtype=np.float64),
                'means': means_arr,
                'vars':  vars_arr,
            }
        return result

    if {'type', 'channel', 'min', 'max'}.issubset(df.columns):
        result = {}
        for key, n_ch in [('lr', n_lr_ch), ('hr', n_hr_ch)]:
            rows = (df[df['type'] == key]
                    .sort_values('channel').iloc[:n_ch].reset_index(drop=True))
            result[key] = {
                'mins':  rows['min'].to_numpy(dtype=np.float64),
                'maxs':  rows['max'].to_numpy(dtype=np.float64),
                'means': (rows['mean'].to_numpy(dtype=np.float64)
                          if 'mean' in df.columns else None),
                'vars':  (rows['variance'].to_numpy(dtype=np.float64)
                          if 'variance' in df.columns
                          else rows['std'].to_numpy(dtype=np.float64) ** 2
                          if 'std' in df.columns else None),
            }
        return result

    raise ValueError(f"Unrecognised statistics CSV format in {csv_path}.")


def _build_normalizer_arrays(raw_stats, n_lr_ch=2, n_hr_ch=2):
    """Build the normaliser mean and variance of the LR channels followed by the HR channels.

    Args:
        raw_stats: output of _read_case_statistics, to not compute them live as datasets are large
        n_lr_ch, n_hr_ch: number of LR and HR channels.
    Returns:
        (means, variances) float32 arrays of length n_lr_ch + n_hr_ch.
    """
    means, variances = [], []
    for key, n_ch in [('lr', n_lr_ch), ('hr', n_hr_ch)]:
        s = raw_stats[key]
        for i in range(n_ch):
            m, v = _normalizer_stats_from_raw(
                raw_min  = s['mins'][i],
                raw_max  = s['maxs'][i],
                raw_mean = s['means'][i] if s['means'] is not None else None,
                raw_var  = s['vars'][i]  if s['vars']  is not None else None,
            )
            means.append(m)
            variances.append(v)
    return (np.array(means,     dtype=np.float32),
            np.array(variances, dtype=np.float32))


class DiffusionModel(keras.Model):
    """Conditional DDPM mapping upsampled ERA5 to CERRA with a Swin2SR denoiser.

    Input tensors have input_frames LR channels followed by output_frames HR channels.
    """

    def __init__(self, image_size_height, input_frames, output_frames,
                 image_size_width, swin_config):
        """Build the normalizer, the Swin2SR denoising network and its EMA copy.

        Args:
            image_size_height, image_size_width: HR grid size.
            input_frames: number of LR channels.
            output_frames: number of HR channels.
            swin_config: dictionary of keyword arguments of get_swin2_denoising_network.
        """
        super().__init__()
        self.normalizer  = layers.Normalization()
        self.network     = get_swin2_denoising_network(
            image_size_height = image_size_height,
            input_frames      = input_frames,
            output_frames     = output_frames,
            image_size_width  = image_size_width,
            **swin_config,
        )
        self.ema_network   = keras.models.clone_model(self.network)
        self.input_frames  = input_frames
        self.output_frames = output_frames

    def compile(self, **kwargs):
        """Compile the model and create the noise and image loss trackers."""
        super().compile(**kwargs)
        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")

    @property
    def metrics(self):
        """Metrics reset by Keras at each epoch: image loss and noise loss."""
        return [self.image_loss_tracker, self.noise_loss_tracker]

    def denormalize(self, images):
        """Invert the normalizer and clip to [0, 1].

        Args:
            images: normalised tensor.
        Returns:
            Tensor in the min-max scaled range.
        """
        images = self.normalizer.mean + images * self.normalizer.variance ** 0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        """Noise and signal rates for a diffusion time (angle interpolated between the signal-rate bounds).

        Args:
            diffusion_times: tensor (B, 1, 1, 1) in [0, 1].
        Returns:
            (noise_rates, signal_rates), tensors (B, 1, 1, 1).
        """
        start_angle = tf.acos(max_signal_rate)
        end_angle   = tf.acos(min_signal_rate)
        angles      = start_angle + diffusion_times * (end_angle - start_angle)
        return tf.sin(angles), tf.cos(angles)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        """Predict the noise and the clean image from noisy inputs.

        Args:
            noisy_images: tensor (B, H, W, input_frames + output_frames).
            noise_rates, signal_rates: tensors (B, 1, 1, 1).
            training: use the network when True, the EMA network otherwise.
        Returns:
            (pred_noises, pred_images), tensors (B, H, W, output_frames).
        """
        # Use network during training, ema_network during inference.
        network     = self.network if training else self.ema_network
        pred_noises = network([noisy_images, noise_rates ** 2], training=training)
        pred_images = (noisy_images[:, :, :, -self.output_frames:]
                    - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        """Run the reverse diffusion from an initial noisy state.

        Args:
            initial_noise: tensor (B, H, W, input_frames + output_frames).
            diffusion_steps: number of denoising steps.
        Returns:
            Predicted HR images (B, H, W, output_frames).
        """
        num_images = initial_noise.shape[0]
        step_size  = 1.0 / diffusion_steps
        past       = initial_noise[:, :, :, :-self.output_frames]
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images    = next_noisy_images
            diffusion_times = (tf.ones((num_images, 1, 1, 1),
                                        dtype=initial_noise.dtype)
                               - step * step_size)
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images  = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False)
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                diffusion_times - step_size)
            next_noisy_frames = (next_signal_rates * pred_images
                                 + next_noise_rates * pred_noises)
            next_noisy_images = tf.concat([past, next_noisy_frames], axis=-1)
        return pred_images

    def generate2(self, images, diffusion_steps):
        """Generate HR fields from LR inputs by reverse diffusion.

        The start state mixes the upsampled LR field (weight min_signal_rate) with Gaussian noise.

        Args:
            images: tensor (B, H, W, input_frames + output_frames); only the LR channels are used.
            diffusion_steps: number of denoising steps.
        Returns:
            Tensor (B, H, W, input_frames + output_frames): LR channels followed by the generated
            HR channels, in the min-max scaled range.
        """
        images = tf.cast(images, tf.float32)
        images = self.normalizer(images, training=False)

        lr_up       = images[:, :, :, :self.input_frames]
        lr_for_init = lr_up[:, :, :, -self.output_frames:]

        lr_weight    = tf.cast(min_signal_rate, dtype=images.dtype)
        noise_weight = tf.sqrt(1.0 - lr_weight ** 2)

        rand_noise = tf.random.normal(
            shape=(images.shape[0], images.shape[1],
                   images.shape[2], self.output_frames),
            dtype=images.dtype)

        initial_hr  = lr_weight * lr_for_init + noise_weight * rand_noise
        images_init = tf.concat([lr_up, initial_hr], axis=-1)

        generated_images = self.reverse_diffusion(images_init, diffusion_steps)
        generated_images = self.denormalize(
            tf.concat([lr_up, generated_images], axis=-1))
        return generated_images

    @tf.function(jit_compile=True)
    def train_step(self, images):
        """One training step: noise the target, predict the noise, update the network and its EMA copy.

        Args:
            images: batch (B, H, W, input_frames + output_frames).
        Returns:
            Dictionary of the image and noise losses.
        """
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(
            shape=(images.shape[0], PX_LAT_HR, PX_LON_HR, self.output_frames),
            dtype=images.dtype)
        diffusion_times = tf.random.uniform(
            shape=(images.shape[0], 1, 1, 1), minval=0.0, maxval=1.0,
            dtype=images.dtype)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        target       = images[:, :, :, -self.output_frames:]
        noisy_images = signal_rates * target + noise_rates * noises
        noise_two    = tf.concat(
            [images[:, :, :, :-self.output_frames], noisy_images], axis=-1)
        with tf.GradientTape() as tape:
            pred_noises, pred_images = self.denoise(
                noise_two, noise_rates, signal_rates, training=True)
            noise_loss = self.loss(noises, pred_noises)
            image_loss = self.loss(target, pred_images)
        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.network.trainable_weights))
        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)
        return {m.name: m.result() for m in self.metrics}

    @tf.function(jit_compile=True)
    def test_step(self, images):
        """One evaluation step with the EMA network.

        Args:
            images: batch (B, H, W, input_frames + output_frames).
        Returns:
            Dictionary of the image and noise losses.
        """
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(
            shape=(images.shape[0], PX_LAT_HR, PX_LON_HR, self.output_frames))
        diffusion_times = tf.random.uniform(
            shape=(images.shape[0], 1, 1, 1), minval=0.0, maxval=1.0)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        target       = images[:, :, :, -self.output_frames:]
        noisy_images = signal_rates * target + noise_rates * noises
        noise_two    = tf.concat(
            [images[:, :, :, :-self.output_frames], noisy_images], axis=-1)
        pred_noises, pred_images = self.denoise(
            noise_two, noise_rates, signal_rates, training=False)
        self.noise_loss_tracker.update_state(self.loss(noises, pred_noises))
        self.image_loss_tracker.update_state(self.loss(target, pred_images))
        return {m.name: m.result() for m in self.metrics}

    def plotter(self, epoch, logs):
        """Placeholder for an epoch-end plotting callback."""
        pass


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {len(gpus)} - {gpus[0].name}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        chosen_gpu = '/device:GPU:0'
        print(f"Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")
        chosen_gpu = '/CPU:0'
else:
    print("No GPU detected. Using CPU")
    chosen_gpu = '/CPU:0'


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
        case="", uv_comp=True, rolling_window=8, data_is_ordered=True):
    """Plot per-sample MSE, PSNR and SSIM over time and save the series.

    Writes temporal_metrics.pdf and temporal_metrics.csv in results_dir.

    Args:
        mse_per_sample, psnr_per_sample, ssim_per_sample: arrays (N_test,).
        n_train, n_val: sizes of the training and validation sets, used as sample index offset.
        results_dir: output directory.
        case: label used in the title.
        uv_comp: True when u and v are averaged.
        rolling_window: window of the rolling mean in samples.
        data_is_ordered: False adds a warning that the test set is shuffled.
    """
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker

    n_test    = len(mse_per_sample)
    all_times = _build_year_times()
    n_full    = len(all_times)
    offset    = n_train + n_val

    test_times = (all_times[0:n_test] if n_test <= n_full
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
    ch_label  = "U and V averaged" if uv_comp else "(u only)"
    raw_kw    = dict(color="#aec6e8", alpha=0.55, linewidth=0.6, zorder=1)
    smooth_kw = dict(color="#1f6fab", linewidth=1.8, zorder=3)
    mean_kw   = dict(color="black",  linewidth=1.2, linestyle="--", zorder=4)

    panels = [
        (mse_per_sample,  mse_smooth,
         f"Swin2SR-Diffusion test metrics — {case} — {ch_label}", "MSE"),
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

    ax_bot = axes[-1]
    ax_bot.set_xlabel("Time (Year 2024 — 3-hourly samples)", fontsize=13)
    ax_bot.xaxis.set_major_locator(mdates.MonthLocator())
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_bot.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
    fig.autofmt_xdate(rotation=35, ha="right")
    fig.suptitle(f"Per-sample test metrics over time  [n_test={n_test}]",
                 fontsize=14, y=1.01)
    plt.tight_layout()

    out_png = os.path.join(results_dir, "temporal_metrics.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved temporal metrics plot → {out_png}")

    out_csv = os.path.join(results_dir, "temporal_metrics.csv")
    pd.DataFrame({
        "datetime":     test_times,
        "sample_index": np.arange(offset, offset + n_test),
        "mse_scaled":   mse_per_sample,
        "psnr_db":      psnr_per_sample,
        "ssim":         ssim_per_sample,
        "mse_rolling":  mse_smooth,
        "psnr_rolling": psnr_smooth,
        "ssim_rolling": ssim_smooth,
    }).to_csv(out_csv, index=False)
    print(f"Saved temporal metrics CSV  → {out_csv}")


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
    ws_vmin = float(min(ws_gt.min(), ws_pred.min(), ws_lr.min()))
    err_max = float(max(err_abs.max(), 0.01))

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
    _imshow(axes[0, 0], ws_lr,  "ERA5 bilinear (LR)", ext_lr, "viridis", ws_vmin, ws_vmax)
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

    Steps: optional raw data conversion or memmap checks, model creation, optional pretrained
    weights, training, loss curves, test evaluation, figures and CSV files, then release of the
    GPU memory.

    Args:
        CASE: key of TRAIN_DURATION.
        run_start_str: start time of the run (ISO string) written in the timing CSV.
        timer: SectionTimer collecting preprocessing, training and sampling times.
        total_start: time.perf_counter value at the start of the run.
    """
    print(f"\n{'='*70}")
    print(f"  Starting Swin2SR-Diffusion case: {CASE}")
    print(f"{'='*70}\n")

    if MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")

    RESULTS_DIR = (
        os.path.join(BASE_DIR, f"Results_ML/Swin_Diff-60eps/{CASE}")
        if not CONVERTING_DATA
        else os.path.join(BASE_DIR, "Results_ML/Swin_Diff_Debug")
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    lat_ERA5_path  = os.path.join(RESULTS_DIR, "lat_LR.npy")
    lon_ERA5_path  = os.path.join(RESULTS_DIR, "lon_LR.npy")
    lat_CERRA_path = os.path.join(RESULTS_DIR, "lat_HR.npy")
    lon_CERRA_path = os.path.join(RESULTS_DIR, "lon_HR.npy")

    with tf.device(chosen_gpu):

        # PLOT_ONLY path - reload saved arrays and skip training/eval
        if PLOT_ONLY:
            pred_path = os.path.join(RESULTS_DIR, f"pred_arrays_{CASE}.npz")
            if not os.path.exists(pred_path):
                raise FileNotFoundError(
                    f"PLOT_ONLY=True but no saved arrays found at {pred_path}.\n"
                    f"Run with PLOT_ONLY=False first to generate predictions.")
            print(f"[PLOT_ONLY] Loading saved arrays from {pred_path}")
            arrs = np.load(pred_path)
            plot_fft_analysis(
                results_dir   = RESULTS_DIR,
                hr_pred       = arrs["hr_pred_unscaled"],
                hr_true       = (arrs["hr_true_unscaled"] if "hr_true_unscaled" in arrs.files else arrs["hr_batch_unscaled"]),
                lr_input      = (arrs["lr_unscaled"] if "lr_unscaled" in arrs.files else arrs["lr_batch_unscaled"]),
                cutoff        = FFT_CUTOFF,
                n_bins        = 128,
                n_samples_avg = 2920,
                case          = CASE,
            )
            return

        if CONVERTING_DATA:
            timer.start("preprocessing")
            print("Starting data extraction from ERA5 GRIB files ...")
            try:
                ds_wind_ERA5  = load_era5_variables(
                    grib_files_ERA5,  wind_vars, file_type="cfgrib")
                ds_wind_CERRA = load_era5_variables(
                    grib_files_CERRA, wind_vars, file_type="h5netcdf")
            except OSError as e:
                print(f"Error loading files: {e}"); raise

            ds_wind_ERA5  = time_crop(ds_wind_ERA5,  TIME_RANGE[0], TIME_RANGE[1])
            ds_wind_CERRA = time_crop(ds_wind_CERRA, TIME_RANGE[0], TIME_RANGE[1])
            ds_wind_ERA5  = domain_crop(ds_wind_ERA5,
                                        LAT_COORDS[0], LAT_COORDS[1],
                                        LON_COORDS[0], LON_COORDS[1],
                                        variable_name="long")
            ds_wind_CERRA = domain_crop(ds_wind_CERRA,
                                        LAT_COORDS_CERRA[0], LAT_COORDS_CERRA[1],
                                        LON_COORDS_CERRA[0], LON_COORDS_CERRA[1],
                                        variable_name="short")

            ds_wind_ERA5  = ds_wind_ERA5.isel(
                time=ds_wind_ERA5["time"].dt.hour.isin([0,3,6,9,12,15,18,21]))
            ds_wind_CERRA = wind_speed_to_uv(ds_wind_CERRA)
            ds_wind_CERRA = ds_wind_CERRA.sel(lat=slice(None, None, -1))

            u_era5  = ds_wind_ERA5["u100"].values.astype(np.float32)
            v_era5  = ds_wind_ERA5["v100"].values.astype(np.float32)
            u_cerra = ds_wind_CERRA["u100"].values.astype(np.float32)
            v_cerra = ds_wind_CERRA["v100"].values.astype(np.float32)

            arrays_np_lr  = np.stack([u_era5, v_era5],   axis=-1)
            arrays_np_hr  = np.stack([u_cerra, v_cerra], axis=-1)
            arrays_seq_lr = sequence_encoding(arrays_np_lr, N_BEFORE, N_AFTER,
                                              keep_channels=True)
            arrays_np_hr  = HR_sequence_slicing(arrays_np_hr, N_BEFORE, N_AFTER)

            raw_stats = {
                'lr': {'mins': np.min(arrays_seq_lr,  axis=(0,1,2)),
                       'maxs': np.max(arrays_seq_lr,  axis=(0,1,2)),
                       'means': np.mean(arrays_seq_lr, axis=(0,1,2)),
                       'vars':  np.var(arrays_seq_lr,  axis=(0,1,2))},
                'hr': {'mins': np.min(arrays_np_hr,   axis=(0,1,2)),
                       'maxs': np.max(arrays_np_hr,   axis=(0,1,2)),
                       'means': np.mean(arrays_np_hr, axis=(0,1,2)),
                       'vars':  np.var(arrays_np_hr,  axis=(0,1,2))},
            }

            lat_HR = ds_wind_CERRA["lat"].values
            lon_HR = ds_wind_CERRA["lon"].values
            lat_lr = ds_wind_ERA5["latitude"].values
            lon_lr = ds_wind_ERA5["longitude"].values
            np.savetxt(lat_ERA5_path,  lat_lr)
            np.savetxt(lon_ERA5_path,  lon_lr)
            np.savetxt(lat_CERRA_path, lat_HR)
            np.savetxt(lon_CERRA_path, lon_HR)
            timer.stop("preprocessing")

            lr_mmap = np.memmap(os.path.join(RESULTS_DIR, "arrays_seq_lr.dat"),
                                dtype='float32', mode='w+',
                                shape=arrays_seq_lr.shape)
            lr_mmap[:] = arrays_seq_lr[:]; lr_mmap.flush()
            np.save(os.path.join(RESULTS_DIR, "lr_shape.npy"), arrays_seq_lr.shape)
            del arrays_seq_lr

            hr_mmap = np.memmap(os.path.join(RESULTS_DIR, "arrays_np_hr.dat"),
                                dtype='float32', mode='w+',
                                shape=arrays_np_hr.shape)
            hr_mmap[:] = arrays_np_hr[:]; hr_mmap.flush()
            np.save(os.path.join(RESULTS_DIR, "hr_shape.npy"), arrays_np_hr.shape)
            del arrays_np_hr, arrays_np_lr, ds_wind_ERA5, ds_wind_CERRA
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
                        f"Prepared memmap not found: {required}\n"
                        f"Run prepare_training_data.py with CASES=['{CASE}'] first.")

            coords_dir = os.path.join(SHARED_DIR, "coords")
            lat_HR = np.loadtxt(os.path.join(coords_dir, "lat_HR.npy"))
            lon_HR = np.loadtxt(os.path.join(coords_dir, "lon_HR.npy"))
            lat_lr = np.loadtxt(os.path.join(coords_dir, "lat_LR.npy"))
            lon_lr = np.loadtxt(os.path.join(coords_dir, "lon_LR.npy"))

            def _sidecar(path):
                """Read the shape and dtype from the sidecar file of a memmap."""
                lines = open(path.replace(".npy", "_shape.txt")).read().strip().splitlines()
                shape = tuple(int(s) for s in lines[0].split(","))
                dtype = np.dtype(lines[1]) if len(lines) > 1 else np.dtype("float32")
                return shape, dtype

            lr_train_shape, _ = _sidecar(os.path.join(case_dir,   "train", "ERA5_train.npy"))
            hr_train_shape, _ = _sidecar(os.path.join(case_dir,   "train", "CERRA_train.npy"))
            lr_val_shape,   _ = _sidecar(os.path.join(SHARED_DIR, "val",   "ERA5_val.npy"))
            hr_val_shape,   _ = _sidecar(os.path.join(SHARED_DIR, "val",   "CERRA_val.npy"))
            lr_test_shape,  _ = _sidecar(os.path.join(SHARED_DIR, "test",  "ERA5_test.npy"))
            hr_test_shape,  _ = _sidecar(os.path.join(SHARED_DIR, "test",  "CERRA_test.npy"))

            expected_lr_ch = (N_BEFORE + 1 + N_AFTER) * (2 if UV_COMP else 1)
            expected_hr_ch = 2 if UV_COMP else 1

            print("\n  Tensor shape verification:")
            for split_name, lr_sh, hr_sh in [
                ("train", lr_train_shape, hr_train_shape),
                ("val",   lr_val_shape,   hr_val_shape),
                ("test",  lr_test_shape,  hr_test_shape),
            ]:
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

            timer.stop("preprocessing")

        # Build and compile the model
        print("Swin2SR-Diffusion model creation ...")
        print(f"  SWIN_CONFIG = {SWIN_CONFIG}")

        model = DiffusionModel(
            image_size_height = PX_LAT_HR,
            input_frames      = 2,
            output_frames     = 2,
            image_size_width  = PX_LON_HR,
            swin_config       = SWIN_CONFIG,
        )
        model.network.summary()
        print("Swin2SR-Diffusion model compiling ...")

        PRETRAINED_BASE = os.path.join(BASE_DIR, "Results_ML/Swin_Diff-50eps")
        reuse_model = os.path.join(
            PRETRAINED_BASE, CASE, "Diffusion_Checkpoint"
        )
        if not os.path.exists(reuse_model + ".weights.h5"):
            print(f"  [INFO] No pretrained weights found for case '{CASE}' — training from scratch.")
            reuse_model = None
        else:
            print(f"  [INFO] Loading pretrained weights for case '{CASE}' from:\n"
                f"         {reuse_model}")
            model.network.load_weights(reuse_model + ".weights.h5")
            model.ema_network.load_weights(reuse_model + "_ema.weights.h5")

        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=learning_rate, weight_decay=weight_decay),
            loss=keras.losses.MeanSquaredError(),
            steps_per_execution = 2,
        )

        # Compute the normaliser statistics
        print("Building diffusion normaliser statistics ...")
        if CONVERTING_DATA:
            normalizer_mean, normalizer_variance = _build_normalizer_arrays(
                raw_stats, n_lr_ch=2, n_hr_ch=2)
        else:
            stats_csv_path = os.path.join(case_dir, f"statistics_{CASE}.csv")
            if not os.path.exists(stats_csv_path):
                raise FileNotFoundError(
                    f"Statistics CSV not found: {stats_csv_path}\n"
                    f"Run prepare_training_data.py with CASES=['{CASE}'] first.")
            raw_stats = _read_case_statistics(
                stats_csv_path, n_lr_ch=2, n_hr_ch=2, split="train")
            normalizer_mean, normalizer_variance = _build_normalizer_arrays(
                raw_stats, n_lr_ch=2, n_hr_ch=2)

        print(f"  normalizer_mean     = {normalizer_mean}")
        print(f"  normalizer_variance = {normalizer_variance}")

        # Training
        timer.start("training")

        if CONVERTING_DATA:
            raise NotImplementedError(
                "ml_diffusion_pipeline_streamed requires pre-split memmaps.\n"
                "Set CONVERTING_DATA=False and run prepare_training_data.py first.")

        shuf_buffer = lr_train_shape[0]
        print(f"shuffle_buffer = {shuf_buffer}")


        model, history, datasets, minmax_stats = ml_diffusion_pipeline_streamed(
            case_dir                = case_dir,
            shared_dir              = SHARED_DIR,
            csv_path                = stats_csv_path,
            scale_factor            = RF,
            batch_size              = BATCH_SIZE,
            shuffle_buffer          = shuf_buffer,
            epochs                  = EPOCHS,
            early_stopping_patience = 10,
            normalizer_mean         = normalizer_mean,
            normalizer_variance     = normalizer_variance,
            model_provided          = model,
            load_pretrained_path    = reuse_model,
            results_dir             = RESULTS_DIR,
            steps_per_execution = 2,
        )

        # Convert the min/max statistics to arrays
        hr_mins = np.array([float(v) for v in minmax_stats["hr"][0]], dtype=np.float32)
        hr_maxs = np.array([float(v) for v in minmax_stats["hr"][1]], dtype=np.float32)
        lr_mins = np.array([float(v) for v in minmax_stats["lr"][0]], dtype=np.float32)
        lr_maxs = np.array([float(v) for v in minmax_stats["lr"][1]], dtype=np.float32)

        print("Training complete.")
        timer.stop("training")
        np.save(os.path.join(RESULTS_DIR, "training_history.npy"), history.history)

        h = history.history
        if h.get('val_n_loss'):

            best_epoch = int(np.argmin(h['val_n_loss']))
            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                    gridspec_kw={"hspace": 0.08})
            axes[0].plot(h['n_loss'],     color='steelblue', linewidth=1.5,
                        label='train noise loss')
            axes[0].plot(h['val_n_loss'], color='tomato',    linewidth=1.5,
                        linestyle='--', label='val noise loss')
            axes[0].axvline(best_epoch, color='black', linewidth=0.9, linestyle=':',
                            label=f'best epoch {best_epoch+1}  '
                            f'(val={h["val_n_loss"][best_epoch]:.6f})')
            axes[0].set_ylabel('Noise loss (optimised quantity)', fontsize=12)
            axes[0].legend(fontsize=10, framealpha=0.7)
            axes[0].set_title(f'Swin2SR-DiffusionModel training curves — {CASE}',
                            fontsize=20)
            axes[1].plot(h['i_loss'],     color='steelblue', linewidth=1.5,
                        label='train image loss')
            axes[1].plot(h['val_i_loss'], color='tomato',    linewidth=1.5,
                        linestyle='--', label='val image loss')
            axes[1].axvline(best_epoch, color='black', linewidth=0.9, linestyle=':')
            axes[1].set_ylabel('Image loss [MSE pixels]', fontsize=12)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].legend(fontsize=10, framealpha=0.7)
            plt.tight_layout()
            fig_path = os.path.join(RESULTS_DIR, 'loss_history.pdf')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved training curves → {fig_path}")

        else:
            print("  [INFO] EPOCHS=0 — skipping loss curve plot.")

        # Evaluate on the test set

        timer.start("sampling")
        print(f"Computing metrics on test dataset [{CASE}]:")
        print("XLA + vectorized evaluation enabled")

        @tf.function(jit_compile=True)
        def run_model(model, x):
            """Generate predictions for a batch (compiled)."""
            return model.generate2(x, DIFFUSION_STEPS_CHOICE)

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

        for sample_batched in datasets['test_ds'].take(2):
            y_images = run_model(model, sample_batched)
            y_pred   = y_images[..., -2:]
            y_true   = tf.cast(sample_batched[..., -2:], tf.float32)
            mse, psnr, ssim, mse_s, psnr_s, ssim_s = compute_metrics(y_true, y_pred)
            mse_all.append(mse);  psnr_all.append(psnr);  ssim_all.append(ssim)
            mse_per_sample_all.append(mse_s)
            psnr_per_sample_all.append(psnr_s)
            ssim_per_sample_all.append(ssim_s)
            # Move each batch to CPU memory so GPU memory does not accumulate
            hr_pred_list.append(y_pred.numpy())
            hr_true_list.append(y_true.numpy())
            lr_input_list.append(sample_batched[..., :2].numpy())

        mse_mean  = tf.reduce_mean(mse_all).numpy()
        psnr_mean = tf.reduce_mean(psnr_all).numpy()
        ssim_mean = tf.reduce_mean(ssim_all).numpy()
        mse_per_sample  = tf.concat(mse_per_sample_all,  axis=0).numpy()
        psnr_per_sample = tf.concat(psnr_per_sample_all, axis=0).numpy()
        ssim_per_sample = tf.concat(ssim_per_sample_all, axis=0).numpy()

        print(f"\nMean MSE (scaled):  {mse_mean:.6f}")
        print(f"Mean PSNR (scaled): {psnr_mean:.3f} dB")
        print(f"Mean SSIM (scaled): {ssim_mean:.4f}")

        # Concatenate on CPU - stays in RAM, no GPU allocation
        hr_pred_all  = np.concatenate(hr_pred_list,  axis=0)
        hr_true_all  = np.concatenate(hr_true_list,  axis=0)
        lr_input_all = np.concatenate(lr_input_list, axis=0)

        hr_pred_unscaled = hr_pred_all  * (hr_maxs - hr_mins) + hr_mins
        hr_true_unscaled = hr_true_all  * (hr_maxs - hr_mins) + hr_mins
        lr_unscaled      = lr_input_all * (lr_maxs - lr_mins) + lr_mins

        mse_unscaled     = float(np.mean((hr_pred_unscaled - hr_true_unscaled) ** 2))
        max_val_unscaled = float(hr_true_unscaled.max())
        psnr_unscaled    = 10.0 * np.log10(max_val_unscaled ** 2 / (mse_unscaled + 1e-12))
        n_ssim           = min(200, hr_true_unscaled.shape[0])
        ssim_unscaled    = float(tf.reduce_mean(tf.image.ssim(
            tf.constant(hr_true_unscaled[:n_ssim]),
            tf.constant(hr_pred_unscaled[:n_ssim]),
            max_val=max_val_unscaled, filter_size=7,
        )).numpy())

        print(f"\n--- UN-SCALED METRICS (SSIM over first {n_ssim} samples) ---")
        print(f"MSE  (unscaled): {mse_unscaled:.6f}")
        print(f"PSNR (unscaled): {psnr_unscaled:.3f} dB")
        print(f"SSIM (unscaled): {ssim_unscaled:.4f}")

        timer.stop("sampling")

        n_save = 200
        pred_save_path = os.path.join(RESULTS_DIR, f"pred_arrays_{CASE}.npz")
        np.savez_compressed(pred_save_path,
                            hr_pred_unscaled = hr_pred_unscaled[:n_save],
                            hr_true_unscaled = hr_true_unscaled[:n_save],
                            lr_unscaled      = lr_unscaled[:n_save])
        print(f"Saved prediction arrays → {pred_save_path}  (first {n_save} samples)")

        def _np(x):
            """Convert a tensor to numpy."""
            return x.numpy() if isinstance(x, tf.Tensor) else np.asarray(x)

        plot_comparison_diffusion(
            _np(lr_unscaled), _np(hr_pred_unscaled), _np(hr_true_unscaled),
            ssim_mean_u=None, ssim_std_u=None,
            ssim_mean_v=None, ssim_std_v=None,
            lat_hr=lat_HR, lon_hr=lon_HR,
            lat_lr=lat_lr, lon_lr=lon_lr,
            index=0, results_dir=RESULTS_DIR,
        )

        plot_temporal_metrics(
            mse_per_sample  = mse_per_sample.astype(np.float32),
            psnr_per_sample = psnr_per_sample.astype(np.float32),
            ssim_per_sample = ssim_per_sample.astype(np.float32),
            n_train         = datasets["n_train"],
            n_val           = datasets["n_val"],
            results_dir     = RESULTS_DIR,
            case            = CASE,
            uv_comp         = UV_COMP,
            rolling_window  = 8,
            data_is_ordered = not CONVERTING_DATA,
        )

        plot_fft_analysis(
            results_dir   = RESULTS_DIR,
            hr_pred       = _np(hr_pred_unscaled),
            hr_true       = _np(hr_true_unscaled),
            lr_input      = _np(lr_unscaled),
            cutoff        = FFT_CUTOFF,
            n_bins        = 128,
            n_samples_avg = 2920,
            case          = CASE,
        )

        plot_comparison_wind_speed(
            lr          = _np(lr_unscaled),
            pred        = _np(hr_pred_unscaled),
            gt          = _np(hr_true_unscaled),
            lat_hr=lat_HR, lon_hr=lon_HR, lat_lr=lat_lr, lon_lr=lon_lr,
            index=0, results_dir=RESULTS_DIR, case=CASE,
        )

        plot_quiver_wind(
            pred          = _np(hr_pred_unscaled),
            gt            = _np(hr_true_unscaled),
            lat_hr=lat_HR, lon_hr=lon_HR,
            index=0, results_dir=RESULTS_DIR, quiver_stride=12, case=CASE,
        )

        # Append the test metrics to test_metrics.csv
        metrics_path = os.path.join(RESULTS_DIR, "test_metrics.csv")
        if not os.path.exists(metrics_path):
            with open(metrics_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "run_time", "case", "val_loss_mse_scaled",
                    "test_mse_scaled", "test_psnr", "test_ssim", "epochs",
                    "embed_dim", "n_stages", "window_size",
                ])
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(), CASE,
                round(mse_mean, 4), round(mse_mean, 4),
                round(psnr_mean, 4), round(ssim_mean, 4), int(EPOCHS),
                SWIN_CONFIG["embed_dim"],
                len(SWIN_CONFIG["depths"]),
                SWIN_CONFIG["window_size"],
            ])
        print(f"Saved test metrics → {metrics_path}")

        # Append the run times to computation_time_runs.csv
        total_time  = time.perf_counter() - total_start
        csv_file    = os.path.join(RESULTS_DIR, "computation_time_runs.csv")
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["run_start_time", "case",
                                 "preprocessing_seconds", "training_seconds",
                                 "sampling_seconds", "total_seconds"])
            writer.writerow([
                run_start_str,
                CASE if not CONVERTING_DATA else "debug",
                f"{timer.times.get('preprocessing', 0.0):.4f}",
                f"{timer.times.get('training',      0.0):.4f}",
                f"{timer.times.get('sampling',      0.0):.4f}",
                f"{total_time:.4f}",
            ])
        print(f"[{CASE}] Total elapsed: {total_time:.1f} s")

    del model, datasets, minmax_stats
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

    print(f"Multi-case Swin2SR-Diffusion run — {len(CASES)} case(s): {CASES}")
    print(f"PLOT_ONLY = {PLOT_ONLY}")
    print(f"SWIN_CONFIG = {SWIN_CONFIG}")
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
