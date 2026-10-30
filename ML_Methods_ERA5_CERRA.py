"""Data pipelines, evaluation helpers and comparison plots shared by the downscaling scripts.

tf_pipeline_from_prepared: tf.data datasets from the memmaps written by prepare_training_data.py.
tf_pipeline: tf.data datasets from raw memmap files with a train/val/test split by ratio.
ml_diffusion_pipeline_streamed: training pipeline of the diffusion models.
plot_comparison_transformer, plot_comparison_diffusion: LR / SR / HR comparison figures.
Author: Marion Falkenauer (with the help of Claude)
"""
import os
import csv
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tensorflow import keras


def _read_sidecar(npy_path: str):
    """Read the array shape and dtype stored next to a .npy memmap.

    Args:
        npy_path: path of the .npy file; the sidecar is the same path ending in _shape.txt.
    Returns:
        (shape, dtype) with shape a tuple of ints and dtype a numpy dtype (float32 if not stored).
    """
    sidecar = npy_path.replace(".npy", "_shape.txt")
    lines   = open(sidecar).read().strip().splitlines()
    shape   = tuple(int(s) for s in lines[0].split(","))
    dtype   = np.dtype(lines[1]) if len(lines) > 1 else np.dtype("float32")
    return shape, dtype


@tf.keras.utils.register_keras_serializable()
def psnr_metric(y_true, y_pred):
    """Mean PSNR over a batch for data scaled to [0, 1].

    Args:
        y_true: reference tensor (B, H, W) or (B, H, W, C).
        y_pred: predicted tensor with the same shape.
    Returns:
        Scalar tensor, mean PSNR in dB with max_val=1.
    """
    for arr in (y_true, y_pred):
        pass
    if y_true.shape[-1] == 1:
        y_true = tf.squeeze(y_true, axis=-1)
    if y_true.shape[-1] == 1:
        y_true = tf.squeeze(y_true, axis=-1)
    if y_pred.shape[-1] == 1:
        y_pred = tf.squeeze(y_pred, axis=-1)
    if y_pred.shape[-1] == 1:
        y_pred = tf.squeeze(y_pred, axis=-1)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if len(y_true.shape) == 3:
        y_true = tf.expand_dims(y_true, axis=-1)
        y_pred = tf.expand_dims(y_pred, axis=-1)
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


@tf.keras.utils.register_keras_serializable()
def ssim_metric(y_true, y_pred):
    """Mean SSIM over a batch for data scaled to [0, 1].

    Args:
        y_true: reference tensor (B, H, W) or (B, H, W, C).
        y_pred: predicted tensor with the same shape.
    Returns:
        Scalar tensor, mean SSIM with max_val=1.
    """
    if y_true.shape[-1] == 1:
        y_true = tf.squeeze(y_true, axis=-1)
    if y_true.shape[-1] == 1:
        y_true = tf.squeeze(y_true, axis=-1)
    if y_pred.shape[-1] == 1:
        y_pred = tf.squeeze(y_pred, axis=-1)
    if y_pred.shape[-1] == 1:
        y_pred = tf.squeeze(y_pred, axis=-1)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if len(y_true.shape) == 3:
        y_true = tf.expand_dims(y_true, axis=-1)
        y_pred = tf.expand_dims(y_pred, axis=-1)
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def test_diffusion_model(test_dataset, model, scaler_X, scaler_y, history,
                         model_provided=None, results_dir=None):
    """Evaluate a trained diffusion model on a test dataset.

    Args:
        test_dataset: tf.data.Dataset of (B, H, W, 4) tensors (2 LR channels, 2 HR channels).
        model: DiffusionModel with generate2.
        scaler_X: (mins, maxs) of the LR channels.
        scaler_y: (mins, maxs) of the HR channels.
        history: Keras History of the training run.
        model_provided: "Diffusion" to append the test losses to test_metrics_new.csv.
        results_dir: output directory (required).
    Returns:
        (y_pred_unscaled, y_true_unscaled, x_unscaled) of the last test batch, in physical units.
    """
    if results_dir is None:
        raise ValueError(
            "results_dir must be passed explicitly to test_diffusion_model().")

    print("Test dataset")
    test_metrics = model.evaluate(test_dataset, verbose=0)
    metrics_path = os.path.join(results_dir, "test_metrics_new.csv")

    if model_provided == "Diffusion":
        test_i_loss, test_n_loss = test_metrics
        print(f"Test image loss: {test_i_loss:.3f}  noise loss: {test_n_loss:.3f}")
        epochs_trained = len(history.history.get("n_loss", []))
        if not os.path.exists(metrics_path):
            with open(metrics_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "run_time", "test_image_loss",
                    "test_noise_loss", "test_psnr", "test_ssim", "epochs"])

    lr_mins, lr_maxs = scaler_X
    hr_mins, hr_maxs = scaler_y
    hr_mins = np.array([float(m) for m in hr_mins]).reshape((1, 1, 1, -1))
    hr_maxs = np.array([float(m) for m in hr_maxs]).reshape((1, 1, 1, -1))
    lr_mins = np.array([float(m) for m in lr_mins]).reshape((1, 1, 1, -1))
    lr_maxs = np.array([float(m) for m in lr_maxs]).reshape((1, 1, 1, -1))

    total_mse = total_mse_scaled = total_psnr = total_ssim = 0.0
    n_batches = 0
    for sample_batched in test_dataset:
        y_images = model.generate2(sample_batched, 10)
        y_pred  = tf.cast(y_images[:, :, :, -2:], tf.float32)
        y_batch = tf.cast(sample_batched[:, :, :, -2:], tf.float32)
        total_mse_scaled += tf.reduce_mean((y_pred - y_batch) ** 2).numpy()
        total_psnr += psnr_metric(y_batch, y_pred).numpy()
        total_ssim += ssim_metric(y_batch, y_pred).numpy()
        y_pred_unscaled = y_pred.numpy()  * (hr_maxs - hr_mins) + hr_mins
        y_true_unscaled = y_batch.numpy() * (hr_maxs - hr_mins) + hr_mins
        x_unscaled = sample_batched[:, :, :, :2].numpy() * (lr_maxs - lr_mins) + lr_mins
        total_mse += np.mean((y_pred_unscaled - y_true_unscaled) ** 2)
        n_batches += 1

    mse   = total_mse        / n_batches
    psnr  = total_psnr       / n_batches
    ssim  = total_ssim       / n_batches

    print(f"\n--- EVALUATION ---\nMSE={mse:.6f}  PSNR={float(psnr):.6f}  SSIM={float(ssim):.6f}")

    if model_provided == "Diffusion":
        new_row = [datetime.now().isoformat(),
                   round(test_i_loss, 6), round(test_n_loss, 6),
                   round(float(psnr), 6), round(float(ssim), 6),
                   int(epochs_trained)]
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(new_row)

    return y_pred_unscaled, y_true_unscaled, x_unscaled


def plot_comparison_transformer(
    lr, pred, hr,
    ssim_mean_u, ssim_std_u,
    ssim_mean_v, ssim_std_v,
    lat_hr, lon_hr,
    lat_lr, lon_lr,
    index=0,
    results_dir="results"
):
    """Save the LR / SR / HR comparison and the spatial SSIM figures of one sample.

    Writes comparison_sample{index}.png (u100 and v100 rows) and SSIM_Viewer.png.

    Args:
        lr, pred, hr: arrays (N, H, W, 2) of inputs, predictions and targets.
        ssim_mean_u, ssim_std_u, ssim_mean_v, ssim_std_v: SSIM maps (H_hr, W_hr).
        lat_hr, lon_hr, lat_lr, lon_lr: coordinate vectors of the HR and LR grids.
        index: sample index to plot.
        results_dir: output directory.
    """
    os.makedirs(results_dir, exist_ok=True)

    lr_img  = lr[index]
    sr_img  = pred[index]
    hr_img  = hr[index]
    # Convert predictions to numpy for plotting
    if not isinstance(pred, np.ndarray):
        pred = pred.numpy()
    print(f"Shapes of selected images (scaled): LR {lr_img.shape}, SR {sr_img.shape}, HR {hr_img.shape}")

    proj      = ccrs.PlateCarree()
    hr_extent = [float(lon_hr.min()), float(lon_hr.max()),
                 float(lat_hr.min()), float(lat_hr.max())]
    lr_extent = [float(lon_lr.min()), float(lon_lr.max()),
                 float(lat_lr.min()), float(lat_lr.max())]

    # geographic overlay applied to every panel
    def _geo(ax, row, col, nrows, ncols):
        """Add coastlines and country borders to an axis."""
        ax.coastlines(resolution='10m', linewidth=0.9, color='black', zorder=6)
        ax.add_feature(
            cfeature.BORDERS.with_scale('10m'),
            linewidth=0.6, edgecolor='black', zorder=6
        )

    # single-panel imshow wrapper
    def _field(ax, data, extent, title, vmin, vmax,
               row=0, col=0, nrows=1, ncols=1, cmap="plasma"):
        """Draw one map panel with imshow and return the image handle."""
        im = ax.imshow(
            data, origin="lower", extent=extent, transform=proj,
            cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
        )
        ax.set_extent(extent, crs=proj)
        ax.set_title(title, fontsize=11, pad=4)
        _geo(ax, row, col, nrows, ncols)
        return im

    # Figure 1 - field comparison (2 rows x 3 cols)
    # Extra right margin reserved for the two colorbars (one per row)
    fig1, axes1 = plt.subplots(
        2, 3, figsize=(22, 10),
        subplot_kw={"projection": proj}
    )
    # Compress the panels leftward so the right margin is free for colorbars
    fig1.subplots_adjust(left=0.05, right=0.88, top=0.93, bottom=0.06,
                         hspace=0.25, wspace=0.08)

    channel_info = [(0, "U_100 [m/s]"), (1, "V_100 [m/s]")]

    for row, (ch, label) in enumerate(channel_info):
        lr_img   = lr  [index, ..., ch]
        pred_img = pred[index, ..., ch]
        hr_img   = hr  [index, ..., ch]

        vmin = float(min(lr_img.min(), pred_img.min(), hr_img.min()))
        vmax = float(max(lr_img.max(), pred_img.max(), hr_img.max()))

        kw = dict(vmin=vmin, vmax=vmax, nrows=2, ncols=3)
        _field(axes1[row, 0], lr_img,   lr_extent,
               f"LR Input — {label}",        row=row, col=0, **kw)
        _field(axes1[row, 1], pred_img, hr_extent,
               f"SR Prediction — {label}",   row=row, col=1, **kw)
        im = _field(axes1[row, 2], hr_img, hr_extent,
               f"HR Ground Truth — {label}", row=row, col=2, **kw)

        # manually positioned colorbar, right of the row
        pos_left  = axes1[row, 0].get_position()
        pos_right = axes1[row, 2].get_position()

        # Colorbar axis: same top/bottom as the row, 1.2 % wide, 1 % gap
        cbar_left   = pos_right.x1 + 0.012
        cbar_bottom = pos_right.y0
        cbar_width  = 0.012
        cbar_height = pos_left.y1 - pos_right.y0

        cax = fig1.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        fig1.colorbar(im, cax=cax, orientation="vertical", label=label)
        cax.tick_params(labelsize=9)

    fig1.suptitle(f"Sample {index} — LR / SR / HR comparison",
                  fontsize=13, x=0.46)
    path1 = os.path.join(results_dir, f"comparison_sample{index}.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {path1}")

    # Figure 2 - SSIM_Viewer.png (2 rows x 2 cols)
    fig2, axes2 = plt.subplots(
        2, 2, figsize=(14, 10),
        subplot_kw={"projection": proj}
    )
    # Leave right margin for two colorbars (one per column)
    fig2.subplots_adjust(left=0.06, right=0.88, top=0.93, bottom=0.06,
                         hspace=0.25, wspace=0.10)

    mean_vmin, mean_vmax = 0.0, 1.0
    std_vmin  = 0.0
    std_vmax  = float(max(ssim_std_u.max(), ssim_std_v.max()))

    panels = [
        (0, 0, ssim_mean_u, "Mean SSIM — u₁₀₀"),
        (0, 1, ssim_std_u,  "SSIM Spread (std) — u₁₀₀"),
        (1, 0, ssim_mean_v, "Mean SSIM — v₁₀₀"),
        (1, 1, ssim_std_v,  "SSIM Spread (std) — v₁₀₀"),
    ]

    im_mean = im_std = None
    for row, col, data, title in panels:
        vmin = mean_vmin if col == 0 else std_vmin
        vmax = mean_vmax if col == 0 else std_vmax
        im = _field(
            axes2[row, col], data, hr_extent, title,
            vmin=vmin, vmax=vmax,
            row=row, col=col, nrows=2, ncols=2,
            cmap="Greens"
        )
        if col == 0:
            im_mean = im
        else:
            im_std  = im

    # one colorbar per column, spanning both rows, manually placed
    for col_idx, (im_ref, cbar_label) in enumerate([
        (im_mean, "SSIM  [0 – 1]"),
        (im_std,  "Std SSIM"),
    ]):
        pos_top    = axes2[0, col_idx].get_position()
        pos_bottom = axes2[1, col_idx].get_position()

        cbar_left   = pos_top.x1 + 0.012
        cbar_bottom = pos_bottom.y0
        cbar_width  = 0.012
        cbar_height = pos_top.y1 - pos_bottom.y0

        cax = fig2.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        fig2.colorbar(im_ref, cax=cax, orientation="vertical", label=cbar_label)
        cax.tick_params(labelsize=9)

    fig2.suptitle("Spatial SSIM — mean and spread over the test set",
                  fontsize=13, x=0.46)
    path2 = os.path.join(results_dir, "SSIM_Viewer.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {path2}")


def plot_comparison_diffusion(
    lr, pred, hr,
    ssim_mean_u, ssim_std_u,
    ssim_mean_v, ssim_std_v,
    lat_hr, lon_hr, lat_lr, lon_lr,
    index=0,
    results_dir="results",
):
    """Save the ERA5 / prediction / CERRA maps and the SSIM maps of one sample.

    Writes comparison_sample{index}_{u100,v100}.png and SSIM_{u100,v100}.png.

    Args:
        lr, pred, hr: arrays (N, H, W, 2) of inputs, predictions and targets.
        ssim_mean_u, ssim_std_u, ssim_mean_v, ssim_std_v: SSIM maps (H_hr, W_hr), or None to skip them.
        lat_hr, lon_hr, lat_lr, lon_lr: coordinate vectors of the HR and LR grids.
        index: sample index to plot.
        results_dir: output directory.
    """
    os.makedirs(results_dir, exist_ok=True)

    lon2d_hr, lat2d_hr = np.meshgrid(lon_hr, lat_hr)
    lon2d_lr, lat2d_lr = np.meshgrid(lon_lr, lat_lr)

    for ch_idx, ch_name in enumerate(["u100", "v100"]):
        lr_ch   = lr[index,   ..., ch_idx]
        pred_ch = pred[index, ..., ch_idx]
        hr_ch   = hr[index,   ..., ch_idx]

        def to_numpy(x):
            """Convert a TensorFlow tensor to numpy, leave other arrays unchanged."""
            if tf.is_tensor(x):
                return x.numpy()
            return x

        lr_ch   = to_numpy(lr[index,   ..., ch_idx])
        pred_ch = to_numpy(pred[index, ..., ch_idx])
        hr_ch   = to_numpy(hr[index,   ..., ch_idx])

        vmin, vmax = np.nanpercentile(
            np.concatenate([lr_ch.ravel(), pred_ch.ravel(), hr_ch.ravel()]),
            [1, 99])

        fig, ax = plt.subplots(1, 3, figsize=(18, 6),
                               subplot_kw={'projection': ccrs.PlateCarree()})
        for a, data, title in zip(
            ax,
            [lr_ch, pred_ch, hr_ch],
            [f"ERA5 {ch_name}", f"Diffusion {ch_name}", f"CERRA {ch_name}"],
        ):
            im = a.pcolormesh(lon2d_hr, lat2d_hr, data,
                              cmap="plasma", vmin=vmin, vmax=vmax,
                              transform=ccrs.PlateCarree(), shading='auto')
            a.set_title(title, fontsize=16)
            a.coastlines(resolution='50m')
            a.add_feature(cfeature.BORDERS, linestyle=':')
            a.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
        plt.tight_layout()
        path = os.path.join(results_dir, f"comparison_sample{index}_{ch_name}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # SSIM maps
    if ssim_mean_u is not None:
        for ssim_mean, ssim_std, ch_name in [
            (ssim_mean_u, ssim_std_u, "u100"),
            (ssim_mean_v, ssim_std_v, "v100"),
        ]:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                                    subplot_kw={'projection': ccrs.PlateCarree()})
            for ax, data, title in zip(
                axes,
                [ssim_mean, ssim_std],
                [f"Mean SSIM {ch_name}", f"Std SSIM {ch_name}"],
            ):
                im = ax.pcolormesh(lon2d_hr, lat2d_hr, data, cmap="Greens",
                                transform=ccrs.PlateCarree(), shading='auto')
                ax.set_title(title, fontsize=14)
                ax.coastlines(resolution='50m')
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            path = os.path.join(results_dir, f"SSIM_{ch_name}.png")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {path}")


def tf_pipeline_from_prepared(
    case_dir,
    shared_dir,
    csv_path,
    batch_size   = 4,
    shuffle_buffer = 2900,
    model_provided = None,
    results_dir  = None,
):
    """Build the train / val / test datasets from the memmaps of prepare_training_data.py.

    Args:
        case_dir: folder Data_<case> containing train/.
        shared_dir: folder Data_Shared containing val/ and test/.
        csv_path: statistics CSV of the training case (min/max of ERA5 and CERRA).
        batch_size: batch size of all datasets.
        shuffle_buffer: shuffle buffer of the training dataset.
        model_provided: returned unchanged.
        results_dir: unused.
    Returns:
        (model_provided, datasets, minmax_stats). datasets has train_ds, val_ds, test_ds
        (batches of (lr, hr) scaled to [0, 1]) and n_train, n_val, n_test.
        minmax_stats is {"lr": (mins, maxs), "hr": (mins, maxs)} with one tf scalar per channel.
    """

    # Helper: read shape + dtype from sidecar
    def _read_sidecar(npy_path):
        """Read shape and dtype from the sidecar file of a memmap."""
        sidecar = npy_path.replace(".npy", "_shape.txt")
        lines   = open(sidecar).read().strip().splitlines()
        shape   = tuple(int(s) for s in lines[0].split(","))
        dtype   = np.dtype(lines[1]) if len(lines) > 1 else np.dtype("float32")
        return shape, dtype

    # Open all six memmaps (read-only)
    print("[tf_pipeline_from_prepared] Opening memmaps ...")

    paths = {
        "lr_train": os.path.join(case_dir,   "train", "ERA5_train.npy"),
        "hr_train": os.path.join(case_dir,   "train", "CERRA_train.npy"),
        "lr_val":   os.path.join(shared_dir, "val",   "ERA5_val.npy"),
        "hr_val":   os.path.join(shared_dir, "val",   "CERRA_val.npy"),
        "lr_test":  os.path.join(shared_dir, "test",  "ERA5_test.npy"),
        "hr_test":  os.path.join(shared_dir, "test",  "CERRA_test.npy"),
    }

    mmaps  = {}
    shapes = {}
    for key, path in paths.items():
        shape, dtype = _read_sidecar(path)
        mmaps[key]   = np.memmap(path, dtype=dtype, mode="r", shape=shape)
        shapes[key]  = shape
        print(f"  {key:12s}  shape={shape}  dtype={dtype}")

    # Spatial and channel dimensions
    H_lr, W_lr, lr_ch = shapes["lr_train"][1], shapes["lr_train"][2], shapes["lr_train"][3]
    H_hr, W_hr, hr_ch = shapes["hr_train"][1], shapes["hr_train"][2], shapes["hr_train"][3]

    # Verify val and test have the same spatial layout
    for split in ("val", "test"):
        for res, (H, W, C) in [("lr", (H_lr, W_lr, lr_ch)),
                                ("hr", (H_hr, W_hr, hr_ch))]:
            s = shapes[f"{res}_{split}"]
            if (s[1], s[2], s[3]) != (H, W, C):
                raise ValueError(
                    f"Spatial/channel mismatch between train and {split} "
                    f"for {res}: train=({H},{W},{C}), {split}={s[1:]}")

    # Read min/max constants from the statistics CSV (training only)
    print("[tf_pipeline_from_prepared] Reading min/max from statistics CSV ...")
    import pandas as pd
    df = pd.read_csv(csv_path)

    def _get_row(dataset_name):
        """Return the training row of the statistics CSV for a dataset name."""
        row = df[(df["dataset"] == dataset_name) & (df["split"] == "train")]
        if row.empty:
            raise ValueError(
                f"No 'train' row for dataset={dataset_name!r} in {csv_path}")
        return row.iloc[0]

    row_lr = _get_row("ERA5")
    row_hr = _get_row("CERRA")

    # Build per-channel min/max lists (same structure as tf_pipeline output)
    # LR channels: depending on sequence encoding the first half is U-family
    # and the second half is V-family, but all U channels share the same
    # global min/max and all V channels share their own.
    # For simplicity (and consistency with tf_pipeline) we replicate the
    # single global min/max across all sequence channels.
    if lr_ch == 1:
        lr_mins_np = [float(row_lr["min_u"])]
        lr_maxs_np = [float(row_lr["max_u"])]
    elif lr_ch == 2:
        lr_mins_np = [float(row_lr["min_u"]), float(row_lr["min_v"])]
        lr_maxs_np = [float(row_lr["max_u"]), float(row_lr["max_v"])]
    else:
        # Sequence-encoded: lr_ch = (N_BEFORE+1+N_AFTER) * n_base_ch
        n_base = 2
        half   = lr_ch // n_base  # number of time-step slots
        lr_mins_np = ([float(row_lr["min_u"])] * half
                      + [float(row_lr["min_v"])] * half)
        lr_maxs_np = ([float(row_lr["max_u"])] * half
                      + [float(row_lr["max_v"])] * half)

    if hr_ch == 1:
        hr_mins_np = [float(row_hr["min_u"])]
        hr_maxs_np = [float(row_hr["max_u"])]
    else:
        hr_mins_np = [float(row_hr["min_u"]), float(row_hr["min_v"])]
        hr_maxs_np = [float(row_hr["max_u"]), float(row_hr["max_v"])]

    # Convert to tf scalars (matches tf_pipeline minmax_stats format) to compute of float32 even for the diffusion model
    lr_mins = [tf.constant(v, dtype=tf.float32) for v in lr_mins_np]
    lr_maxs = [tf.constant(v, dtype=tf.float32) for v in lr_maxs_np]
    hr_mins = [tf.constant(v, dtype=tf.float32) for v in hr_mins_np]
    hr_maxs = [tf.constant(v, dtype=tf.float32) for v in hr_maxs_np]

    print(f"  LR  min/max: {list(zip(lr_mins_np, lr_maxs_np))}")
    print(f"  HR  min/max: {list(zip(hr_mins_np, hr_maxs_np))}")

    # Normalisation helpers
    def normalize_lr(x):
        """Min-max scale a LR sample with the training constants."""
        channels = []
        for i in range(lr_ch):
            channels.append(
                (x[..., i] - lr_mins[i]) / (lr_maxs[i] - lr_mins[i]))
        return tf.stack(channels, axis=-1)

    def normalize_hr(x):
        """Min-max scale a HR sample with the training constants."""
        channels = []
        for i in range(hr_ch):
            channels.append(
                (x[..., i] - hr_mins[i]) / (hr_maxs[i] - hr_mins[i]))
        return tf.stack(channels, axis=-1)

    # Generator factory
    def _make_generators(lr_mm, hr_mm, n):
        """Return generators yielding float32 LR and HR samples from the memmaps."""
        def lr_gen():
            """Yield LR samples."""
            for i in range(n):
                # Cast float16 to float32 here; never do float16 arithmetic
                yield lr_mm[i].astype(np.float32)

        def hr_gen():
            """Yield HR samples."""
            for i in range(n):
                yield hr_mm[i].astype(np.float32)

        return lr_gen, hr_gen

    # Build tf.data datasets for each split
    def _build_dataset(lr_mm, hr_mm, n, shuffle=False, repeat=False):
        """Build a batched dataset of scaled (lr, hr) pairs from two memmaps.

        Args:
            lr_mm, hr_mm: LR and HR memmaps.
            n: number of samples.
            shuffle: shuffle with shuffle_buffer.
            repeat: repeat indefinitely.
        Returns:
            Prefetched tf.data.Dataset.
        """
        lr_gen, hr_gen = _make_generators(lr_mm, hr_mm, n)

        tf_lr = tf.data.Dataset.from_generator(
            lr_gen,
            output_signature=tf.TensorSpec(
                shape=(H_lr, W_lr, lr_ch), dtype=tf.float32))
        tf_hr = tf.data.Dataset.from_generator(
            hr_gen,
            output_signature=tf.TensorSpec(
                shape=(H_hr, W_hr, hr_ch), dtype=tf.float32))

        tf_lr = tf_lr.map(normalize_lr, num_parallel_calls=tf.data.AUTOTUNE)
        tf_hr = tf_hr.map(normalize_hr, num_parallel_calls=tf.data.AUTOTUNE)

        ds = tf.data.Dataset.zip((tf_lr, tf_hr))

        if shuffle:
            ds = ds.shuffle(
                buffer_size=shuffle_buffer,
                reshuffle_each_iteration=True,
                seed=42,
            )

        ds = ds.batch(batch_size, drop_remainder=True)

        if repeat:
            ds = ds.repeat()

        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    print("[tf_pipeline_from_prepared] Building tf.data datasets ...")

    n_train = shapes["lr_train"][0]
    n_val   = shapes["lr_val"][0]
    n_test  = shapes["lr_test"][0]

    train_ds = _build_dataset(mmaps["lr_train"], mmaps["hr_train"],
                            n_train, shuffle=True,  repeat=True)  # Repeat to get through it all for each epochs
    val_ds   = _build_dataset(mmaps["lr_val"],   mmaps["hr_val"],
                            n_val,   shuffle=False, repeat=True)  # Repeat for each epochs
    test_ds  = _build_dataset(mmaps["lr_test"],  mmaps["hr_test"],
                            n_test,  shuffle=False, repeat=False)  # No repeat as it's only testing once

    print(f"  train_ds: {n_train} samples  "
          f"({n_train // batch_size} batches of {batch_size})")
    print(f"  val_ds  : {n_val}   samples  "
          f"({n_val   // batch_size} batches)")
    print(f"  test_ds : {n_test}  samples  "
          f"({n_test  // batch_size} batches)")

    datasets     = {"train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds,"n_train":  n_train,"n_val":n_val,"n_test":n_test}
    minmax_stats = {"lr": (lr_mins, lr_maxs), "hr": (hr_mins, hr_maxs)}

    return model_provided, datasets, minmax_stats


def tf_pipeline(lr_mmap_path,
        hr_mmap_path,
        lr_shape,
        hr_shape,
                           scale_factor=5,
                           batch_size=2,
                           split_ratios=(0.7, 0.15, 0.15),
                           shuffle_buffer=3000,
                           epochs=2,
                           early_stopping_patience=15,
                           widths=[32, 64, 96, 128],
                           block_depth=3,
                           normalizer_mean=None,
                           normalizer_variance=None,
                           results_dir=None,
                           model_provided=None):
    """Build train / val / test datasets from raw float16 memmap files.

    Min/max scaling constants are computed on the training split only.

    Args:
        lr_mmap_path, hr_mmap_path: memmap files of the LR and HR arrays.
        lr_shape, hr_shape: shapes (N, H, W, C) of the two arrays.
        scale_factor, epochs, early_stopping_patience, widths, block_depth,
        normalizer_mean, normalizer_variance: unused.
        batch_size: batch size.
        split_ratios: (train, val, test) fractions.
        shuffle_buffer: unused.
        results_dir: output directory.
        model_provided: model returned as is (must not be None).
    Returns:
        (model, datasets, minmax_stats), with the same structure as tf_pipeline_from_prepared.
    """

    # Load shapes and open memmaps read-only - no RAM loading
    print("[0/7] Opening memory-mapped files...")

    lr_mmap = np.memmap(lr_mmap_path, dtype='float16', mode='r', shape=lr_shape)
    hr_mmap = np.memmap(hr_mmap_path, dtype='float16', mode='r', shape=hr_shape)

    N    = lr_shape[0]
    H_lr, W_lr, lr_ch = lr_shape[1], lr_shape[2], lr_shape[3]
    H_hr, W_hr, hr_ch = hr_shape[1], hr_shape[2], hr_shape[3]

    # TF datasets via generator - streams from disk, no RAM copy
    print("[1/7] Creating streaming TF datasets from memmap...")

    def lr_generator():
        """Yield LR samples from the memmap."""
        for i in range(N):
            yield lr_mmap[i]  # (H_lr, W_lr, 2) - memmap reads page on demand

    def hr_generator():
        """Yield HR samples from the memmap."""
        for i in range(N):
            yield hr_mmap[i]  # (H_hr, W_hr, 1)

    tf_ds_lr = tf.data.Dataset.from_generator(
        lr_generator,
        output_signature=tf.TensorSpec(shape=(H_lr, W_lr, lr_ch), dtype=tf.float32)
    )
    tf_ds_hr = tf.data.Dataset.from_generator(
        hr_generator,
        output_signature=tf.TensorSpec(shape=(H_hr, W_hr, hr_ch), dtype=tf.float32)
    )

    # Min/max per channel - computed on train split only (no leakage)
    print("[2/7] Computing per-channel min/max on train split only...")

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
            min_val, max_val = dataset.map(lambda x: x[..., c]).reduce(
                (tf.constant(1e9,  dtype=tf.float32),
                 tf.constant(-1e9, dtype=tf.float32)),
                lambda state, x: (
                    tf.minimum(state[0], tf.reduce_min(tf.cast(x, tf.float32))),
                    tf.maximum(state[1], tf.reduce_max(tf.cast(x, tf.float32)))
                )
            )
            mins.append(min_val)
            maxs.append(max_val)
        return mins, maxs

    # Only iterate over the training portion to avoid test leakage
    lr_mins, lr_maxs = compute_channel_minmax(tf_ds_lr.take(n_train), lr_ch)
    hr_mins, hr_maxs = compute_channel_minmax(tf_ds_hr.take(n_train), hr_ch)

    # Normalization - identical to original
    print("[3/7] Normalizing...")

    def normalize_lr(x):
        """Min-max scale a LR sample channel by channel."""
        return tf.stack(
            [(x[..., i] - lr_mins[i]) / (lr_maxs[i] - lr_mins[i])
             for i in range(lr_ch)],
            axis=-1
        )

    def normalize_hr(x):
        """Min-max scale a HR sample channel by channel."""
        return tf.stack(
            [(x[..., i] - hr_mins[i]) / (hr_maxs[i] - hr_mins[i])
            for i in range(hr_ch)],
            axis=-1
        )

    tf_ds_lr_scaled = tf_ds_lr.map(normalize_lr, num_parallel_calls=tf.data.AUTOTUNE)
    tf_ds_hr_scaled = tf_ds_hr.map(normalize_hr, num_parallel_calls=tf.data.AUTOTUNE)

    print("[5/7] Building (H, W, 5) tensors  [4 LR channels + 1 HR channel]...")
    def concat_lr_hr(lr_up, hr):
        """Concatenate LR and HR channels of a sample and cast to float32."""
        full = tf.concat([lr_up, hr], axis=-1)  # (H_hr, W_hr, 5)
        return tf.cast(full, tf.float32)  # Put it in float32 instead for more precision

    print(tf_ds_hr_scaled)
    tf_ds_lr_up = tf_ds_lr_scaled
    # zip produces (lr_up_sample, hr_sample) pairs, then we merge them
    dataset = tf.data.Dataset.zip((tf_ds_lr_up, tf_ds_hr_scaled))

    # each element is now a single (H_hr, W_hr, 5) tensor - no (X,y) split

    # Train / val / test split
    print("[6/7] Splitting into train / val / test...")
    total_size = lr_shape[0]
    n_train    = int(split_ratios[0] * total_size)
    n_val      = int(split_ratios[1] * total_size)
    n_test     = total_size - n_train - n_val

    print("Total size",total_size)
    print("Train",n_train)
    print("Val",n_val)
    print("Test",n_test)

    # DiffusionModel.train_step / test_step handle the split internally.

    train_ds = dataset.take(n_train).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).repeat().shuffle(42,reshuffle_each_iteration=True)
    val_ds   = dataset.skip(n_train).take(n_val).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    test_ds  = dataset.skip(n_train + n_val).take(n_test).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    # free numpy arrays early
    del tf_ds_lr, tf_ds_hr, tf_ds_lr_scaled, tf_ds_hr_scaled, tf_ds_lr_up, dataset
    import gc; gc.collect()

    # Build or receive the model
    print("[7/7] Model setup and training...")
    if model_provided is None:

        model_path = os.path.join(results_dir, "Diffusion_best_model")
        model = model_provided
        print("No model has been provided, we can reconstruct it here:")
        model.compile(
            optimizer=keras.optimizers.experimental.AdamW(
                learning_rate=1e-5, weight_decay=1e-6
            ),
            loss=keras.losses.mean_squared_error,
        )

    else:
        model = model_provided

    datasets     = {"train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds,"n_train":  n_train,"n_val":n_val,"n_test":n_test}
    minmax_stats = {'lr': (lr_mins, lr_maxs), 'hr': (hr_mins, hr_maxs)}

    return model, datasets, minmax_stats


def ml_diffusion_pipeline_streamed(
    case_dir,
    shared_dir,
    csv_path,
    scale_factor            = 5,
    batch_size              = 2,
    shuffle_buffer          = 200,
    epochs                  = 150,
    early_stopping_patience = 10,
    normalizer_mean         = None,
    normalizer_variance     = None,
    model_provided          = None,
    load_pretrained_path    = None,
    results_dir             = None,
    extra_callbacks = None,
    steps_per_execution = 1,
    checkpoint_best         = float("inf"),
):
    """Build the datasets, prepare and train a diffusion model from prepared memmaps.

    Args:
        case_dir: folder Data_<case> containing train/.
        shared_dir: folder Data_Shared containing val/ and test/.
        csv_path: statistics CSV of the training case.
        scale_factor: unused.
        batch_size: batch size.
        shuffle_buffer: shuffle buffer of the training dataset.
        epochs: maximum number of epochs (0 skips training).
        early_stopping_patience: patience on val_n_loss.
        normalizer_mean, normalizer_variance: statistics set on the model normalizer.
        model_provided: compiled DiffusionModel (required).
        load_pretrained_path: prefix of network and EMA weight files to load.
        results_dir: output directory of the checkpoints (required).
        extra_callbacks: additional Keras callbacks.
        steps_per_execution: steps per compiled execution; step counts are rounded to a multiple.
        checkpoint_best: initial best val_n_loss of the checkpoint callback.
    Returns:
        (model, history, datasets, minmax_stats), with datasets and minmax_stats as in tf_pipeline_from_prepared.
    """
    if results_dir is None:
        raise ValueError(
            "results_dir must be passed explicitly to "
            "ml_diffusion_pipeline_streamed().  "
            "Pass RESULTS_DIR from A_UNet_Diff_V2.py.")

    # Open memmaps
    print("[ml_diffusion_pipeline_streamed] 0/7 — Opening memmaps ...")
    paths = {
        "lr_train": os.path.join(case_dir,   "train", "ERA5_train.npy"),
        "hr_train": os.path.join(case_dir,   "train", "CERRA_train.npy"),
        "lr_val":   os.path.join(shared_dir, "val",   "ERA5_val.npy"),
        "hr_val":   os.path.join(shared_dir, "val",   "CERRA_val.npy"),
        "lr_test":  os.path.join(shared_dir, "test",  "ERA5_test.npy"),
        "hr_test":  os.path.join(shared_dir, "test",  "CERRA_test.npy"),
    }
    mmaps, shapes = {}, {}
    for key, path in paths.items():
        shape, dtype    = _read_sidecar(path)
        mmaps[key]      = np.memmap(path, dtype=dtype, mode="r", shape=shape)
        shapes[key]     = shape
        print(f"  {key:12s}  shape={shape}  dtype={dtype}")

    H_lr, W_lr, lr_ch = shapes["lr_train"][1], shapes["lr_train"][2], shapes["lr_train"][3]
    H_hr, W_hr, hr_ch = shapes["hr_train"][1], shapes["hr_train"][2], shapes["hr_train"][3]

    if lr_ch != 2:
        raise ValueError(f"Expected 2 LR channels (u,v), got {lr_ch}.")
    if hr_ch != 2:
        raise ValueError(f"Expected 2 HR channels (u,v), got {hr_ch}.")

    for split in ("val", "test"):
        for res, (H, W, C) in [("lr", (H_lr, W_lr, lr_ch)),
                                ("hr", (H_hr, W_hr, hr_ch))]:
            s = shapes[f"{res}_{split}"]
            if (s[1], s[2], s[3]) != (H, W, C):
                raise ValueError(
                    f"Spatial mismatch between train and {split} for {res}.")

    # Min/max from CSV
    print("[ml_diffusion_pipeline_streamed] 1/7 — Reading min/max from CSV ...")
    df = pd.read_csv(csv_path)

    def _get_row(dataset_name, split="train"):
        """Return the row of the statistics CSV for a dataset name and split."""
        row = df[(df["dataset"] == dataset_name) & (df["split"] == split)]
        if row.empty:
            raise ValueError(f"No '{split}' row for '{dataset_name}' in {csv_path}.")
        return row.iloc[0]

    row_lr = _get_row("ERA5",  split="train")
    row_hr = _get_row("CERRA", split="train")
    lr_mins_np = [float(row_lr["min_u"]), float(row_lr["min_v"])]
    lr_maxs_np = [float(row_lr["max_u"]), float(row_lr["max_v"])]
    hr_mins_np = [float(row_hr["min_u"]), float(row_hr["min_v"])]
    hr_maxs_np = [float(row_hr["max_u"]), float(row_hr["max_v"])]

    lr_mins = [tf.constant(v, dtype=tf.float32) for v in lr_mins_np]
    lr_maxs = [tf.constant(v, dtype=tf.float32) for v in lr_maxs_np]
    hr_mins = [tf.constant(v, dtype=tf.float32) for v in hr_mins_np]
    hr_maxs = [tf.constant(v, dtype=tf.float32) for v in hr_maxs_np]
    print(f"  LR min/max: {list(zip(lr_mins_np, lr_maxs_np))}")
    print(f"  HR min/max: {list(zip(hr_mins_np, hr_maxs_np))}")

    # Normalisation
    print("[ml_diffusion_pipeline_streamed] 2/7 — Defining normalisation ...")

    def normalize_lr(x):
        """Min-max scale a LR sample channel by channel."""
        return tf.stack(
            [(x[..., i] - lr_mins[i]) / (lr_maxs[i] - lr_mins[i])
             for i in range(lr_ch)], axis=-1)

    def normalize_hr(x):
        """Min-max scale a HR sample channel by channel."""
        return tf.stack(
            [(x[..., i] - hr_mins[i]) / (hr_maxs[i] - hr_mins[i])
             for i in range(hr_ch)], axis=-1)

    # Generator factory
    print("[ml_diffusion_pipeline_streamed] 3/7 — Building generators ...")

    def _make_generators(lr_mm, hr_mm, n):
        """Return generators yielding float32 LR and HR samples from the memmaps."""
        def lr_gen():
            """Yield LR samples."""
            for i in range(n):
                yield lr_mm[i].astype(np.float32)
        def hr_gen():
            """Yield HR samples."""
            for i in range(n):
                yield hr_mm[i].astype(np.float32)
        return lr_gen, hr_gen

    target_size = (H_hr, W_hr)

    def upsample_lr(x):
        """Bilinearly resize a LR sample to the HR grid."""
        return tf.image.resize(x, target_size, method="bilinear")

    # Build tf.data datasets
    print("[ml_diffusion_pipeline_streamed] 4/7 — Building tf.data datasets ...")

    def _build_dataset(lr_mm, hr_mm, n, shuffle=False, repeat=False,shuffle_buffer=shuffle_buffer):
        """Build a batched dataset of (B, H_hr, W_hr, 4) tensors: upsampled LR channels followed by HR channels.

        Args:
            lr_mm, hr_mm: LR and HR memmaps.
            n: number of samples.
            shuffle: shuffle with shuffle_buffer.
            repeat: repeat indefinitely.
            shuffle_buffer: shuffle buffer size.
        Returns:
            Prefetched tf.data.Dataset.
        """
        lr_gen, hr_gen = _make_generators(lr_mm, hr_mm, n)
        tf_lr = tf.data.Dataset.from_generator(
            lr_gen,
            output_signature=tf.TensorSpec(shape=(H_lr, W_lr, lr_ch), dtype=tf.float32))
        tf_hr = tf.data.Dataset.from_generator(
            hr_gen,
            output_signature=tf.TensorSpec(shape=(H_hr, W_hr, hr_ch), dtype=tf.float32))
        tf_lr = tf_lr.map(normalize_lr, num_parallel_calls=tf.data.AUTOTUNE)
        tf_hr = tf_hr.map(normalize_hr, num_parallel_calls=tf.data.AUTOTUNE)
        tf_lr = tf_lr.map(upsample_lr,  num_parallel_calls=tf.data.AUTOTUNE)

        def concat_lr_hr(lr_up, hr):
            """Concatenate upsampled LR and HR channels."""
            return tf.cast(tf.concat([lr_up, hr], axis=-1), tf.float32)

        ds = tf.data.Dataset.zip((tf_lr, tf_hr))
        ds = ds.map(concat_lr_hr, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(buffer_size=shuffle_buffer,
                            reshuffle_each_iteration=True, seed=42)
        ds = ds.batch(batch_size, drop_remainder=True)
        if repeat:
            ds = ds.repeat()
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    n_train = shapes["lr_train"][0]
    n_val   = shapes["lr_val"][0]
    n_test  = shapes["lr_test"][0]
    print("Here is the number of testing steps:")
    print(n_test)

    train_ds = _build_dataset(mmaps["lr_train"], mmaps["hr_train"],
                              n_train, shuffle=True,  repeat=True)

    val_shuffle_buffer = min(n_val, 500)

    val_ds = _build_dataset(mmaps["lr_val"], mmaps["hr_val"],
                            n_val, shuffle=False, repeat=True,
                            )

    test_ds  = _build_dataset(mmaps["lr_test"],  mmaps["hr_test"],
                              n_test,  shuffle=False, repeat=False)

    print(f"  train: {n_train} samples  val: {n_val}  test: {n_test}")

    # Model setup
    print("[ml_diffusion_pipeline_streamed] 5/7 — Model setup ...")
    if model_provided is None:
        raise ValueError("model_provided is required. Pass a compiled DiffusionModel.")
    model = model_provided

    for sample_batch in train_ds.take(1):
        model.normalizer.adapt(sample_batch)

    if normalizer_mean is not None and normalizer_variance is not None:
        model.normalizer.mean     = normalizer_mean
        model.normalizer.variance = normalizer_variance
        print(f"  Normaliser set from CSV: mean={normalizer_mean}")
    else:
        print("  [WARNING] normalizer stats not provided; using single-batch estimate.")

    # Load pretrained weights if provided
    if load_pretrained_path is not None:
        network_path = load_pretrained_path + ".weights.h5"
        ema_path     = load_pretrained_path + "_ema.weights.h5"

        # The sub-networks must be built before load_weights can map variables.
        # A single dummy forward pass through the compiled model is sufficient.
        _dummy_concat = tf.zeros(
            [1, H_hr, W_hr, 2 + 2], dtype=tf.float32)  # input_frames + output_frames
        _dummy_rate   = tf.zeros([1, 1, 1, 1],  dtype=tf.float32)
        _ = model.network    ([_dummy_concat, _dummy_rate], training=False)
        _ = model.ema_network([_dummy_concat, _dummy_rate], training=False)

        if os.path.exists(network_path):
            model.network.load_weights(network_path)
            print(f"  Loaded network weights     ← {network_path}")
        else:
            print(f"  WARNING: network weights not found at {network_path}")

        if os.path.exists(ema_path):
            model.ema_network.load_weights(ema_path)
            print(f"  Loaded EMA network weights ← {ema_path}")
        else:
            print(f"  WARNING: EMA weights not found at {ema_path}")
    else:
        print("  [INFO] load_pretrained_path=None — starting from random init.")

    # Callbacks
    print("[ml_diffusion_pipeline_streamed] 6/7 — Setting up callbacks ...")
    model_checkpoint_path = os.path.join(results_dir, "Diffusion_Checkpoint")

    class DiffusionCheckpoint(keras.callbacks.Callback):
        """Callback saving the network and EMA weights when the monitored loss improves."""

        def __init__(self, filepath, monitor="val_n_loss"):
            """Store the file prefix, the monitored metric and the best value so far."""
            super().__init__()
            self.filepath = filepath
            self.monitor  = monitor
            self.best     = checkpoint_best

        def on_epoch_end(self, epoch, logs=None):
            """Save the weights if the monitored value improved."""
            current = logs.get(self.monitor)
            if current is None:
                return
            if current < self.best:
                self.best = current
                self.model.network.save_weights(self.filepath + ".weights.h5")
                self.model.ema_network.save_weights(self.filepath + "_ema.weights.h5")
                print(f"\n  Epoch {epoch+1}: {self.monitor}={current:.7f} → saved")

    steps_per_epoch_raw = n_train // batch_size
    # Round down so steps_per_epoch is always an exact multiple
    steps_per_epoch = max(
        steps_per_execution,
        (steps_per_epoch_raw // steps_per_execution) * steps_per_execution
    )
    validation_steps_raw = n_val   // batch_size

    validation_steps_min = min(n_val // batch_size, 512)  # To only do the validation on 50 samples
    validation_steps = max(
        steps_per_execution,
        (validation_steps_raw // steps_per_execution) * steps_per_execution
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_n_loss", mode="min",
            patience=early_stopping_patience,
            min_delta=1e-10,
            restore_best_weights=True,
        ),
        DiffusionCheckpoint(filepath=model_checkpoint_path, monitor="val_n_loss"),
    ]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    # Training
    print("[ml_diffusion_pipeline_streamed] 7/7 — Training ...")

    if epochs == 0:
        print("  epochs=0 — skipping training, using loaded weights as-is.")
        # Return a dummy history so callers don't need to guard against None
        history = tf.keras.callbacks.History()
        history.history = {}
    else:
        print(f"  steps_per_epoch={steps_per_epoch}  "
              f"validation_steps={validation_steps}")
        history = model.fit(
            train_ds,
            validation_data  = val_ds,
            epochs           = epochs,
            steps_per_epoch  = steps_per_epoch,
            validation_steps = validation_steps,
            verbose          = 1,
            callbacks        = callbacks,
        )

    import gc
    gc.collect()

    datasets = {
        "train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds,
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
    }
    minmax_stats = {"lr": (lr_mins, lr_maxs), "hr": (hr_mins, hr_maxs)}
    return model, history, datasets, minmax_stats
