"""Figures on the best and worst test samples and on the metric distributions of two training cases.

Input: temporal_metrics.csv of each case (written by the evaluation scripts, columns datetime,
psnr_db, ssim). main compares two cases (CASE and CASE_2):
    overlaid histograms of PSNR and SSIM (plot_metric_distributions_compare),
    month and hour distribution of the worst and best samples (plot_temporal_distributions_compare).
With PLOT_SINGLE_CASE_FIGURES the same figures are also drawn for each case separately.

Per-sample maps of the extreme samples are built from get_extreme_positions, fetch_from_test_ds
(after build_test_dataset_only) and _run_pass.

Configuration (module constants):
    CASE, CASE_2: the two cases; RESULTS_DIR, RESULTS_DIR_2: their result folders.
    N_SAMPLES, N_SAMPLES_2: number of best and worst samples per metric.
    OUT_BASE: output folder of the figures.
    CASE_COLORS, CASE_LABELS: colour and label of each case.
    PLOT_SINGLE_CASE_FIGURES: also draw the per-case figures.
    OUTPUT_ROOT: root folder of the data and results, to adapt to the local file system.
Author: Marion Falkenauer (with the help of Claude)
"""

import os
import numpy as np
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTPUT_ROOT = "Output of the code to be found there"

CASE = "4year"
RESULTS_DIR = os.path.join(
    OUTPUT_ROOT,
    f"Results_ML/UNet_Eval_Clean/{CASE}",
)
N_SAMPLES = 29 * 50

CASE_2 = "2year"
RESULTS_DIR_2 = os.path.join(
    OUTPUT_ROOT,
    f"Results_ML/UNet_Eval_Clean/{CASE_2}",
)
N_SAMPLES_2 = 29 * 50

SHARED_DIR = os.path.join(OUTPUT_ROOT, "Data_Shared")
CASE_DIR   = os.path.join(OUTPUT_ROOT, f"Data_{CASE}")
BATCH_SIZE = 2
PLOT_SINGLE_CASE_FIGURES = False
OUT_BASE   = os.path.join("Res_temporalV5-Comparison-50")

CASE_COLORS = {
    CASE:   "#1C3A5E",
    CASE_2: "#8B1A1A",
}
CASE_LABELS = {
    CASE:   "4-year",
    CASE_2: "2-year",
}


def _read_sidecar(npy_path: str):
    """Read the shape and dtype from the sidecar file of a memmap."""
    sidecar = npy_path.replace(".npy", "_shape.txt")
    lines   = open(sidecar).read().strip().splitlines()
    shape   = tuple(int(s) for s in lines[0].split(","))
    dtype   = np.dtype(lines[1]) if len(lines) > 1 else np.dtype("float32")
    return shape, dtype


def build_test_dataset_only(
    case_dir:   str,
    shared_dir: str,
    case_name:  str,
    batch_size: int = 2,
) -> tuple:
    """Build the test dataset of the shared memmaps with LR scaled and upsampled to the HR grid.

    Args:
        case_dir: folder Data_<case> containing statistics_<case>.csv.
        shared_dir: folder Data_Shared containing test/.
        case_name: case label.
        batch_size: batch size (the last incomplete batch is dropped).
    Returns:
        (test_ds, N_test, minmax_stats): dataset of (lr, hr) batches scaled to [0, 1], number of samples
        and {"lr": (mins, maxs), "hr": (mins, maxs)} of tf scalars.
    """
    lr_path = os.path.join(shared_dir, "test", "ERA5_test.npy")
    hr_path = os.path.join(shared_dir, "test", "CERRA_test.npy")

    for p in (lr_path, hr_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Memmap not found: {p}")

    lr_shape, lr_dtype = _read_sidecar(lr_path)
    hr_shape, hr_dtype = _read_sidecar(hr_path)

    N_test                 = lr_shape[0]
    H_lr, W_lr, lr_ch     = lr_shape[1], lr_shape[2], lr_shape[3]
    H_hr, W_hr, hr_ch     = hr_shape[1], hr_shape[2], hr_shape[3]

    print(f"  Test split:  LR {lr_shape}  HR {hr_shape}")

    stats_path = os.path.join(case_dir, f"statistics_{case_name}.csv")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Statistics CSV not found: {stats_path}")

    stats_df = pd.read_csv(stats_path, index_col=0)

    def _get_minmax(row_label, n_ch):
        """Return the min and max constants of a dataset row of the statistics CSV for each channel."""
        comp_map = {0: ("min_u", "max_u"), 1: ("min_v", "max_v")}
        mins, maxs = [], []
        for c in range(n_ch):
            min_col, max_col = comp_map[c % 2]
            mins.append(tf.constant(float(stats_df.loc[row_label, min_col]),
                                    dtype=tf.float32))
            maxs.append(tf.constant(float(stats_df.loc[row_label, max_col]),
                                    dtype=tf.float32))
        return mins, maxs

    lr_mins, lr_maxs = _get_minmax("ERA5",  lr_ch)
    hr_mins, hr_maxs = _get_minmax("CERRA", hr_ch)

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

    lr_mm = np.memmap(lr_path, dtype=lr_dtype, mode="r", shape=lr_shape)
    hr_mm = np.memmap(hr_path, dtype=hr_dtype, mode="r", shape=hr_shape)

    def _lr_gen():
        """Yield LR samples."""
        for i in range(N_test):
            yield lr_mm[i].astype(np.float32)

    def _hr_gen():
        """Yield HR samples."""
        for i in range(N_test):
            yield hr_mm[i].astype(np.float32)

    tf_lr = (tf.data.Dataset
             .from_generator(_lr_gen,
                             output_signature=tf.TensorSpec(
                                 shape=(H_lr, W_lr, lr_ch), dtype=tf.float32))
             .map(normalize_lr,  num_parallel_calls=tf.data.AUTOTUNE)
             .map(upsample_lr,   num_parallel_calls=tf.data.AUTOTUNE))

    tf_hr = (tf.data.Dataset
             .from_generator(_hr_gen,
                             output_signature=tf.TensorSpec(
                                 shape=(H_hr, W_hr, hr_ch), dtype=tf.float32))
             .map(normalize_hr,  num_parallel_calls=tf.data.AUTOTUNE))

    test_ds = (tf.data.Dataset.zip((tf_lr, tf_hr))
               .batch(batch_size, drop_remainder=True)
               .prefetch(tf.data.AUTOTUNE))

    minmax_stats = {"lr": (lr_mins, lr_maxs), "hr": (hr_mins, hr_maxs)}
    return test_ds, N_test, minmax_stats


def get_extreme_positions(
    csv_path: str,
    n:        int   = 10,
    metrics:  tuple = ("psnr_db", "ssim"),
    mode:     str   = "worst",
) -> dict:
    """Find the best or worst samples of each metric in a temporal metrics CSV.

    Args:
        csv_path: CSV with columns datetime and the metrics.
        n: number of samples per metric.
        metrics: metric columns.
        mode: "worst" (lowest values) or "best" (highest values).
    Returns:
        Dictionary metric -> DataFrame with columns row_pos, datetime and the metric.
    """
    if mode not in ("worst", "best"):
        raise ValueError(f"mode must be 'worst' or 'best', got '{mode}'")

    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df.index = np.arange(len(df))

    results = {}
    for metric in metrics:
        if metric not in df.columns:
            raise ValueError(f"Column '{metric}' not in CSV. "
                             f"Available: {list(df.columns)}")

        if mode == "worst":
            extreme = df[["datetime", metric]].nsmallest(n, metric).copy()
        else:
            extreme = df[["datetime", metric]].nlargest(n, metric).copy()

        extreme.index.name = "row_pos"
        extreme = extreme.reset_index()
        label   = "best" if mode == "best" else "worst"
        print(f"\n── {n} {label} '{metric}' ──")
        print(extreme.to_string(index=False))
        results[metric] = extreme

    return results


def fetch_from_test_ds(
    test_ds:       tf.data.Dataset,
    row_positions: list,
    batch_size:    int,
    n_test:        int,
) -> dict:
    """Retrieve samples of the test dataset by their position.

    Args:
        test_ds: dataset of (lr, hr) batches.
        row_positions: sample positions in the test set.
        batch_size: batch size of test_ds.
        n_test: number of test samples (positions in the dropped last batch are skipped).
    Returns:
        Dictionary position -> {"lr": array, "hr": array}.
    """
    n_usable = (n_test // batch_size) * batch_size
    dropped  = [p for p in row_positions if p >= n_usable]
    valid    = [p for p in row_positions if p <  n_usable]

    if dropped:
        print(f"\n⚠  {len(dropped)} position(s) fall in the dropped tail "
              f"(drop_remainder=True keeps {n_usable}/{n_test}): {dropped}")

    batch_map: dict = {}
    for pos in valid:
        b = pos // batch_size
        w = pos %  batch_size
        batch_map.setdefault(b, []).append(w)

    max_batch = max(batch_map) if batch_map else -1

    retrieved = {}
    for b_idx, (lr_batch, hr_batch) in enumerate(test_ds):
        if b_idx > max_batch:
            break
        if b_idx not in batch_map:
            continue
        for w_idx in batch_map[b_idx]:
            row_pos = b_idx * batch_size + w_idx
            retrieved[row_pos] = {
                "lr": lr_batch[w_idx].numpy(),
                "hr": hr_batch[w_idx].numpy(),
            }

    print(f"\n  Retrieved {len(retrieved)}/{len(valid)} requested samples.")
    return retrieved


def _make_filename(metric_name: str, metric_val: float,
                   timestamp: pd.Timestamp) -> str:
    """File stem of a sample figure, from the metric value and the timestamp.

    Args:
        metric_name: "ssim" (value x 1000) or another metric (value x 100).
        metric_val: metric value.
        timestamp: time of the sample.
    Returns:
        String <code>-<Month>-<day>-<hour>h.
    """
    if metric_name == "ssim":
        code = str(int(round(metric_val * 1000)))
    else:
        code = str(int(round(metric_val * 100)))

    month = timestamp.strftime("%B")
    day   = f"{timestamp.day:02d}"
    hour  = f"{timestamp.hour:02d}h"
    return f"{code}-{month}-{day}-{hour}"


def _uv_to_speed_dir(u: np.ndarray, v: np.ndarray) -> tuple:
    """Convert u and v to wind speed and direction.

    Args:
        u, v: components in m/s.
    Returns:
        (speed, wdir) float32 arrays; wdir in degrees, from arctan2(-u, -v).
    """
    speed = np.sqrt(u**2 + v**2)
    wdir  = np.degrees(np.arctan2(-u, -v)) % 360
    return speed.astype(np.float32), wdir.astype(np.float32)


def plot_sample(
    lr_norm, hr_norm, minmax, lat_hr, lon_hr,
    metric_name, metric_val, timestamp, save_path, mode="worst",
):
    """Save the LR and HR maps of u and v of one sample.

    Args:
        lr_norm, hr_norm: arrays (H, W, 2) scaled to [0, 1].
        minmax: min/max constants of the pipeline, used to unscale.
        lat_hr, lon_hr: coordinate vectors of the HR grid.
        metric_name, metric_val: metric shown in the title.
        timestamp: time of the sample.
        save_path: output PDF.
        mode: "worst" or "best", shown in the title.
    """
    lr_mins_np = np.array([v.numpy() for v in minmax["lr"][0]],
                          dtype=np.float32).reshape(1, 1, 2)
    lr_maxs_np = np.array([v.numpy() for v in minmax["lr"][1]],
                          dtype=np.float32).reshape(1, 1, 2)
    hr_mins_np = np.array([v.numpy() for v in minmax["hr"][0]],
                          dtype=np.float32).reshape(1, 1, 2)
    hr_maxs_np = np.array([v.numpy() for v in minmax["hr"][1]],
                          dtype=np.float32).reshape(1, 1, 2)

    lr_phys = lr_norm * (lr_maxs_np - lr_mins_np) + lr_mins_np
    hr_phys = hr_norm * (hr_maxs_np - hr_mins_np) + hr_mins_np

    extent     = [float(lon_hr.min()), float(lon_hr.max()),
                  float(lat_hr.min()), float(lat_hr.max())]
    ch_names   = ["U100  [m/s]", "V100  [m/s]"]
    col_titles = ["ERA5  (LR bilinear up.)", "CERRA  (HR ground truth)"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12),
                             gridspec_kw={"hspace": 0.30, "wspace": 0.10},
                             layout="constrained")
    fig.suptitle(
        f"[{mode.upper()}]  {metric_name.upper()} = {metric_val:.4f}"
        f"   —   {timestamp.strftime('%Y-%m-%d  %H:%M UTC')}",
        y=1.01,
    )

    for ch, ch_label in enumerate(ch_names):
        data_lr = lr_phys[..., ch]
        data_hr = hr_phys[..., ch]
        all_vals = np.concatenate([data_lr.ravel(), data_hr.ravel()])
        vmin, vmax = np.nanpercentile(all_vals, [1, 99])

        for col, (data, col_title) in enumerate(
            [(data_lr, col_titles[0]), (data_hr, col_titles[1])]
        ):
            ax = axes[ch, col]
            im = ax.imshow(data, origin="upper", extent=extent, aspect="auto",
                           cmap="inferno", vmin=vmin, vmax=vmax,
                           interpolation="nearest")
            if ch == 0:
                ax.set_title(col_title, pad=6)
            if col == 0:
                ax.set_ylabel(ch_label + "\nLatitude [°N]")
            else:
                ax.set_ylabel("Latitude [°N]")
            ax.set_xlabel("Longitude [°E]")
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f°"))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f°"))
            fig.colorbar(im, ax=ax, fraction=0.032, pad=0.04,
                         label="m/s", orientation="vertical")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_sample_speed_dir(
    lr_norm, hr_norm, minmax, lat_hr, lon_hr,
    metric_name, metric_val, timestamp, save_path,
    mode="worst", quiver_skip=8,
):
    """Save wind speed, direction and arrows of the LR and HR fields of one sample.

    Args:
        lr_norm, hr_norm: arrays (H, W, 2) scaled to [0, 1].
        minmax: min/max constants of the pipeline, used to unscale.
        lat_hr, lon_hr: coordinate vectors of the HR grid.
        metric_name, metric_val: metric shown in the title.
        timestamp: time of the sample.
        save_path: output PDF.
        mode: "worst" or "best", shown in the title.
        quiver_skip: pixel spacing between arrows.
    """
    lr_mins_np = np.array([v.numpy() for v in minmax["lr"][0]],
                          dtype=np.float32).reshape(1, 1, 2)
    lr_maxs_np = np.array([v.numpy() for v in minmax["lr"][1]],
                          dtype=np.float32).reshape(1, 1, 2)
    hr_mins_np = np.array([v.numpy() for v in minmax["hr"][0]],
                          dtype=np.float32).reshape(1, 1, 2)
    hr_maxs_np = np.array([v.numpy() for v in minmax["hr"][1]],
                          dtype=np.float32).reshape(1, 1, 2)

    lr_phys = lr_norm * (lr_maxs_np - lr_mins_np) + lr_mins_np
    hr_phys = hr_norm * (hr_maxs_np - hr_mins_np) + hr_mins_np

    lr_speed, lr_wdir = _uv_to_speed_dir(lr_phys[..., 0], lr_phys[..., 1])
    hr_speed, hr_wdir = _uv_to_speed_dir(hr_phys[..., 0], hr_phys[..., 1])

    all_speeds         = np.concatenate([lr_speed.ravel(), hr_speed.ravel()])
    speed_vmin, speed_vmax = np.nanpercentile(all_speeds, [1, 99])

    lon_2d, lat_2d = np.meshgrid(lon_hr, lat_hr)
    sk = quiver_skip
    extent = [float(lon_hr.min()), float(lon_hr.max()),
              float(lat_hr.min()), float(lat_hr.max())]

    row_labels = ["ERA5  (LR bilinear up.)", "CERRA  (HR ground truth)"]
    col_labels = ["Wind Speed  [m/s]",
                  "Wind Direction  [° FROM North, CW]",
                  "Vector Field  (direction + speed)"]
    rows_data = [
        (lr_phys[..., 0], lr_phys[..., 1], lr_speed, lr_wdir),
        (hr_phys[..., 0], hr_phys[..., 1], hr_speed, hr_wdir),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(22, 12),
                             gridspec_kw={"hspace": 0.30, "wspace": 0.18},
                             layout="constrained")
    fig.suptitle(
        f"[{mode.upper()}]  {metric_name.upper()} = {metric_val:.4f}"
        f"   —   {timestamp.strftime('%Y-%m-%d  %H:%M UTC')}"
        f"   —   Speed / Direction / Vectors",
        y=1.01,
    )

    for row_idx, (u_phys, v_phys, speed, wdir) in enumerate(rows_data):
        norm_u = u_phys / (speed + 1e-8)
        norm_v = v_phys / (speed + 1e-8)

        ax = axes[row_idx, 0]
        im0 = ax.imshow(speed, origin="upper", extent=extent, aspect="auto",
                        cmap="viridis", vmin=speed_vmin, vmax=speed_vmax,
                        interpolation="nearest")
        if row_idx == 0:
            ax.set_title(col_labels[0], pad=6)
        ax.set_ylabel(row_labels[row_idx] + "\nLatitude [°N]")
        ax.set_xlabel("Longitude [°E]")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f°"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f°"))
        fig.colorbar(im0, ax=ax, fraction=0.032, pad=0.04, label="m/s")

        ax = axes[row_idx, 1]
        im1 = ax.imshow(wdir, origin="upper", extent=extent, aspect="auto",
                        cmap="hsv", vmin=0, vmax=360, interpolation="nearest")
        if row_idx == 0:
            ax.set_title(col_labels[1], pad=6)
        ax.set_ylabel("Latitude [°N]")
        ax.set_xlabel("Longitude [°E]")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f°"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f°"))
        cbar1 = fig.colorbar(im1, ax=ax, fraction=0.032, pad=0.04,
                             label="° (FROM N, CW)")
        cbar1.set_ticks([0, 90, 180, 270, 360])
        cbar1.set_ticklabels(["0° N", "90° E", "180° S", "270° W", "360° N"])

        ax = axes[row_idx, 2]
        im2 = ax.imshow(speed, origin="upper", extent=extent, aspect="auto",
                        cmap="viridis", vmin=speed_vmin, vmax=speed_vmax,
                        interpolation="nearest", alpha=0.85)
        ax.quiver(lon_2d[::sk, ::sk], lat_2d[::sk, ::sk],
                  norm_u[::sk, ::sk], norm_v[::sk, ::sk],
                  color="white", scale=30, width=0.003,
                  headwidth=4, headlength=4, alpha=0.90)
        if row_idx == 0:
            ax.set_title(col_labels[2], fontsize=13, pad=6)
        ax.set_ylabel("Latitude [°N]", fontsize=11)
        ax.set_xlabel("Longitude [°E]", fontsize=11)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f°"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f°"))
        fig.colorbar(im2, ax=ax, fraction=0.032, pad=0.04, label="m/s")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_temporal_distributions(worst, best, out_base, metrics=("psnr_db", "ssim")):
    """Save month and hour histograms of the worst and best samples of each metric.

    Args:
        worst, best: outputs of get_extreme_positions.
        out_base: output folder.
        metrics: metric names.
    """
    import calendar

    os.makedirs(out_base, exist_ok=True)

    MONTH_LABELS = [calendar.month_abbr[m] for m in range(1, 13)]
    HOUR_BINS    = [0, 3, 6, 9, 12, 15, 18, 21]
    HOUR_LABELS  = [f"{h:02d}" for h in HOUR_BINS]
    MODE_COLOUR  = {"worst": "#8B1A1A", "best": "#1A4A2E"}

    panels = [
        ("months", "Month",         list(range(1, 13)), MONTH_LABELS,
         lambda df: pd.to_datetime(df["datetime"]).dt.month.tolist()),
        ("hours",  "Time [Hours]",  HOUR_BINS,          HOUR_LABELS,
         lambda df: pd.to_datetime(df["datetime"]).dt.hour.tolist()),
    ]
    METRIC_LABELS = {"psnr_db": "PSNR [dB]", "ssim": "SSIM [-]"}

    for fig_stem, xlabel, bins, bin_labels, extractor in panels:
        for metric in metrics:
            metric_label = METRIC_LABELS.get(metric, metric.upper())
            fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                                     layout="constrained", sharey=True)

            for col_idx, (mode, data_dict) in enumerate(
                [("worst", worst), ("best", best)]
            ):
                ax     = axes[col_idx]
                values = extractor(data_dict[metric])
                color  = MODE_COLOUR[mode]
                counts = {b: 0 for b in bins}
                for v in values:
                    if v in counts:
                        counts[v] += 1
                heights = [counts[b] for b in bins]
                x_pos   = range(len(bins))

                ax.bar(x_pos, heights, color=color, edgecolor="black",
                       linewidth=0.7, alpha=0.88)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(
                    bin_labels,
                    rotation=40 if fig_stem == "months" else 0,
                    ha="right" if fig_stem == "months" else "center",
                )
                ax.set_xlabel(xlabel)
                if col_idx == 0:
                    ax.set_ylabel("Count")
                ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                ax.set_axisbelow(True)

            save_path = os.path.join(
                out_base, f"distribution_{fig_stem}_{metric_label}.pdf"
            )
            plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
            plt.close(fig)
            print(f"  Saved distribution plot → {save_path}")


def plot_metric_distributions(csv_path, out_base, metrics=("psnr_db", "ssim")):
    """Save the histogram of each metric of a case, with its mean and the 95 % interval.

    Args:
        csv_path: temporal metrics CSV.
        out_base: output folder.
        metrics: metric columns.
    """
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    METRIC_LABELS = {"psnr_db": "PSNR [dB]", "ssim": "SSIM [-]"}

    for metric in metrics:
        values = df[metric].dropna().values
        mu     = values.mean()
        sigma  = values.std()
        lo     = mu - 1.96 * sigma
        hi     = mu + 1.96 * sigma

        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        ax.hist(values, bins=60, color="#1C3A5E", edgecolor="#0a0a0a",
                linewidth=0.6, alpha=0.90, zorder=2)
        ax.axvline(mu, color="#8B1A1A", linewidth=2.2, linestyle="-",
                   label=f"Mean = {mu:.4f}")
        ax.axvline(lo, color="#6B3A0F", linewidth=1.8, linestyle="--")
        ax.axvline(hi, color="#6B3A0F", linewidth=1.8, linestyle="--")
        ax.axvspan(lo, hi, alpha=0.12, color="#6B3A0F", zorder=1)

        metric_label = METRIC_LABELS.get(metric, metric.upper())
        ax.set_xlabel(metric_label)
        ax.set_ylabel("Count")
        ax.set_axisbelow(True)

        save_path = os.path.join(out_base, f"distribution_{metric}.pdf")
        os.makedirs(out_base, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
        plt.close(fig)
        print(f"  Saved metric distribution → {save_path}")


def plot_metric_distributions_compare(
    csv_path_1:  str,
    csv_path_2:  str,
    label_1:     str,
    label_2:     str,
    color_1:     str,
    color_2:     str,
    out_base:    str,
    metrics:     tuple = ("psnr_db", "ssim"),
):
    """Save overlaid histograms of each metric for two cases.

    Prints the mean, standard deviation, variance and RMS of each case and marks the means.

    Args:
        csv_path_1, csv_path_2: temporal metrics CSV files.
        label_1, label_2: legend labels.
        color_1, color_2: colours.
        out_base: output folder.
        metrics: metric columns.
    """
    df1 = pd.read_csv(csv_path_1, parse_dates=["datetime"])
    df2 = pd.read_csv(csv_path_2, parse_dates=["datetime"])

    METRIC_LABELS = {"psnr_db": "PSNR [dB]", "ssim": "SSIM [-]"}

    for metric in metrics:
        v1 = df1[metric].dropna().values
        v2 = df2[metric].dropna().values

        # Mean, standard deviation and RMS of each case
        mu1,  mu2  = v1.mean(),                v2.mean()
        std1, std2 = v1.std(),                 v2.std()
        var1, var2 = v1.var(),                 v2.var()
        rms1, rms2 = np.sqrt(np.mean(v1**2)), np.sqrt(np.mean(v2**2))

        print(f"\n── {metric.upper()} — two-case comparison ──")
        print(f"  {'Case':10s}  {'N':>6s}  {'mean':>8s}  {'std':>8s}  "
              f"{'variance':>10s}  {'RMS':>8s}")
        print(f"  {label_1:10s}  {len(v1):>6d}  {mu1:>8.4f}  {std1:>8.4f}  "
              f"{var1:>10.6f}  {rms1:>8.4f}")
        print(f"  {label_2:10s}  {len(v2):>6d}  {mu2:>8.4f}  {std2:>8.4f}  "
              f"{var2:>10.6f}  {rms2:>8.4f}")

        metric_label = METRIC_LABELS.get(metric, metric.upper())

        # Shared bin edges for a direct comparison
        bin_edges = np.histogram_bin_edges(
            np.concatenate([v1, v2]), bins=60
        )

        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")

        # Draw both histograms
        for vals, mu, std, rms, color, label in [
            (v1, mu1, std1, rms1, color_1, label_1),
            (v2, mu2, std2, rms2, color_2, label_2),
        ]:
            lo = mu - 1.96 * std
            hi = mu + 1.96 * std

            ax.hist(
                vals, bins=bin_edges,
                color=color, edgecolor="#0a0a0a",
                linewidth=0.5, alpha=0.55, zorder=2,
                label=f"{label}  (RMS = {rms:.4f})",
            )
            ax.axvline(mu, color=color, linewidth=2.2, linestyle="-",
                       zorder=4)
            ax.axvline(lo, color=color, linewidth=1.6, linestyle="--",
                       zorder=4)
            ax.axvline(hi, color=color, linewidth=1.6, linestyle="--",
                       zorder=4)
            ax.axvspan(lo, hi, alpha=0.09, color=color, zorder=1)

        ax.set_xlabel(metric_label)
        ax.set_ylabel("Count")
        ax.legend(fontsize=14)
        ax.set_axisbelow(True)

        save_path = os.path.join(out_base, f"distribution_compare_{metric}.pdf")
        os.makedirs(out_base, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
        plt.close(fig)
        print(f"  Saved → {save_path}")


def plot_temporal_distributions_compare(
    worst_1:  dict,
    best_1:   dict,
    worst_2:  dict,
    best_2:   dict,
    label_1:  str,
    label_2:  str,
    color_1:  str,
    color_2:  str,
    out_base: str,
    metrics:  tuple = ("psnr_db", "ssim"),
):
    """Save grouped bars of the month and hour distribution of the worst and best samples of two cases.

    Args:
        worst_1, best_1, worst_2, best_2: outputs of get_extreme_positions for each case.
        label_1, label_2: legend labels.
        color_1, color_2: colours.
        out_base: output folder.
        metrics: metric names.
    """
    import calendar

    os.makedirs(out_base, exist_ok=True)

    MONTH_LABELS  = [calendar.month_abbr[m] for m in range(1, 13)]
    HOUR_BINS     = [0, 3, 6, 9, 12, 15, 18, 21]
    HOUR_LABELS   = [f"{h:02d}" for h in HOUR_BINS]
    METRIC_LABELS = {"psnr_db": "PSNR [dB]", "ssim": "SSIM [-]"}

    panels = [
        ("months", "Month",        list(range(1, 13)), MONTH_LABELS,
         lambda df: pd.to_datetime(df["datetime"]).dt.month.tolist()),
        ("hours",  "Time [Hours]", HOUR_BINS,          HOUR_LABELS,
         lambda df: pd.to_datetime(df["datetime"]).dt.hour.tolist()),
    ]

    bar_w = 0.38  # Bar width

    for fig_stem, xlabel, bins, bin_labels, extractor in panels:
        for metric in metrics:
            metric_label = METRIC_LABELS.get(metric, metric.upper())

            fig, axes = plt.subplots(
                1, 2,
                figsize=(15, 5),
                layout="constrained",
                sharey=True,
            )

            for col_idx, (mode, d1, d2) in enumerate(
                [("worst", worst_1, worst_2), ("best", best_1, best_2)]
            ):
                ax = axes[col_idx]

                for offset, vals_dict, color, label in [
                    (-bar_w / 2, d1, color_1, label_1),
                    (+bar_w / 2, d2, color_2, label_2),
                ]:
                    values = extractor(vals_dict[metric])
                    counts = {b: 0 for b in bins}
                    for v in values:
                        if v in counts:
                            counts[v] += 1
                    heights = [counts[b] for b in bins]
                    x_pos   = np.arange(len(bins))

                    ax.bar(
                        x_pos + offset, heights,
                        width=bar_w, color=color, edgecolor="black",
                        linewidth=0.7, alpha=0.82, label=label,
                    )

                ax.set_xticks(np.arange(len(bins)))
                ax.set_xticklabels(
                    bin_labels,
                    rotation=40 if fig_stem == "months" else 0,
                    ha="right" if fig_stem == "months" else "center",
                )
                ax.set_xlabel(xlabel)

                if col_idx == 0:
                    ax.set_ylabel("Count")
                    ax.legend(fontsize=13)

                ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                ax.set_axisbelow(True)

            save_path = os.path.join(
                out_base,
                f"distribution_compare_{fig_stem}_{metric_label}.pdf",
            )
            plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
            plt.close(fig)
            print(f"  Saved temporal comparison → {save_path}")


def _run_pass(mode, extremes, retrieved, minmax, lat_hr, lon_hr, out_root):
    """Save the speed and direction figures of the extreme samples of each metric.

    Args:
        mode: "worst" or "best".
        extremes: output of get_extreme_positions.
        retrieved: output of fetch_from_test_ds.
        minmax: min/max constants of the pipeline.
        lat_hr, lon_hr: coordinate vectors of the HR grid.
        out_root: output folder (PSNR and SSIM subfolders).
    """
    metric_dirs = {
        "psnr_db": os.path.join(out_root, "PSNR"),
        "ssim":    os.path.join(out_root, "SSIM"),
    }
    for metric_name, out_dir in metric_dirs.items():
        os.makedirs(out_dir, exist_ok=True)
        df = extremes[metric_name]
        print(f"\n── Plotting {len(df)} {mode} '{metric_name}' samples ──")

        for row in df.itertuples(index=False):
            pos = row.row_pos
            val = getattr(row, metric_name)
            ts  = pd.Timestamp(row.datetime)

            if pos not in retrieved:
                print(f"  ⚠  row_pos={pos} not retrieved — skipping.")
                continue

            stem = _make_filename(metric_name, val, ts)
            plot_sample_speed_dir(
                lr_norm     = retrieved[pos]["lr"],
                hr_norm     = retrieved[pos]["hr"],
                minmax      = minmax,
                lat_hr      = lat_hr,
                lon_hr      = lon_hr,
                metric_name = metric_name,
                metric_val  = val,
                timestamp   = ts,
                save_path   = os.path.join(out_dir, f"{stem}_sdv.pdf"),
                mode        = mode,
            )

    print(f"  → Saved to: {out_root}")


def main():
    """Compare the two cases: overlaid histograms and temporal distributions of the extreme samples."""

    # Publication style
    plt.rcParams.update({
        "font.family":         "sans-serif",
        "font.sans-serif":     ["Helvetica Neue", "Helvetica", "Arial",
                                "DejaVu Sans"],
        "font.size":           25,
        "axes.labelsize":      25,
        "axes.titlesize":      20,
        "xtick.labelsize":     20,
        "ytick.labelsize":     20,
        "axes.linewidth":      1.8,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.facecolor":      "white",
        "figure.facecolor":    "white",
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "xtick.major.size":    5,
        "ytick.major.size":    5,
        "xtick.major.width":   1.4,
        "ytick.major.width":   1.4,
        "xtick.minor.visible": True,
        "xtick.minor.size":    3,
        "ytick.minor.visible": True,
        "ytick.minor.size":    3,
        "lines.linewidth":     2.0,
        "axes.grid":           False,
        "pdf.fonttype":        42,
        "ps.fonttype":         42,
        "savefig.dpi":         300,
    })
    matplotlib.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["STIXGeneral"],
        "mathtext.fontset": "stix",
    })

    # Coords
    coords_dir = os.path.join(SHARED_DIR, "coords")
    lat_hr = np.loadtxt(os.path.join(coords_dir, "lat_HR.npy"))
    lon_hr = np.loadtxt(os.path.join(coords_dir, "lon_HR.npy"))
    print(f"Loaded HR grid: lat {lat_hr.shape}  lon {lon_hr.shape}")

    # CSV paths
    csv_path_1 = os.path.join(RESULTS_DIR,   "temporal_metrics.csv")
    csv_path_2 = os.path.join(RESULTS_DIR_2, "temporal_metrics.csv")
    print(f"\nCSV 1 ({CASE_LABELS[CASE]}):   {csv_path_1}")
    print(f"CSV 2 ({CASE_LABELS[CASE_2]}):   {csv_path_2}")

    # Extreme positions for both cases
    print(f"\nFinding {N_SAMPLES} worst/best for {CASE_LABELS[CASE]} ...")
    worst_1 = get_extreme_positions(csv_path_1, n=N_SAMPLES,
                                    metrics=("psnr_db", "ssim"), mode="worst")
    best_1  = get_extreme_positions(csv_path_1, n=N_SAMPLES,
                                    metrics=("psnr_db", "ssim"), mode="best")

    print(f"\nFinding {N_SAMPLES_2} worst/best for {CASE_LABELS[CASE_2]} ...")
    worst_2 = get_extreme_positions(csv_path_2, n=N_SAMPLES_2,
                                    metrics=("psnr_db", "ssim"), mode="worst")
    best_2  = get_extreme_positions(csv_path_2, n=N_SAMPLES_2,
                                    metrics=("psnr_db", "ssim"), mode="best")

    # Two-case overlaid metric histograms (with RMS printed + in legend)
    print("\n" + "═" * 60)
    print("  Overlaid metric histograms  (4-year vs 2-year)")
    print("═" * 60)
    plot_metric_distributions_compare(
        csv_path_1 = csv_path_1,
        csv_path_2 = csv_path_2,
        label_1    = CASE_LABELS[CASE],
        label_2    = CASE_LABELS[CASE_2],
        color_1    = CASE_COLORS[CASE],
        color_2    = CASE_COLORS[CASE_2],
        out_base   = OUT_BASE,
    )

    # Two-case temporal distributions (month + hour, worst | best columns)
    print("\n" + "═" * 60)
    print("  Temporal distributions  (4-year vs 2-year)")
    print("═" * 60)
    plot_temporal_distributions_compare(
        worst_1  = worst_1,
        best_1   = best_1,
        worst_2  = worst_2,
        best_2   = best_2,
        label_1  = CASE_LABELS[CASE],
        label_2  = CASE_LABELS[CASE_2],
        color_1  = CASE_COLORS[CASE],
        color_2  = CASE_COLORS[CASE_2],
        out_base = OUT_BASE,
    )

    print(f"\n✓  All comparison figures saved under: {OUT_BASE}/")

    if PLOT_SINGLE_CASE_FIGURES:
        plot_temporal_distributions(worst_1, best_1, out_base=OUT_BASE + "_4year")
        plot_metric_distributions(csv_path_1, out_base=OUT_BASE + "_4year")
        plot_temporal_distributions(worst_2, best_2, out_base=OUT_BASE + "_2year")
        plot_metric_distributions(csv_path_2, out_base=OUT_BASE + "_2year")


if __name__ == "__main__":
    main()
