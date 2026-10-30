"""Spatial maps of SSIM, PSNR and wind statistics over a test set.

plot_spatial_metric_maps computes, for every pixel and averaged over the test samples:
    SSIM: uniform sliding window, computed with scipy.ndimage.uniform_filter on chunks of samples.
    PSNR: from the mean squared error accumulated in chunks.
    Wind speed mean and standard deviation, and wind direction (meteorological, direction the
    wind comes from) circular mean and standard deviation, of the ground truth.
It saves the six maps as .npy files and one figure; colour scales use QuantileNorm so that each
quartile of the values takes a quarter of the colour range. plot_spatial_metric_maps_from_npy
redraws the figure from the saved maps without inference.

Usage after an evaluation (arrays of the whole test set):
    plot_spatial_metric_maps(hr_pred_scaled, hr_true_scaled, hr_true_unscaled,
                             lat_hr, lon_hr, results_dir, case, win_size=11, uv_comp=True)

Author: Marion Falkenauer (with the help of Claude)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import uniform_filter


class QuantileNorm(mcolors.Normalize):
    """Normalisation mapping each value to its percentile rank, so every quartile takes 25 % of the colormap."""

    def __init__(self, data: np.ndarray, n_quantiles: int = 2000,
                 clip: bool = False):
        """Store the quantiles of the data.

        Args:
            data: array used to compute the quantiles (non-finite values are ignored).
            n_quantiles: number of quantile levels.
            clip: clip values outside the data range.
        """
        valid = data[np.isfinite(data)].ravel()
        super().__init__(vmin=float(valid.min()), vmax=float(valid.max()),
                         clip=clip)
        q = np.linspace(0.0, 100.0, n_quantiles)
        self._qvals  = np.percentile(valid, q).astype(np.float64)
        self._qnorms = (q / 100.0).astype(np.float64)

    def __call__(self, value, clip=None):
        """Map values to their percentile rank in [0, 1]."""
        arr    = np.asarray(value, dtype=np.float64)
        scaled = np.interp(arr, self._qvals, self._qnorms)
        return np.ma.masked_array(scaled, mask=~np.isfinite(arr))

    def inverse(self, value):
        """Map positions in [0, 1] back to data values."""
        return np.interp(np.asarray(value, dtype=np.float64),
                         self._qnorms, self._qvals)


def _add_quartile_ticks(cb, arr: np.ndarray, n_decimal: int = 3) -> None:
    """Set the colourbar ticks at the five quartile values (min, Q1, median, Q3, max).

    Args:
        cb: matplotlib colourbar built with a QuantileNorm.
        arr: array used to compute the quartiles.
        n_decimal: number of decimals of the tick labels.
    """
    percs = np.percentile(arr[np.isfinite(arr)].ravel(), [0, 25, 50, 75, 100])
    cb.set_ticks(percs.tolist())
    fmt = f"{{:.{n_decimal}f}}"
    cb.set_ticklabels([fmt.format(v) for v in percs])


def _ssim_map_batch_accum(
    true_all:   np.ndarray,
    pred_all:   np.ndarray,
    data_range: float = 1.0,
    win_size:   int   = 11,
    chunk_size: int   = 64,
) -> np.ndarray:
    """Mean SSIM map of a channel over all samples.

    Args:
        true_all, pred_all: arrays (N, H, W).
        data_range: dynamic range of the data (1.0 for data scaled to [0, 1]).
        win_size: side of the uniform window.
        chunk_size: number of samples per vectorised pass.
    Returns:
        float array (H, W).
    """
    N, H, W = true_all.shape
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    # Filter size: leave the batch axis (axis-0) untouched
    fsize = (1, win_size, win_size)

    acc = np.zeros((H, W), dtype=np.float64)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        x = true_all[start:end].astype(np.float64)  # (B, H, W)
        y = pred_all[start:end].astype(np.float64)

        # Local means
        mu_x = uniform_filter(x, size=fsize, mode="reflect")
        mu_y = uniform_filter(y, size=fsize, mode="reflect")

        # Local variances and covariance (clamp negatives from floating point)
        sig_x2 = np.maximum(
            uniform_filter(x * x, size=fsize, mode="reflect") - mu_x ** 2,
            0.0)
        sig_y2 = np.maximum(
            uniform_filter(y * y, size=fsize, mode="reflect") - mu_y ** 2,
            0.0)
        sig_xy = (uniform_filter(x * y, size=fsize, mode="reflect")
                  - mu_x * mu_y)

        # SSIM per pixel per sample in this chunk
        ssim_chunk = (
            (2.0 * mu_x * mu_y + C1) * (2.0 * sig_xy  + C2) /
            ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x2 + sig_y2 + C2))
        )
        acc += ssim_chunk.sum(axis=0)  # accumulate over batch

    return (acc / N).astype(np.float32)


def _wind_circular_mean_std(
    u_all:      np.ndarray,
    v_all:      np.ndarray,
    chunk_size: int = 256,
) -> tuple:
    """Pixel-wise wind speed and direction statistics over the samples.

    Args:
        u_all, v_all: arrays (N, H, W) of the u and v components in m/s.
        chunk_size: number of samples per pass.
    Returns:
        (ws_mean, ws_std, wdir_mean, wdir_std): arrays (H, W); the direction is in degrees (circular
        mean and circular standard deviation).
    """
    N, H, W = u_all.shape
    sin_acc = np.zeros((H, W), dtype=np.float64)
    cos_acc = np.zeros((H, W), dtype=np.float64)
    ws_acc  = np.zeros((H, W), dtype=np.float64)
    ws2_acc = np.zeros((H, W), dtype=np.float64)

    for start in range(0, N, chunk_size):
        end  = min(start + chunk_size, N)
        u64  = u_all[start:end].astype(np.float64)  # (B, H, W)
        v64  = v_all[start:end].astype(np.float64)

        ws   = np.sqrt(u64 ** 2 + v64 ** 2)
        # Meteorological FROM-direction: 0 deg = from North, 90 deg = from East
        theta_rad = np.deg2rad(
            (270.0 - np.degrees(np.arctan2(v64, u64))) % 360.0)

        sin_acc += np.sin(theta_rad).sum(axis=0)
        cos_acc += np.cos(theta_rad).sum(axis=0)
        ws_acc  += ws.sum(axis=0)
        ws2_acc += (ws ** 2).sum(axis=0)

    # Circular mean direction
    sin_m    = sin_acc / N
    cos_m    = cos_acc / N
    wdir_mean = (np.degrees(np.arctan2(sin_m, cos_m)) % 360.0).astype(np.float32)

    # Circular std: sqrt(-2*ln R), R = mean resultant length
    R        = np.clip(np.sqrt(sin_m ** 2 + cos_m ** 2), 1e-10, 1.0)
    wdir_std = np.degrees(
        np.sqrt(np.maximum(-2.0 * np.log(R), 0.0))).astype(np.float32)

    ws_mean = (ws_acc  / N).astype(np.float32)
    ws_var  = np.maximum(ws2_acc / N - (ws_acc / N) ** 2, 0.0)
    ws_std  = ws_var.astype(np.float32) ** 0.5

    return wdir_mean, wdir_std, ws_mean, ws_std


def plot_spatial_metric_maps(
    hr_pred_scaled:   np.ndarray,
    hr_true_scaled:   np.ndarray,
    hr_true_unscaled: np.ndarray,
    lat_hr:           np.ndarray,
    lon_hr:           np.ndarray,
    results_dir:      str   = ".",
    case:             str   = "",
    win_size:         int   = 11,
    uv_comp:          bool  = True,
    save_npy:         bool  = True,
    chunk_size:       int   = 64,
    data_range:       float = 1.0,
) -> dict:
    """Compute and plot the spatial SSIM, PSNR and wind statistic maps.

    Writes spatial_metric_maps<_case>.pdf, the individual SSIM and PSNR figures and, with save_npy,
    maps_<name><_case>.npy for ssim_mean, psnr_map, ws_mean, ws_std, wdir_mean and wdir_std.

    Args:
        hr_pred_scaled, hr_true_scaled: arrays (N, H, W, C) scaled to [0, 1].
        hr_true_unscaled: ground truth (N, H, W, 2) in m/s (u, v).
        lat_hr, lon_hr: coordinate vectors of the HR grid.
        results_dir: output directory.
        case: label added to titles and file names.
        win_size: SSIM window side (11 as scikit-image, 7 as tf.image.ssim).
        uv_comp: True averages SSIM and PSNR over u and v, False uses u only.
        save_npy: save the maps as .npy files.
        chunk_size: samples per vectorised pass (reduce to save memory).
        data_range: dynamic range of the scaled data.
    Returns:
        Dictionary of the six maps (H, W).
    """
    os.makedirs(results_dir, exist_ok=True)
    N, H, W, C = hr_pred_scaled.shape
    tag         = f"_{case}" if case else ""
    print(f"\n[plot_spatial_metric_maps]  N={N}  H={H}  W={W}  C={C}  "
          f"case={case!r}  win_size={win_size}  chunk_size={chunk_size}")

    # SSIM spatial map
    print("  Step 1/3 — vectorised SSIM maps ...")
    ssim_u = _ssim_map_batch_accum(
        hr_true_scaled[..., 0], hr_pred_scaled[..., 0],
        data_range=data_range, win_size=win_size, chunk_size=chunk_size,
    )
    if uv_comp and C >= 2:
        ssim_v    = _ssim_map_batch_accum(
            hr_true_scaled[..., 1], hr_pred_scaled[..., 1],
            data_range=data_range, win_size=win_size, chunk_size=chunk_size,
        )
        ssim_mean = 0.5 * (ssim_u + ssim_v)
    else:
        ssim_mean = ssim_u.copy()
    print(f"    SSIM  min={ssim_mean.min():.4f}  "
          f"median={np.median(ssim_mean):.4f}  max={ssim_mean.max():.4f}")

    # PSNR spatial map (chunked MSE accumulation)
    print("  Step 2/3 — pixel-wise PSNR map ...")
    mse_acc_u = np.zeros((H, W), dtype=np.float64)
    mse_acc_v = np.zeros((H, W), dtype=np.float64) if (uv_comp and C >= 2) else None
    for start in range(0, N, chunk_size):
        end  = min(start + chunk_size, N)
        diff = (hr_pred_scaled[start:end].astype(np.float64)
                - hr_true_scaled[start:end].astype(np.float64))
        mse_acc_u += (diff[..., 0] ** 2).sum(axis=0)
        if mse_acc_v is not None:
            mse_acc_v += (diff[..., 1] ** 2).sum(axis=0)

    mse_map = mse_acc_u / N
    if mse_acc_v is not None:
        mse_map = 0.5 * (mse_map + mse_acc_v / N)
    psnr_map = (10.0 * np.log10(data_range ** 2
                                / (mse_map + 1e-10))).astype(np.float32)
    print(f"    PSNR  min={psnr_map.min():.2f} dB  "
          f"median={np.median(psnr_map):.2f} dB  max={psnr_map.max():.2f} dB")

    # Wind statistics
    print("  Step 3/3 — wind speed & direction statistics ...")
    wdir_mean, wdir_std, ws_mean, ws_std = _wind_circular_mean_std(
        hr_true_unscaled[..., 0], hr_true_unscaled[..., 1],
        chunk_size=min(chunk_size * 4, 256),
    )
    print(f"    Wind speed mean  min={ws_mean.min():.2f}  "
          f"max={ws_mean.max():.2f} m/s")
    print(f"    Wind dir  mean   min={wdir_mean.min():.1f}°  "
          f"max={wdir_mean.max():.1f}°")
    print(f"    Wind dir  std    min={wdir_std.min():.1f}°  "
          f"max={wdir_std.max():.1f}°")

    # Save .npy
    maps = {
        "ssim_mean": ssim_mean,
        "psnr_map":  psnr_map,
        "ws_mean":   ws_mean,
        "ws_std":    ws_std,
        "wdir_mean": wdir_mean,
        "wdir_std":  wdir_std,
    }
    if save_npy:
        for name, arr in maps.items():
            path = os.path.join(results_dir, f"maps_{name}{tag}.npy")
            np.save(path, arr)
            print(f"    Saved {path}")

    # Render figure
    _render_figure(
        maps=maps, lat_hr=lat_hr, lon_hr=lon_hr,
        results_dir=results_dir, case=case, tag=tag,
        N=N, win_size=win_size, uv_comp=uv_comp,
    )

    return maps


def plot_spatial_metric_maps_from_npy(
    results_dir: str,
    lat_hr:      np.ndarray,
    lon_hr:      np.ndarray,
    case:        str = "",
) -> None:
    """Redraw the figure from maps saved by plot_spatial_metric_maps.

    Args:
        results_dir: folder containing the maps_*.npy files.
        lat_hr, lon_hr: coordinate vectors of the HR grid.
        case: case label used when the maps were saved.
    """
    tag = f"_{case}" if case else ""
    keys = ["ssim_mean", "psnr_map", "ws_mean", "ws_std",
            "wdir_mean", "wdir_std"]
    maps = {}
    for k in keys:
        path = os.path.join(results_dir, f"maps_{k}{tag}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Map file not found: {path}\n"
                f"Run plot_spatial_metric_maps(...) first to generate it.")
        maps[k] = np.load(path)
        print(f"  Loaded {path}  shape={maps[k].shape}")

    _render_figure(
        maps=maps, lat_hr=lat_hr, lon_hr=lon_hr,
        results_dir=results_dir, case=case, tag=tag,
        N=None, win_size=None, uv_comp=True,
    )


def _render_figure(
    maps:        dict,
    lat_hr:      np.ndarray,
    lon_hr:      np.ndarray,
    results_dir: str,
    case:        str,
    tag:         str,
    N,
    win_size,
    uv_comp:     bool = True,
) -> None:
    """Draw and save the 2 x 3 figure of the maps and the individual SSIM and PSNR figures.

    Args:
        maps: dictionary of the six maps (H, W).
        lat_hr, lon_hr: coordinate vectors of the HR grid.
        results_dir: output directory.
        case: case label for the titles.
        tag: suffix of the output file names.
        N: number of samples (shown in the title, None to omit).
        win_size: SSIM window side (shown in the title, None to omit).
        uv_comp: whether SSIM and PSNR average u and v.
    """

    # Geography
    ext = [float(lon_hr.min()), float(lon_hr.max()),
           float(lat_hr.min()), float(lat_hr.max())]
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        USE_CARTOPY = True
        proj        = ccrs.PlateCarree()
    except ImportError:
        USE_CARTOPY = False
        proj        = None

    subplot_kw = {"projection": proj} if USE_CARTOPY else {}

    # Style
    with plt.rc_context({
        "font.family":      "STIXGeneral",
        "font.size":        11,
        "axes.labelsize":   10,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "axes.titlesize":   11,
        "xtick.direction":  "out",
        "ytick.direction":  "out",
    }):
        fig, axes = plt.subplots(
            2, 3, figsize=(22, 13),
            subplot_kw=subplot_kw,
            gridspec_kw={"hspace": 0.42, "wspace": 0.13},
        )

        title_tag   = f"  [{case}]" if case else ""
        n_str       = f"N = {N} samples  |  " if N is not None else ""
        ws_str      = f"SSIM window = {win_size} px  |  " if win_size else ""
        uv_str      = "U+V avg  |  " if uv_comp else ""
        fig.suptitle(
            f"Spatially-averaged metric & wind-field maps — test set{title_tag}\n"
            f"{n_str}{ws_str}{uv_str}Colorbars: QuantileNorm  (min / Q1 / Q2 / Q3 / max)",
            fontsize=13, y=1.01,
        )

        # (key, title, colormap, unit, n_decimal for ticks)
        # Colormaps: SSIM RdYlGn, PSNR inferno, wind speed mean inferno, wind speed std magma, wind direction mean twilight_shifted, wind direction std YlOrRd
        panels = [
            ("ssim_mean", "Mean SSIM  (U+V avg)",        "RdYlGn",           "SSIM", 3),
            ("ws_mean",   "Wind speed mean  [m/s]",      "inferno",          "m/s",  2),
            ("wdir_mean", "Wind direction mean  [°]",    "twilight_shifted",  "°",    1),
            ("psnr_map",  "Mean PSNR  [dB]",             "inferno",          "dB",   2),
            ("ws_std",    "Wind speed std  [m/s]",       "magma",            "m/s",  2),
            ("wdir_std",  "Wind direction std  [°]",     "YlOrRd",           "°",    1),
        ]

        for idx, (key, title, cmap_name, unit, ndec) in enumerate(panels):
            row, col = divmod(idx, 3)
            ax  = axes[row, col]
            arr = maps[key]

            norm = QuantileNorm(arr)

            im = ax.imshow(
                arr,
                origin="upper", extent=ext, aspect="auto",
                cmap=cmap_name, norm=norm, interpolation="nearest",
            )

            if USE_CARTOPY:
                ax.add_feature(cfeature.BORDERS,
                               linewidth=0.4, alpha=0.75, edgecolor="#555555")
                ax.add_feature(cfeature.COASTLINE,
                               linewidth=0.6, edgecolor="#222222")

            ax.set_title(title, fontsize=11, pad=5)

            if not USE_CARTOPY:
                ax.tick_params(direction="out", top=False, right=False)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                if col == 0:
                    ax.set_ylabel("Latitude", fontsize=9)
                if row == 1:
                    ax.set_xlabel("Longitude", fontsize=9)

            # Colorbar - ticks at actual quartile values
            cb = fig.colorbar(
                im, ax=ax,
                orientation="vertical",
                shrink=0.90, pad=0.02, aspect=28, extend="neither",
            )
            cb.set_label(unit, fontsize=9)
            _add_quartile_ticks(cb, arr, n_decimal=ndec)
            cb.ax.tick_params(labelsize=8)

        out = os.path.join(results_dir, f"spatial_metric_maps{tag}.pdf")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out}")

        # Individual figures
        for key, title, cmap_name, unit, ndec in panels:
            if key not in ("ssim_mean", "psnr_map"):
                continue

            fig_s, ax_s = plt.subplots(
                1, 1, figsize=(10, 8), subplot_kw=subplot_kw)
            arr  = maps[key]
            norm = QuantileNorm(arr)
            im_s = ax_s.imshow(
                arr, origin="upper", extent=ext, aspect="auto",
                cmap=cmap_name, norm=norm, interpolation="nearest",
            )
            if USE_CARTOPY:
                ax_s.add_feature(cfeature.BORDERS,
                                    linewidth=0.4, alpha=0.75,
                                    edgecolor="#555555")
                ax_s.add_feature(cfeature.COASTLINE,
                                    linewidth=0.6, edgecolor="#222222")
            ax_s.set_title(title, fontsize=12, pad=5)
            if not USE_CARTOPY:
                ax_s.tick_params(direction="out", top=False, right=False)
                for sp in ("top", "right"):
                    ax_s.spines[sp].set_visible(False)
                ax_s.set_ylabel("Latitude",  fontsize=10)
                ax_s.set_xlabel("Longitude", fontsize=10)
            cb_s = fig_s.colorbar(
                im_s, ax=ax_s, orientation="vertical",
                shrink=0.90, pad=0.02, aspect=28)
            cb_s.set_label(unit, fontsize=10)
            _add_quartile_ticks(cb_s, arr, n_decimal=ndec)
            cb_s.ax.tick_params(labelsize=9)
            out_s = os.path.join(results_dir, f"{key}{tag}.pdf")
            fig_s.savefig(out_s, dpi=300, bbox_inches="tight")
            plt.close(fig_s)
            print(f"  Saved → {out_s}")
    print("[plot_spatial_metric_maps] Done.\n")
