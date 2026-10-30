"""Figures comparing the downscaling algorithms: spectra, test metrics, inference time and time series.

Inputs, for each algorithm and training case, in the result folders written by the evaluation
scripts: average_fft_spectra.csv, test_metrics.csv, computation_time_runs.csv and temporal_metrics.csv
(paths are built from the *_TEMPLATE constants). Spectra are horizontal kinetic energy spectra
E_HKE(k) = 0.5 * (PSD_u + PSD_v) * dx against the wavenumber in 1/m (dx = DX metres per pixel).

Output figures, in OUT_DIR:
    fig_<algo>_years.pdf: spectrum of each algorithm, one curve per training case.
    fig_2year_all_algorithms.pdf: spectra of all algorithms for one case.
    fig_metrics_by_case.pdf: PSNR and SSIM against training duration.
    fig_time_vs_metrics.pdf: inference time per sample against PSNR and SSIM.
    fig_timeseries_psnr_ssim.pdf: PSNR and SSIM rolling means over the test year.
    fig_baselines_psd.pdf: spectra of ground truth, LR, bicubic and SRCNN.
    fig_bar_variants.pdf, fig_bar_baselines.pdf: PSNR and SSIM bars of the variants and baselines.
    metrics_table.csv: PSNR and SSIM of every algorithm and case, for the LaTeX table.

Configuration (module constants):
    ALL_ALGORITHMS: label -> result folder of every available algorithm; ACTIVE_ALGORITHMS selects the
        ones plotted. CASES: training cases. BASELINES: label -> folder of the baselines.
    TIME_CASE, TEMPORAL_CASE, BASELINE_CASE: cases used for timing, time series and baselines.
    N_TEST_SAMPLES: number of test samples (inference time per sample).
    PLOT_BASELINE_CURVES: add the bicubic and SRCNN curves to the spectra figures.
    INCLUDE_BASELINES_IN_TABLE: add the baselines to metrics_table.csv.
    BAR_FIGURE_FAMILY_STYLE: redraw fig_bar_variants.pdf with family colours and variant hatches.
    PLOT_TIMESERIES_FIGURES: also draw fig_timeseries_psnr.pdf and fig_timeseries_ssim.pdf.
    _BASE, OUT_DIR: result root and output folder, to adapt to the local file system.

Author: Marion Falkenauer (with the help of Claude)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.patches import Patch as _Patch
from matplotlib.ticker import ScalarFormatter

plt.rcParams.update({
    "font.size":       25,
    "axes.labelsize":  25,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "axes.titlesize":  20,
})
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["STIXGeneral"],
    "mathtext.fontset": "stix",
})


CASES: list[str] = ["1year", "2year", "4year"]

ALL_ALGORITHMS: dict[str, str] = {
    "UNet":               "UNet_Eval_Clean",
    "UNet-2t":            "UNet_2t_Eval-clean-all",
    "UNet-Spectral":      "UNet_Spectral_Eval_Full",
    "UNet-Diff":          "UNet_Diff_Eval-All",
    "UNet-Diff-Spectral": "UNet_Diff_Spectral_Eval-All",
    "Swin2SR":            "Swin2SR_Eval",
    "Swin2SR-2t":         "Swin2SR_2t_100eps",
    "Swin2SR-Spectral":   "Swin2SR_Spectral_Eval",
    "Swin2SR-Diff":       "Swin_Diff-50eps_Eval-All",
}
ACTIVE_ALGORITHMS: list[str] = ["UNet", "Swin2SR"]
ALGORITHMS: dict[str, str] = {a: ALL_ALGORITHMS[a] for a in ACTIVE_ALGORITHMS}

BASELINES: dict[str, str] = {
    "Bicubic": "bicubic",
    "SRCNN":   "srcnn",
}

DX: float = 5_500.0

_BASE = (
    "Base of the main folder with all the results to be fetched and plotted in the script"
)
_BASE_BL = _BASE + "/BASELINES"

PATH_TEMPLATE_V1: str = _BASE + "/{algo}/{CASE}/average_fft_spectra.csv"
PATH_TEMPLATE_V2: str = _BASE + "/{algo}/2year/average_fft_spectra.csv"

METRICS_CSV_TEMPLATE:      str = _BASE    + "/{algo}/{CASE}/test_metrics.csv"
TIME_CSV_TEMPLATE:         str = _BASE    + "/{algo}/{CASE}/computation_time_runs.csv"
TEMPORAL_CSV_TEMPLATE:     str = _BASE    + "/{algo}/{CASE}/temporal_metrics.csv"
BASELINE_PSD_TEMPLATE:     str = _BASE_BL + "/{folder}/{CASE}/average_fft_spectra.csv"
BASELINE_METRICS_TEMPLATE: str = _BASE_BL + "/{folder}/{CASE}/test_metrics.csv"

REF_CSV: str = _BASE + "/UNet_Eval_Clean/2year/average_fft_spectra.csv"

N_TEST_SAMPLES = 2928
TIME_CASE      = "2year"
TEMPORAL_CASE  = "2year"
BASELINE_CASE  = "2year"

LEGEND_KW     = dict(fontsize=15, framealpha=0.75, loc="lower left")
PINNED_LABELS = ("HR", "LR Upsampled")

OUT_DIR: str = "./figures_Final_Backbones"

PLOT_BASELINE_CURVES = False
INCLUDE_BASELINES_IN_TABLE = False
BAR_FIGURE_FAMILY_STYLE = False
PLOT_TIMESERIES_FIGURES = False


CSV_COLS = [
    "k_mid_cy_per_px",
    "gt_psd_u", "pred_psd_u", "lr_psd_u",
    "gt_psd_v", "pred_psd_v", "lr_psd_v",
]

ALGO_LABELS: list[str] = list(ALGORITHMS.keys())


def path_v1(case: str, algo_key: str) -> str:
    """Path of the spectra CSV of an algorithm for a case."""
    return PATH_TEMPLATE_V1.format(CASE=case, algo=ALGORITHMS[algo_key])


def path_v2(algo_key: str) -> str:
    """Path of the spectra CSV of an algorithm for the 2-year case."""
    return PATH_TEMPLATE_V2.format(algo=ALGORITHMS[algo_key])


def path_metrics(case: str, algo_key: str) -> str:
    """Path of the test metrics CSV of an algorithm for a case."""
    return METRICS_CSV_TEMPLATE.format(CASE=case, algo=ALGORITHMS[algo_key])


def path_time(case: str, algo_key: str) -> str:
    """Path of the computation time CSV of an algorithm for a case."""
    return TIME_CSV_TEMPLATE.format(CASE=case, algo=ALGORITHMS[algo_key])


def path_temporal(case: str, algo_key: str) -> str:
    """Path of the temporal metrics CSV of an algorithm for a case."""
    return TEMPORAL_CSV_TEMPLATE.format(CASE=case, algo=ALGORITHMS[algo_key])


def path_baseline_psd(case: str, bl_key: str) -> str:
    """Path of the spectra CSV of a baseline for a case."""
    return BASELINE_PSD_TEMPLATE.format(CASE=case, folder=BASELINES[bl_key])


def path_baseline_metrics(case: str, bl_key: str) -> str:
    """Path of the test metrics CSV of a baseline for a case."""
    return BASELINE_METRICS_TEMPLATE.format(CASE=case, folder=BASELINES[bl_key])


def load_csv(filepath: str) -> pd.DataFrame | None:
    """Read a spectra CSV sorted by wavenumber.

    Args:
        filepath: CSV with the columns CSV_COLS.
    Returns:
        DataFrame, or None (with a message) if the file does not exist.
    """
    if not os.path.exists(filepath):
        print(f"  [MISSING]  {filepath}  — skipping")
        return None
    df = pd.read_csv(filepath, usecols=CSV_COLS)
    return df.sort_values("k_mid_cy_per_px").reset_index(drop=True)


def k_to_meters(k_cy_per_px: np.ndarray) -> np.ndarray:
    """Convert the wavenumber from cycles per pixel to 1/m (divide by DX).

    Args:
        k_cy_per_px: wavenumbers in cycles per pixel.
    Returns:
        Wavenumbers in 1/m.
    """
    return k_cy_per_px / DX


def hke(psd_u: np.ndarray, psd_v: np.ndarray) -> np.ndarray:
    """Horizontal kinetic energy spectrum from the PSD of the two components.

    Args:
        psd_u, psd_v: arrays of the PSD of u and v, normalised by 1 / (H * W) per pixel wavenumber.
    Returns:
        0.5 * (psd_u + psd_v) * DX, the spectral density per 1/m.
    """
    return 0.5 * (psd_u + psd_v) * DX


GT_STYLE: dict = dict(color="black", linewidth=2.2, zorder=6)
LR_STYLE: dict = dict(color="black", linewidth=2.2, linestyle="--", zorder=6)

LINESTYLE_MAP: dict[str, str] = {
    "spectral": "-",
    "2t":       "-",
    "default":  "-",
}

ALGO_COLORS: dict[str, str] = {
    "UNet":                  "#C0392B",
    "UNet-2t":               "#D4680F",
    "UNet-Spectral":         "#8E1B11",
    "Swin2SR":               "#1A5EA8",
    "Swin2SR-2t":            "#1A9BAD",
    "Swin2SR-Spectral":      "#4D8FCF",

    "Swin2SR-Diff":          "#1D7A3A",
    "Swin2SR-Diff-Spectral": "#5E8C1A",

    "UNet-Diff":             "#7B2FA0",
    "UNet-Diff-Spectral":     "#7B2FA0",

}
CASE_COLORS = cm.viridis(np.linspace(0.05, 0.88, len(CASES)))

BASELINE_COLORS: dict[str, str] = {
    "Bicubic": "#808080",
    "SRCNN":   "#E67E22",
}

BASELINE_LINE_STYLES: dict[str, dict] = {
    "Bicubic": dict(color="#808080", linewidth=1.8, linestyle=":",  zorder=4),
    "SRCNN":   dict(color="#E67E22", linewidth=1.8, linestyle="-.", zorder=4),
}


def get_linestyle(name: str) -> str:
    """Line style of an algorithm label (all variants use solid lines)."""
    n = name.lower()
    if "spectral" in n: return LINESTYLE_MAP["spectral"]
    if "2t"       in n: return LINESTYLE_MAP["2t"]
    return LINESTYLE_MAP["default"]


def get_marker(algo_key: str) -> str:
    """Marker of an algorithm label: D for Swin2SR diffusion, ^ for U-Net diffusion, s for Swin2SR, o otherwise."""
    k = algo_key.lower()
    if "swin" in k and "diff" in k: return "D"
    if "unet" in k and "diff" in k: return "^"
    if "swin" in k:                 return "s"
    return "o"


def case_shades(base_hex: str, n: int, shift: float = 0.3) -> list[str]:
    """Shades of a colour, from lighter to darker, one per training case.

    Args:
        base_hex: base colour.
        n: number of shades.
        shift: maximum lightening or darkening.
    Returns:
        List of n hex colours.
    """
    base = np.array(mcolors.to_rgb(base_hex))
    mid  = (n - 1) / 2
    if n == 1:
        return [base_hex]
    shades = []
    for i in range(n):
        t = (i - mid) / mid * shift
        if   t < 0: shade = base + abs(t) * (np.ones(3) - base)
        elif t > 0: shade = base * (1 - t)
        else:       shade = base.copy()
        shades.append(mcolors.to_hex(np.clip(shade, 0, 1)))
    return shades


def _compute_legend_order(
    ax:            plt.Axes,
    pinned_top:    tuple[str, ...] = ("HR",),
    pinned_bottom: tuple[str, ...] = ("LR",),
) -> list[int]:
    """Legend order: pinned labels first, then curves sorted by decreasing mean log10 value, pinned bottom labels last.

    Args:
        ax: axis with labelled lines.
        pinned_top, pinned_bottom: labels kept at the top and at the bottom.
    Returns:
        List of handle indices.
    """
    handles, labels = ax.get_legend_handles_labels()
    top_idx    = [i for i, l in enumerate(labels) if l in pinned_top]
    bottom_idx = [i for i, l in enumerate(labels) if l in pinned_bottom]
    other_idx  = [i for i, l in enumerate(labels)
                  if l not in pinned_top and l not in pinned_bottom]
    magnitudes = []
    for i in other_idx:
        try:
            y = np.asarray(handles[i].get_ydata(), dtype=float)
            y = y[np.isfinite(y) & (y > 0)]
            magnitudes.append(np.mean(np.log10(y)) if len(y) else -np.inf)
        except Exception:
            magnitudes.append(-np.inf)
    sorted_other = [other_idx[j] for j in np.argsort(magnitudes)[::-1]]
    return top_idx + sorted_other + bottom_idx


def _apply_legend_order(ax: plt.Axes, order: list[int], **kw) -> None:
    """Draw the legend of an axis in a given order.

    Args:
        ax: axis.
        order: list of handle indices.
        **kw: arguments passed to ax.legend.
    """
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[i] for i in order], [labels[i] for i in order], **kw)


def _style_ax(ax: plt.Axes,
              show_ylabel: bool = True, show_xlabel: bool = True) -> None:
    """Style a spectrum axis: wavenumber and E_HKE labels, no top and right spines, ticks inside."""
    if show_xlabel:
        ax.set_xlabel(r"Wavenumber  $k$  [m$^{-1}$]")
    if show_ylabel:
        ax.set_ylabel(
            r"$E_{\mathrm{HKE}}(k)$"
            r"  $\left[\dfrac{\mathrm{m}^3}{\mathrm{s}^2}\right]$"
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", which="both", direction="in", pad=6)
    ax.tick_params(axis="y", which="both", direction="in")


def _style_metric_ax(ax: plt.Axes, ylabel: str, title: str,
                     show_xlabel: bool = False) -> None:
    """Style a metric axis with a label, a bold title and no top and right spines.

    Args:
        ax: axis.
        ylabel: y label.
        title: title.
        show_xlabel: add the x label "Training duration".
    """
    if show_xlabel:
        ax.set_xlabel("Training duration")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", direction="in", pad=6)


def _load_metrics(algo_key: str, case: str):
    """Last PSNR and SSIM of an algorithm for a case.

    Args:
        algo_key: algorithm label.
        case: training case.
    Returns:
        (psnr, ssim), or (None, None) if the file or the case is missing.
    """
    p = path_metrics(case, algo_key)
    if not os.path.exists(p):
        print(f"  [MISSING]  {p}")
        return None, None
    try:
        df   = pd.read_csv(p)
        rows = df[df["case"].astype(str) == str(case)]
        if rows.empty:
            return None, None
        last = rows.iloc[-1]
        return float(last["test_psnr"]), float(last["test_ssim"])
    except Exception as exc:
        print(f"  [WARNING]  metrics {p}: {exc}")
        return None, None


def _load_last_sampling_time(algo_key: str, case: str):
    """Last sampling time per test sample of an algorithm for a case.

    Args:
        algo_key: algorithm label.
        case: training case.
    Returns:
        Seconds per sample, or None if unavailable.
    """
    p = path_time(case, algo_key)
    if not os.path.exists(p):
        print(f"  [MISSING]  {p}")
        return None
    try:
        df = pd.read_csv(p)
        rows = df[df["case"].astype(str) == str(case)] \
               if "case" in df.columns else df
        if rows.empty:
            return None
        last = rows.iloc[-1]
        if "sampling_s" in last.index:
            col = "sampling_s"
        elif "sampling_seconds" in last.index:
            col = "sampling_seconds"
        else:
            print(f"  [WARNING]  no sampling column in {p}")
            return None
        return float(last[col]) / N_TEST_SAMPLES
    except Exception as exc:
        print(f"  [WARNING]  timing {p}: {exc}")
        return None


def _load_baseline_metrics(bl_key: str, case: str):
    """Last PSNR, SSIM and scaled MSE of a baseline.

    Args:
        bl_key: baseline label.
        case: training case.
    Returns:
        (psnr, ssim, mse_scaled), or (None, None, None) if unavailable.
    """
    p = path_baseline_metrics(case, bl_key)
    if not os.path.exists(p):
        print(f"  [MISSING]  {p}")
        return None, None, None
    try:
        df   = pd.read_csv(p)
        last = df.iloc[-1]
        return (
            float(last["test_psnr"]),
            float(last["test_ssim"]),
            float(last["test_mse_scaled"]),
        )
    except Exception as exc:
        print(f"  [WARNING]  baseline metrics {p}: {exc}")
        return None, None, None


def export_metrics_csv(out_path: str) -> None:
    """Write PSNR and SSIM of every algorithm and case to a wide CSV.

    Columns: algorithm, psnr_<case> and ssim_<case> for each case; missing values are "--".

    Args:
        out_path: output CSV.
    """
    rows = []
    for algo in ALGO_LABELS:
        row: dict = {"algorithm": algo}
        for case in CASES:
            psnr, ssim = _load_metrics(algo, case)
            row[f"psnr_{case}"] = f"{psnr:.2f}" if psnr is not None else "--"
            row[f"ssim_{case}"] = f"{ssim:.4f}" if ssim is not None else "--"
        rows.append(row)
    # Append baseline rows
    if INCLUDE_BASELINES_IN_TABLE:
        for bl_key in BASELINES:
            psnr, ssim, mse = _load_baseline_metrics(bl_key, BASELINE_CASE)
            row = {"algorithm": bl_key}
            for case in CASES:
                if case == BASELINE_CASE:
                    row[f"psnr_{case}"] = f"{psnr:.2f}" if psnr is not None else "--"
                    row[f"ssim_{case}"] = f"{ssim:.4f}" if ssim is not None else "--"
                else:
                    row[f"psnr_{case}"] = "--"
                    row[f"ssim_{case}"] = "--"
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[saved]  {out_path}")


def _algo_family(name: str) -> str:
    """Backbone of an algorithm label with the variant suffixes removed (e.g. UNet-Diff-Spectral gives UNet)."""
    for suffix in ("-Diff-Spectral", "-Diff-2t", "-Diff", "-Spectral", "-2t"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _variant_order(name: str) -> int:
    """Position of a variant within its family: base 0, 2t 1, Spectral 2, Diff 3, Diff-2t 4, Diff-Spectral 5."""
    if name.endswith("-Diff-Spectral"): return 5
    if name.endswith("-Diff-2t"):       return 4
    if name.endswith("-Diff"):          return 3
    if name.endswith("-Spectral"):      return 2
    if name.endswith("-2t"):            return 1
    return 0


def main():
    """Draw all figures and the metrics table in OUT_DIR."""
    # Output folder
    os.makedirs(OUT_DIR, exist_ok=True)

    # PSNR and SSIM of every algorithm and case; algorithms sorted by mean PSNR
    _YEAR_MAP = {"1year": 1, "2year": 2, "4year": 4,
                 "5year": 5, "10year": 10, "20year": 20}
    x_years = [_YEAR_MAP.get(c, i + 1) for i, c in enumerate(CASES)]

    _psnr_data: dict[str, list] = {}
    _ssim_data: dict[str, list] = {}
    _valid_x:   dict[str, list] = {}

    for _algo in ALGO_LABELS:
        _xs, _ps, _ss = [], [], []
        for _xi, _case in zip(x_years, CASES):
            _p, _s = _load_metrics(_algo, _case)
            if _p is not None:
                _xs.append(_xi); _ps.append(_p); _ss.append(_s)
        _psnr_data[_algo] = _ps
        _ssim_data[_algo] = _ss
        _valid_x[_algo]   = _xs

    _mean_psnr   = {a: (np.mean(v) if v else -np.inf) for a, v in _psnr_data.items()}
    _algo_sorted = sorted(ALGO_LABELS, key=lambda a: _mean_psnr[a], reverse=True)
    _ncol        = max(2, len(_algo_sorted) // 3)


    # Figures 1-4: spectrum of each algorithm, one curve per training case
    for algo in ALGO_LABELS:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        shades      = case_shades(ALGO_COLORS[algo], len(CASES))
        ref_plotted = False

        for i, case in enumerate(CASES):
            df  = load_csv(path_v1(case, algo))
            if df is None:
                continue
            k   = k_to_meters(df["k_mid_cy_per_px"].values)

            if not ref_plotted:
                ax.loglog(k, hke(df["gt_psd_u"].values, df["gt_psd_v"].values),
                          label="HR", **GT_STYLE)
                ax.loglog(k, hke(df["lr_psd_u"].values, df["lr_psd_v"].values),
                          label="LR", **LR_STYLE)
                ref_plotted = True

            ax.loglog(k, hke(df["pred_psd_u"].values, df["pred_psd_v"].values),
                      color=shades[i], linewidth=1.6,
                      label=case, linestyle=get_linestyle(algo))

        _style_ax(ax)
        order = _compute_legend_order(ax)
        _apply_legend_order(ax, order, **LEGEND_KW)
        fig.tight_layout()

        out = os.path.join(OUT_DIR, f"fig_{algo.replace('-', '_')}_years.pdf")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved]  {out}")


    # Figure 5: spectra of all algorithms for one case
    fig5, ax5 = plt.subplots(1, 1, figsize=(8, 6))

    ref_df = load_csv(REF_CSV)
    if ref_df is not None:
        k_ref  = k_to_meters(ref_df["k_mid_cy_per_px"].values)
        ax5.loglog(k_ref, hke(ref_df["gt_psd_u"].values, ref_df["gt_psd_v"].values),
                   label="HR", **GT_STYLE)
        ax5.loglog(k_ref, hke(ref_df["lr_psd_u"].values, ref_df["lr_psd_v"].values),
                   label="LR", **LR_STYLE)

    for algo in ALGO_LABELS:
        df = load_csv(path_v2(algo))
        if df is None:
            continue
        k  = k_to_meters(df["k_mid_cy_per_px"].values)
        ax5.loglog(k, hke(df["pred_psd_u"].values, df["pred_psd_v"].values),
                   color=ALGO_COLORS[algo], linewidth=1.6,
                   label=algo, linestyle=get_linestyle(algo))

    if PLOT_BASELINE_CURVES:
        for bl_key in BASELINES:
            df_bl = load_csv(path_baseline_psd(BASELINE_CASE, bl_key))
            if df_bl is None:
                continue
            k_bl = k_to_meters(df_bl["k_mid_cy_per_px"].values)
            ax5.loglog(k_bl, hke(df_bl["pred_psd_u"].values, df_bl["pred_psd_v"].values),
                       label=bl_key, **BASELINE_LINE_STYLES[bl_key])
    _style_ax(ax5)
    order5 = _compute_legend_order(ax5)
    _apply_legend_order(ax5, order5, **LEGEND_KW)
    fig5.tight_layout()

    out5 = os.path.join(OUT_DIR, "fig_2year_all_algorithms.pdf")
    fig5.savefig(out5, bbox_inches="tight")
    plt.close(fig5)
    print(f"[saved]  {out5}")


    # Figure 6: PSNR and SSIM against training duration
    fig6, (ax6_psnr, ax6_ssim) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    for algo in _algo_sorted:
        xs = _valid_x[algo]
        if not xs:
            continue
        kw = dict(
            color     = ALGO_COLORS[algo],
            linewidth = 1.8,
            linestyle = get_linestyle(algo),
            marker    = get_marker(algo),
            markersize= 7,
            label     = algo,
        )
        ax6_psnr.plot(xs, _psnr_data[algo], **kw)
        ax6_ssim.plot(xs, _ssim_data[algo], **kw)

    _style_metric_ax(ax6_psnr, "PSNR  [dB]", "PSNR", show_xlabel=False)
    _style_metric_ax(ax6_ssim, "SSIM  [–]",  "SSIM", show_xlabel=True)
    ax6_ssim.set_xticks(x_years)
    ax6_ssim.set_xticklabels(CASES)


    _leg6_kw = dict(
        fontsize       = 15,
        framealpha     = 0.75,
        loc            = "upper center",
        bbox_to_anchor = (0.5, -0.25),
        ncol           = _ncol,
    )
    _h6, _l6 = ax6_psnr.get_legend_handles_labels()
    ax6_ssim.legend(_h6, _l6, **_leg6_kw)

    fig6.subplots_adjust(bottom=0.35)


    ax6_psnr.legend(_h6, _l6, **_leg6_kw)
    ax6_ssim.legend(_h6, _l6, **_leg6_kw)

    fig6.tight_layout()
    out6 = os.path.join(OUT_DIR, "fig_metrics_by_case.pdf")
    fig6.savefig(out6, bbox_inches="tight")
    plt.close(fig6)
    print(f"[saved]  {out6}")


    # Figure 7: inference time per sample against PSNR and SSIM
    fig7, (ax7_psnr, ax7_ssim) = plt.subplots(2, 1, figsize=(8, 10))

    for algo in _algo_sorted:
        psnr, ssim = _load_metrics(algo, TIME_CASE)
        t_per_s    = _load_last_sampling_time(algo, TIME_CASE)
        if None in (psnr, ssim, t_per_s):
            print(f"  [SKIP scatter]  {algo}")
            continue
        clr = ALGO_COLORS[algo]
        mkr = get_marker(algo)
        scatter_kw = dict(color=clr, s=100, zorder=5, marker=mkr, label=algo)
        text_kw    = dict(textcoords="offset points", xytext=(7, 3),
                          fontsize=20, color=clr, fontfamily="serif")
        ax7_psnr.scatter(t_per_s, psnr, **scatter_kw)
        ax7_psnr.annotate(algo, (t_per_s, psnr), **text_kw)
        ax7_ssim.scatter(t_per_s, ssim, **scatter_kw)
        ax7_ssim.annotate(algo, (t_per_s, ssim), **text_kw)

    for ax, ylabel, title in [
        (ax7_psnr, "PSNR  [dB]", "PSNR  vs  inference time"),
        (ax7_ssim, "SSIM  [–]",  "SSIM  vs  inference time"),
    ]:
        ax.set_xlabel(
            rf"Sampling time per test sample  $\left[\mathrm{{s\,/\,sample}}\right]$"
            f"\n(total / {N_TEST_SAMPLES})"
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.set_xscale("log")
        _fmt = ScalarFormatter()
        _fmt.set_scientific(False)
        _fmt.set_useOffset(False)
        ax.xaxis.set_major_formatter(_fmt)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="both", direction="in", pad=6)

    fig7.tight_layout()
    out7 = os.path.join(OUT_DIR, "fig_time_vs_metrics.pdf")
    fig7.savefig(out7, bbox_inches="tight")
    plt.close(fig7)
    print(f"[saved]  {out7}")


    # Figure 8: PSNR and SSIM time series of the temporal case
    fig8, (ax8_psnr, ax8_ssim) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"hspace": 0.06},
    )

    for algo in _algo_sorted:
        p = path_temporal(TEMPORAL_CASE, algo)
        if not os.path.exists(p):
            print(f"  [MISSING]  {p}")
            continue
        try:
            df  = pd.read_csv(p, parse_dates=["datetime"])
            df  = df.sort_values("datetime").reset_index(drop=True)
            clr = ALGO_COLORS[algo]
            ls  = get_linestyle(algo)
            ax8_psnr.plot(df["datetime"], df["psnr_db"],
                          color=clr, linewidth=0.5, linestyle=ls, alpha=0.20)
            ax8_ssim.plot(df["datetime"], df["ssim"],
                          color=clr, linewidth=0.5, linestyle=ls, alpha=0.20)
            ax8_psnr.plot(df["datetime"], df["psnr_rolling"],
                          color=clr, linewidth=1.8, linestyle=ls,
                          label=algo, alpha=0.95)
            ax8_ssim.plot(df["datetime"], df["ssim_rolling"],
                          color=clr, linewidth=1.8, linestyle=ls,
                          label=algo, alpha=0.95)
        except Exception as exc:
            print(f"  [WARNING]  {p}: {exc}")

    ax8_ssim.xaxis.set_major_locator(mdates.MonthLocator())
    ax8_ssim.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax8_ssim.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
    fig8.autofmt_xdate(rotation=35, ha="right")
    ax8_psnr.tick_params(labelbottom=False)

    ax8_psnr.set_ylabel("PSNR  [dB]")
    ax8_ssim.set_ylabel("SSIM  [–]")
    ax8_ssim.set_xlabel("Time  (Year 2024 — 3-hourly samples)")

    for ax in (ax8_psnr, ax8_ssim):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="both", direction="in", pad=6)

    _ht8, _lt8 = ax8_psnr.get_legend_handles_labels()
    ax8_ssim.legend(_ht8, _lt8,
                    fontsize       = 20,
                    framealpha     = 0.75,
                    loc            = "upper center",
                    bbox_to_anchor = (0.5, -0.80),
                    ncol           = len(_algo_sorted))

    fig8.subplots_adjust(bottom=0.50)
    out8 = os.path.join(OUT_DIR, "fig_timeseries_psnr_ssim.pdf")
    fig8.savefig(out8, bbox_inches="tight")
    plt.close(fig8)
    print(f"[saved]  {out8}")


    # Figure 10: spectra of ground truth, LR and baselines
    fig10, ax10 = plt.subplots(1, 1, figsize=(8, 6))

    ref_df10 = load_csv(REF_CSV)
    if ref_df10 is not None:
        k_ref10 = k_to_meters(ref_df10["k_mid_cy_per_px"].values)
        ax10.loglog(k_ref10,
                    hke(ref_df10["gt_psd_u"].values, ref_df10["gt_psd_v"].values),
                    label="HR", **GT_STYLE)
        ax10.loglog(k_ref10,
                    hke(ref_df10["lr_psd_u"].values, ref_df10["lr_psd_v"].values),
                    label="LR", **LR_STYLE)
    if PLOT_BASELINE_CURVES:
        for bl_key in BASELINES:
            df_bl10 = load_csv(path_baseline_psd(BASELINE_CASE, bl_key))
            if df_bl10 is None:
                continue
            k_bl10 = k_to_meters(df_bl10["k_mid_cy_per_px"].values)
            ax10.loglog(k_bl10,
                        hke(df_bl10["pred_psd_u"].values, df_bl10["pred_psd_v"].values),
                        label=bl_key, **BASELINE_LINE_STYLES[bl_key])
    _style_ax(ax10)
    order10 = _compute_legend_order(ax10)
    _apply_legend_order(ax10, order10, **LEGEND_KW)
    fig10.tight_layout()

    out10 = os.path.join(OUT_DIR, "fig_baselines_psd.pdf")
    fig10.savefig(out10, bbox_inches="tight")
    plt.close(fig10)
    print(f"[saved]  {out10}")


    BAR_CASE = "2year"


    # Figure 11: PSNR and SSIM bars grouped by backbone family and variant
    _bar_groups: dict[str, list[str]] = {}
    for _alg in ALGO_LABELS:
        _bar_groups.setdefault(_algo_family(_alg), []).append(_alg)
    for _fam in _bar_groups:
        _bar_groups[_fam].sort(key=_variant_order)


    _bf_psnr: dict[str, float] = {}
    _bf_ssim: dict[str, float] = {}
    for _alg in ALGO_LABELS:
        _p, _s = _load_metrics(_alg, BAR_CASE)
        if _p is not None:
            _bf_psnr[_alg] = _p
            _bf_ssim[_alg] = _s


    BAR_W       = 0.28
    BAR_GAP     = 0.04
    BAR_SPACING = BAR_W + BAR_GAP
    GROUP_GAP   = 0.50

    _bar_x:        dict[str, float] = {}
    _xtick_pos:    list[float]      = []
    _xtick_labels: list[str]        = []

    _x_cursor = 0.0
    for _fam, _algos in _bar_groups.items():
        n = len(_algos)
        for i, _alg in enumerate(_algos):
            _bar_x[_alg] = _x_cursor + i * BAR_SPACING
        _xtick_pos.append(_x_cursor + (n - 1) / 2 * BAR_SPACING)
        _xtick_labels.append(_fam)
        _x_cursor += n * BAR_SPACING + GROUP_GAP


    _all_psnr = list(_bf_psnr.values())
    _all_ssim = list(_bf_ssim.values())
    _psnr_ylim = (
        (min(_all_psnr) - 1.2, max(_all_psnr) + 0.8) if _all_psnr else (27.0, 35.0)
    )
    _ssim_ylim = (
        (min(_all_ssim) - 0.012, max(_all_ssim) + 0.008) if _all_ssim else (0.85, 0.95)
    )


    _fig_width = max(8, len(_bf_psnr) * 1.4)
    fig11, (ax11p, ax11s) = plt.subplots(
        2, 1, figsize=(_fig_width, 9), sharex=True,
        gridspec_kw={"hspace": 0.06},
    )

    for _alg, xc in _bar_x.items():
        if _alg not in _bf_psnr:
            continue
        clr    = ALGO_COLORS.get(_alg, "#888888")
        bar_kw = dict(width=BAR_W, color=clr, edgecolor="black", linewidth=0.8, zorder=3)
        ax11p.bar(xc, _bf_psnr[_alg], **bar_kw)
        ax11s.bar(xc, _bf_ssim[_alg], **bar_kw)


    for _ax, _ylabel, _ylim in [
        (ax11p, "PSNR  [dB]", _psnr_ylim),
        (ax11s, "SSIM  [–]",  _ssim_ylim),
    ]:
        _ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.55, zorder=0)
        _ax.set_axisbelow(True)
        _ax.set_ylabel(_ylabel)
        _ax.set_ylim(_ylim)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _ax.tick_params(axis="both", which="both", direction="in", pad=6)

    ax11s.set_xticks(_xtick_pos)
    ax11s.set_xticklabels(_xtick_labels, fontsize=20)
    ax11p.tick_params(labelbottom=False)


    _leg_patches = [
        _Patch(facecolor=ALGO_COLORS.get(_alg, "#888"), edgecolor="black", label=_alg)
        for _alg in _bar_x
        if _alg in _bf_psnr
    ]
    ax11s.legend(
        handles        = _leg_patches,
        loc            = "upper center",
        bbox_to_anchor = (0.5, -0.12),
        fontsize       = 16,
        framealpha     = 0.8,
        ncol           = len(_leg_patches),
    )

    fig11.subplots_adjust(bottom=0.20)
    out11 = os.path.join(OUT_DIR, "fig_bar_variants.pdf")
    fig11.savefig(out11, bbox_inches="tight")
    plt.close(fig11)
    print(f"[saved]  {out11}")
    if BAR_FIGURE_FAMILY_STYLE:
        BAR_CASE = "2year"

        BAR_FAMILIES: dict[str, dict[str, str]] = {
            "UNet": {
                "Base":     "UNet_Eval_Clean",
                "2t":       "UNet_2t_Eval-clean",
                "Spectral": "UNet_Spectral_Eval",
            },
            "Swin2SR": {
                "Base":     "Swin2SR_Eval",
                "2t":       "Swin2SR_2t_100eps",
                "Spectral": "Swin2SR_Spectral_Eval",
            },
        }

        VARIANTS = ["Base", "2t", "Spectral"]

        VARIANT_HATCH: dict[str, str] = {
            "Base":     "",
            "2t":       "oo",
            "Spectral": "///",
        }

        FAMILY_COLORS: dict[str, str] = {
            "UNet":    ALGO_COLORS["UNet"],
            "Swin2SR": ALGO_COLORS["Swin2SR"],
        }

        BAR_W       = 0.28
        BAR_GAP     = 0.04
        BAR_SPACING = BAR_W + BAR_GAP
        N_VAR       = len(VARIANTS)
        _var_offsets = np.array([
            (i - (N_VAR - 1) / 2) * BAR_SPACING for i in range(N_VAR)
        ])
        _fam_centers = np.array([0.0, (N_VAR * BAR_SPACING) + 0.5])

        _bf_psnr: dict[str, dict[str, float]] = {}
        _bf_ssim: dict[str, dict[str, float]] = {}

        for _fname, _variants in BAR_FAMILIES.items():
            _bf_psnr[_fname] = {}
            _bf_ssim[_fname] = {}
            for _vname, _bfolder in _variants.items():
                _bpath = f"{_BASE}/{_bfolder}/{BAR_CASE}/test_metrics.csv"
                if not os.path.exists(_bpath):
                    print(f"  [MISSING]  {_bpath}  — skipping bar")
                    continue
                try:
                    _bdf   = pd.read_csv(_bpath)
                    _brows = _bdf[_bdf["case"].astype(str) == BAR_CASE]
                    if _brows.empty:
                        continue
                    _blast = _brows.iloc[-1]
                    _bf_psnr[_fname][_vname] = float(_blast["test_psnr"])
                    _bf_ssim[_fname][_vname] = float(_blast["test_ssim"])
                except Exception as _exc:
                    print(f"  [WARNING]  {_bpath}: {_exc}")

        _bf_all_psnr = [v for d in _bf_psnr.values() for v in d.values()]
        _bf_all_ssim = [v for d in _bf_ssim.values() for v in d.values()]

        _bf_psnr_ylim = (
            (min(_bf_all_psnr) - 1.2, max(_bf_all_psnr) + 0.8)
            if _bf_all_psnr else (27.0, 35.0)
        )
        _bf_ssim_ylim = (
            (min(_bf_all_ssim) - 0.012, max(_bf_all_ssim) + 0.008)
            if _bf_all_ssim else (0.85, 0.95)
        )

        fig11, (ax11p, ax11s) = plt.subplots(
            2, 1, figsize=(10, 9), sharex=True,
            gridspec_kw={"hspace": 0.06},
        )

        for fi, (fname, _) in enumerate(BAR_FAMILIES.items()):
            _fclr = FAMILY_COLORS[fname]
            for vi, vname in enumerate(VARIANTS):
                if vname not in _bf_psnr[fname]:
                    continue
                _hatch = VARIANT_HATCH[vname]
                xc = _fam_centers[fi] + _var_offsets[vi]
                kw = dict(
                    width     = BAR_W,
                    color     = _fclr,
                    hatch     = _hatch,
                    edgecolor = "#2c2c2c" if _hatch else "black",
                    linewidth = 0.4       if _hatch else 0.8,
                    zorder    = 3,
                )
                ax11p.bar(xc, _bf_psnr[fname][vname], **kw)
                ax11s.bar(xc, _bf_ssim[fname][vname], **kw)

        for _ax, _ylabel, _ylim in [
            (ax11p, "PSNR  [dB]", _bf_psnr_ylim),
            (ax11s, "SSIM  [–]",  _bf_ssim_ylim),
        ]:
            _ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.55, zorder=0)
            _ax.set_axisbelow(True)
            _ax.set_ylabel(_ylabel)
            _ax.set_ylim(_ylim)
            _ax.spines["top"].set_visible(False)
            _ax.spines["right"].set_visible(False)
            _ax.tick_params(axis="both", which="both", direction="in", pad=6)

        ax11s.set_xticks(_fam_centers)
        ax11s.set_xticklabels(list(BAR_FAMILIES.keys()), fontsize=20)
        ax11p.tick_params(labelbottom=False)

        _leg_family = [
            _Patch(facecolor=FAMILY_COLORS[f], edgecolor="black", label=f)
            for f in BAR_FAMILIES
        ]
        _leg_variant = [
            _Patch(facecolor="white", edgecolor="black",
                   hatch=VARIANT_HATCH[v], label=v)
            for v in VARIANTS
        ]

        _leg1 = ax11s.legend(
            handles        = _leg_family,
            title          = "Family",
            loc            = "upper center",
            bbox_to_anchor = (0.18, -0.12),
            fontsize       = 16,
            title_fontsize = 15,
            framealpha     = 0.8,
            ncol           = len(_leg_family),
        )
        ax11s.add_artist(_leg1)

        ax11s.legend(
            handles        = _leg_variant,
            title          = "Variant",
            loc            = "upper center",
            bbox_to_anchor = (0.72, -0.12),
            fontsize       = 16,
            title_fontsize = 15,
            framealpha     = 0.8,
            ncol           = len(_leg_variant),
        )

        fig11.subplots_adjust(bottom=0.20)
        out11 = os.path.join(OUT_DIR, "fig_bar_variants.pdf")
        fig11.savefig(out11, bbox_inches="tight")
        plt.close(fig11)
        print(f"[saved]  {out11}")


    # Figure 12: PSNR and SSIM bars of the baselines
    _bl_psnr: dict[str, float] = {}
    _bl_ssim: dict[str, float] = {}
    for _bl in BASELINES:
        _p, _s, _ = _load_baseline_metrics(_bl, BAR_CASE)
        if _p is not None:
            _bl_psnr[_bl] = _p
            _bl_ssim[_bl] = _s


    _bl_x: dict[str, float] = {
        _bl: i * (BAR_W + BAR_GAP)
        for i, _bl in enumerate(_bl_psnr)
    }


    _bl_all_psnr = list(_bl_psnr.values())
    _bl_all_ssim = list(_bl_ssim.values())
    _bl_psnr_ylim = (
        (min(_bl_all_psnr) - 1.2, max(_bl_all_psnr) + 0.8) if _bl_all_psnr else (27.0, 35.0)
    )
    _bl_ssim_ylim = (
        (min(_bl_all_ssim) - 0.012, max(_bl_all_ssim) + 0.008) if _bl_all_ssim else (0.85, 0.95)
    )


    fig12, (ax12p, ax12s) = plt.subplots(
        2, 1, figsize=(max(6, len(_bl_psnr) * 1.8), 9), sharex=True,
        gridspec_kw={"hspace": 0.06},
    )

    for _bl, xc in _bl_x.items():
        clr    = BASELINE_COLORS.get(_bl, "#888888")
        bar_kw = dict(width=BAR_W, color=clr, edgecolor="black", linewidth=0.8, zorder=3)
        ax12p.bar(xc, _bl_psnr[_bl], **bar_kw)
        ax12s.bar(xc, _bl_ssim[_bl], **bar_kw)


    for _ax, _ylabel, _ylim in [
        (ax12p, "PSNR  [dB]", _bl_psnr_ylim),
        (ax12s, "SSIM  [–]",  _bl_ssim_ylim),
    ]:
        _ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.55, zorder=0)
        _ax.set_axisbelow(True)
        _ax.set_ylabel(_ylabel)
        _ax.set_ylim(_ylim)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _ax.tick_params(axis="both", which="both", direction="in", pad=6)

    ax12s.set_xticks(list(_bl_x.values()))
    ax12s.set_xticklabels(list(_bl_x.keys()), fontsize=20)
    ax12p.tick_params(labelbottom=False)


    _bl_leg = [
        _Patch(facecolor=BASELINE_COLORS.get(_bl, "#888"), edgecolor="black", label=_bl)
        for _bl in _bl_x
    ]
    ax12s.legend(
        handles        = _bl_leg,
        loc            = "upper center",
        bbox_to_anchor = (0.5, -0.12),
        fontsize       = 16,
        framealpha     = 0.8,
        ncol           = len(_bl_leg),
    )

    fig12.subplots_adjust(bottom=0.20)
    out12 = os.path.join(OUT_DIR, "fig_bar_baselines.pdf")
    fig12.savefig(out12, bbox_inches="tight")
    plt.close(fig12)
    print(f"[saved]  {out12}")


    # Metrics table
    export_metrics_csv(os.path.join(OUT_DIR, "metrics_table.csv"))

    if PLOT_TIMESERIES_FIGURES:
        for metric_col, rolling_col, ylabel, figname in [
            ("psnr_db", "psnr_rolling", "PSNR  [dB]", "fig_timeseries_psnr.pdf"),
            ("ssim",    "ssim_rolling", "SSIM  [–]",  "fig_timeseries_ssim.pdf"),
        ]:
            fig, ax = plt.subplots(figsize=(14, 5))

            for algo in _algo_sorted:
                p = path_temporal(TEMPORAL_CASE, algo)
                if not os.path.exists(p):
                    print(f"  [MISSING]  {p}")
                    continue
                try:
                    df  = pd.read_csv(p, parse_dates=["datetime"])
                    df  = df.sort_values("datetime").reset_index(drop=True)
                    clr = ALGO_COLORS[algo]
                    ls  = get_linestyle(algo)

                    ax.plot(df["datetime"], df[metric_col],
                            color=clr, linewidth=0.5, linestyle=ls, alpha=0.20)
                    ax.plot(df["datetime"], df[rolling_col],
                            color=clr, linewidth=1.8, linestyle=ls,
                            label=algo, alpha=0.95)
                except Exception as exc:
                    print(f"  [WARNING]  {p}: {exc}")

            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
            fig.autofmt_xdate(rotation=35, ha="right")

            ax.set_ylabel(ylabel)
            ax.set_xlabel("Time  (Year 2024 — 3-hourly samples)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", which="both", direction="in", pad=6)

            _ht, _lt = ax.get_legend_handles_labels()
            ax.legend(_ht, _lt,
                      fontsize   = 20,
                      framealpha = 0.75,
                      loc        = "upper center",
                      bbox_to_anchor = (0.5, -0.55),
                      ncol       = _ncol)

            fig.subplots_adjust(bottom=0.52)
            fig.tight_layout()
            out = os.path.join(OUT_DIR, figname)
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"[saved]  {out}")


if __name__ == "__main__":
    main()
