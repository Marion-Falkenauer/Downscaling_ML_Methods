"""Composite figure of wind speed fields for all algorithms, training sizes and reference data.

Grid: one row per method (ALGORITHMS: UNet, Swin2SR, UNet-Diff, Swin2SR-Diff) and one column per
training size (YEARS), plus a reference column whose cells are REFERENCE_METHODS (ground truth,
low resolution, bicubic, SRCNN). Each cell shows the wind speed sqrt(u^2 + v^2) of the first sample of
the pred_arrays*.npz saved by the evaluation scripts, with two zoom insets. One colour scale,
from the 1st to the 99th percentile of all fields, is shared by all cells.

Configuration (module constants):
    BASE_PATH: folder of the results.
    ALGORITHMS: row label -> result folder; YEARS: (case folder, column label).
    REFERENCE_METHODS: (label, npz path, array key, lr_downsample) of the reference cells. With
        lr_downsample the stored upsampled LR field is strided by RF to its native 64 x 80 grid.
    ZOOM1_*, ZOOM2_*: region (fractions x0, x1, y0, y1), inset position, colour and connectors of
        the two zoom insets.
    SHOW_REFERENCE_LABELS: True writes the label inside each reference cell.
    OUT_FILE, CELL_W, CELL_H, DPI: output file and size.

Author: Marion Falkenauer (with the help of Claude)
"""

import glob
import os
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, ConnectionPatch


plt.rcParams.update({
    "font.size": 25,
    "axes.titlesize": 25,
    "axes.labelsize": 25,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "legend.fontsize": 25,
    "figure.titlesize": 25,
})


BASE_PATH: str = (
    "insert the folder where the results will be put"
)

PX_LAT_HR: int = 320
PX_LON_HR: int = 400
RF:         int = 5

ALGORITHMS: dict[str, str] = {
    "UNet":         "UNet_Eval-clean",
    "Swin2SR":      "Swin2SR",
    "UNet-Diff":    "UNet_Diff_Eval-All",
    "Swin2SR-Diff": "Swin_Diff-Eval_All",
}

YEARS: list[tuple[str, str]] = [
    ("1year",  "1 year"),
    ("2year",  "2 years"),
    ("4year",  "4 years"),
]

REFERENCE_NPZ: str = (
    BASE_PATH + "/UNet_Eval-clean/2year/pred_arrays_2year.npz" # Take the path generalised of each picture (the first one of the 
								# predictions by default) 
)

REFERENCE_METHODS: list[tuple[str, str | None, str | None, bool]] = [
    ("High Res.",
     REFERENCE_NPZ,
     "hr_batch_unscaled",
     False),

    ("Low Res.",
     REFERENCE_NPZ,
     "lr_batch_unscaled",
     True),

    ("Bicubic",
     BASE_PATH + "/BASELINES/bicubic/2year/pred_arrays_2year.npz",
     "hr_pred_unscaled",
     False),

    ("SRCNN",
     BASE_PATH + "/BASELINES/srcnn/2year/pred_arrays_2year.npz",
     "hr_pred_unscaled",
     False),
]

REFERENCE_COL_LABEL: str = "References"
SHOW_REFERENCE_LABELS = False

CMAP: str = "viridis"

OUT_FILE: str = "./figure_grid.pdf"

INSET_BBOX: list[float] = [0.01, 0.54, 0.44, 0.44]
ZOOM_COLOR: str   = "red"
ZOOM_LW:    float = 3.5

CELL_W: float = 5.0
CELL_H: float = 4.6
DPI:    int   = 200

ZOOM1_REGION:     tuple = (0.50, 0.75, 0.00, 0.25)
ZOOM1_INSET_BBOX: list  = [0.01, 0.54, 0.44, 0.44]
ZOOM1_COLOR:      str   = "crimson"
ZOOM1_CONNECTORS: tuple = ((2, 3), (0, 1))

ZOOM2_REGION:     tuple = (0.50, 0.75, 0.75, 1.00)
ZOOM2_INSET_BBOX: list  = [0.01, 0.02, 0.44, 0.44]
ZOOM2_COLOR:      str   = "darkorange"
ZOOM2_CONNECTORS: tuple = ((2, 3), (0, 1))


ALGO_LABELS: list[str] = list(ALGORITHMS.keys())
YEAR_TOKENS: list[str] = [y[0] for y in YEARS]
YEAR_LABELS: list[str] = [y[1] for y in YEARS]
N_ROWS      = len(ALGO_LABELS)
N_ALGO_COLS = len(YEAR_TOKENS)
N_COLS      = N_ALGO_COLS + 1

assert len(REFERENCE_METHODS) == N_ROWS, (
    f"REFERENCE_METHODS must have {N_ROWS} entries, got {len(REFERENCE_METHODS)}."
)


def find_npz(algo_key: str, year_token: str) -> str:
    """Find the saved prediction file of an algorithm and a case.

    Args:
        algo_key: key of ALGORITHMS.
        year_token: case folder name.
    Returns:
        Path of the first pred_arrays*.npz found (FileNotFoundError if none).
    """
    folder  = os.path.join(BASE_PATH, ALGORITHMS[algo_key], year_token)
    pattern = os.path.join(folder, "pred_arrays*.npz")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No pred_arrays*.npz in:\n  {folder}"
        )
    return matches[0]


def extract_wind_speed(
    npz_path:      str,
    key:           str | None = None,
    lr_downsample: bool       = False,
) -> np.ndarray:
    """Load a wind speed field from an npz file.

    Args:
        npz_path: .npz file.
        key: array to read; the first array of the file when None.
        lr_downsample: stride the field by RF when it is at HR size (stored upsampled LR fields).
    Returns:
        float32 array (H, W); u and v of arrays (N, H, W, 2) or (H, W, 2) are combined, only the first
        sample is used.
    """
    npz = np.load(npz_path)

    if key is not None:
        arr = npz[key].astype(np.float32)
        print(f"  [{os.path.basename(npz_path)}]  key='{key}'  shape={arr.shape}")
    else:
        first_key = list(npz.keys())[0]
        arr = npz[first_key].astype(np.float32)
        print(f"  [{os.path.basename(npz_path)}]  key='{first_key}'  shape={arr.shape}")

    # Collapse batch dimension if present
    if arr.ndim == 4:
        arr = arr[0]

    # Extract wind speed from (H, W, >=2) or pass through (H, W)
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        speed = np.sqrt(arr[..., 0] ** 2 + arr[..., 1] ** 2)
    elif arr.ndim == 2:
        speed = arr
    else:
        raise ValueError(f"Unexpected shape {arr.shape}.")

    # Stride the HR-sized field back to the native LR size (64, 80)
    if lr_downsample:
        H, W = speed.shape
        if H >= PX_LAT_HR or W >= PX_LON_HR:
            speed = speed[::RF, ::RF]
            print(f"  → LR size-check triggered: downsampled {H}×{W} "
                  f"→ {speed.shape[0]}×{speed.shape[1]} (factor {RF})")
        else:
            print(f"  → LR size-check: field already at native resolution "
                  f"{H}×{W}, no downsampling applied.")

    return speed.astype(np.float32)


def speed_to_rgb(
    speed:     np.ndarray,
    norm:      mcolors.Normalize,
    cmap_name: str = CMAP,
) -> np.ndarray:
    """Convert a speed field to an RGB image with a shared normalisation.

    Args:
        speed: array (H, W).
        norm: matplotlib Normalize shared by all cells.
        cmap_name: colormap name.
    Returns:
        uint8 array (H, W, 3).
    """
    cmap = matplotlib.colormaps[cmap_name]
    rgba = cmap(norm(speed))
    return (rgba[..., :3] * 255).astype(np.uint8)


def _placeholder(message: str, ax: plt.Axes) -> None:
    """Fill an axis with a grey image and a red message.

    Args:
        message: text to display.
        ax: matplotlib axis.
    """
    ax.imshow(np.full((200, 200, 3), 220, dtype=np.uint8), aspect="auto")
    ax.text(0.5, 0.5, message,
            ha="center", va="center", transform=ax.transAxes,
            color="#cc0000", wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def attach_zoom_inset(
    ax:               plt.Axes,
    img:              np.ndarray,
    region:           tuple,
    inset_bbox:       list,
    color:            str   = "red",
    lw:               float = ZOOM_LW,
    connector_pairs:  tuple = ((2, 3), (0, 1)),
) -> None:
    """Add a zoom inset of a region of an image with connector lines to its rectangle.

    Args:
        ax: axis showing the image.
        img: image (H, W, 3).
        region: (x0, x1, y0, y1) as fractions of the image width and height, y from the top.
        inset_bbox: [left, bottom, width, height] of the inset in axis fractions.
        color: colour of the rectangle, the inset frame and the connectors.
        lw: line width.
        connector_pairs: pairs (zoom corner, inset corner) to connect; corners are numbered
            0 lower-left, 1 lower-right, 2 upper-left, 3 upper-right.
    """
    H, W = img.shape[:2]
    x0f, x1f, y0f, y1f = region
    qx0, qx1 = int(x0f * W), int(x1f * W)
    qy0, qy1 = int(y0f * H), int(y1f * H)

    # Inset axes
    axins = ax.inset_axes(inset_bbox)
    axins.imshow(img, aspect="auto", interpolation="nearest")
    axins.set_xlim(qx0, qx1)
    axins.set_ylim(qy1, qy0)  # inverted: image origin = top-left
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)

    # Indicator rectangle on the main axes
    rect_patch = Rectangle(
        (qx0, qy0),
        qx1 - qx0,  # width in pixels
        qy1 - qy0,
        linewidth=lw,
        edgecolor=color,
        facecolor="none",
        zorder=5,
    )
    ax.add_patch(rect_patch)

    # Corner coordinates of the zoom rectangle (data) and of the inset box (axes fractions)
    zoom_corners = {
        0: (qx0, qy1),  # lower-left (high row index = bottom of image)
        1: (qx1, qy1),
        2: (qx0, qy0),  # upper-left (low row index = top of image)
        3: (qx1, qy0),
    }

    # The inset box corners use transAxes, which is not affected by the inverted y axis
    inset_corners = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (0.0, 1.0),
        3: (1.0, 1.0),
    }

    # Draw connector lines
    for zoom_idx, inset_idx in connector_pairs:
        con = ConnectionPatch(
            xyA=zoom_corners[zoom_idx],    coordsA="data",
            xyB=inset_corners[inset_idx],  coordsB=axins.transAxes,
            axesA=ax,
            color=color,
            linewidth=lw,
            zorder=4,
        )
        ax.figure.add_artist(con)


def main():
    """Load the wind speed fields of all cells, then draw and save the composite figure."""
    # Pass 1: load every wind speed field and compute the shared normalisation
    print("── Pass 1: loading all speed arrays ──")

    speed_cache: dict[tuple[int, int], np.ndarray | None] = {}

    # Algorithm cells (high-resolution predictions)
    for i, algo in enumerate(ALGO_LABELS):
        for j, year in enumerate(YEAR_TOKENS):
            try:
                path              = find_npz(algo, year)
                speed_cache[i, j] = extract_wind_speed(path)
            except Exception as exc:
                warnings.warn(f"[{algo}/{year}] {exc}", stacklevel=2)
                speed_cache[i, j] = None

    # Reference cells
    for i, (ref_label, ref_path, ref_key, lr_ds) in enumerate(REFERENCE_METHODS):
        j = N_ALGO_COLS
        if ref_path is None:
            speed_cache[i, j] = None
        else:
            try:
                speed_cache[i, j] = extract_wind_speed(
                    ref_path,
                    key           = ref_key,
                    lr_downsample = lr_ds,
                )
            except Exception as exc:
                warnings.warn(f"[{ref_label}] {exc}", stacklevel=2)
                speed_cache[i, j] = None

    # Shared colour scale from the 1st to the 99th percentile of all fields
    valid = [s.ravel() for s in speed_cache.values() if s is not None]
    if not valid:
        raise RuntimeError("No valid speed arrays found — check your paths.")

    all_vals    = np.concatenate(valid)
    vmin, vmax  = float(np.nanpercentile(all_vals, 1)), \
                  float(np.nanpercentile(all_vals, 99))
    global_norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    print(f"\n  Global speed range (1–99th pct):  "
          f"vmin={vmin:.3f}  vmax={vmax:.3f}  m/s")


    # Pass 2: render the figure with the shared normalisation
    print("\n── Pass 2: rendering figure ──")

    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(CELL_W * N_COLS, CELL_H * N_ROWS),
        gridspec_kw={"hspace": 0.06, "wspace": 0.04},
    )

    # Column headers
    for j, col_label in enumerate(YEAR_LABELS):
        axes[0, j].set_title(col_label, fontweight="bold", pad=10, color="#111111")

    axes[0, N_ALGO_COLS].set_title(REFERENCE_COL_LABEL, fontweight="bold",
                                    pad=10, color="#111111")

    # Fill every cell, with a placeholder when the data are missing
    for (i, j), speed in speed_cache.items():
        ax = axes[i, j]

        if speed is None:
            if j == N_ALGO_COLS:
                ref_label = REFERENCE_METHODS[i][0]
                _placeholder(f"{ref_label}\n(not yet available)", ax)
                ax.text(0.03, 0.97, ref_label,
                        transform=ax.transAxes, fontweight="bold",
                        va="top", ha="left", color="white",
                        bbox=dict(facecolor="black", alpha=0.45, pad=3,
                                  linewidth=0))
            else:
                algo = ALGO_LABELS[i]
                year = YEAR_TOKENS[j]
                _placeholder(f"File not found\n{algo}/{year}", ax)
            continue

        img = speed_to_rgb(speed, global_norm)
        ax.imshow(img, aspect="auto", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

        attach_zoom_inset(ax, img,
                            region=ZOOM1_REGION, inset_bbox=ZOOM1_INSET_BBOX,
                            color=ZOOM1_COLOR,   connector_pairs=ZOOM1_CONNECTORS)
        attach_zoom_inset(ax, img,
                            region=ZOOM2_REGION, inset_bbox=ZOOM2_INSET_BBOX,
                            color=ZOOM2_COLOR,   connector_pairs=ZOOM2_CONNECTORS)

        if SHOW_REFERENCE_LABELS:
            if j == N_ALGO_COLS:
                ref_label = REFERENCE_METHODS[i][0]
                ax.text(0.03, 0.97, ref_label,
                        transform=ax.transAxes, fontweight="bold",
                        va="top", ha="left", color="white",
                        bbox=dict(facecolor="black", alpha=0.45, pad=3, linewidth=0))


    # Row labels
    for i, algo in enumerate(ALGO_LABELS):
        axes[i, 0].set_ylabel(algo, fontweight="bold",
                              rotation=90, labelpad=14,
                              va="center", color="#111111")

    for i, (ref_label, _, _, _) in enumerate(REFERENCE_METHODS):
        axes[i, N_ALGO_COLS].yaxis.set_label_position("right")
        axes[i, N_ALGO_COLS].set_ylabel(
            ref_label,
            fontweight="bold",
            rotation=270,
            labelpad=14,
            va="center",
            color="#111111",
        )

    # Layout, separator line and group titles
    fig.subplots_adjust(left=0.08, right=0.91, top=0.92, bottom=0.01,
                        hspace=0.06, wspace=0.04)

    top    = axes[0, 0].get_position().y1
    bottom = axes[-1, 0].get_position().y0
    pos_col3 = axes[0, 2].get_position()
    pos_col4 = axes[0, 3].get_position()
    x_sep    = 0.5 * (pos_col3.x1 + pos_col4.x0)

    fig.add_artist(
        plt.Line2D(
            [x_sep, x_sep],
            [bottom, 0.96],
            transform=fig.transFigure,
            color="black",
            linewidth=3,
        )
    )

    left_group_x = (
        axes[0, 0].get_position().x0 +
        axes[0, 2].get_position().x1
    ) / 2

    ref_group_x = (
        axes[0, 3].get_position().x0 +
        axes[0, 3].get_position().x1
    ) / 2

    fig.text(left_group_x, 0.985, "Downscaling Models",
             ha="center", va="bottom", fontsize=25, fontweight="bold")

    fig.text(ref_group_x, 0.985, "Reference Data",
             ha="center", va="bottom", fontsize=25, fontweight="bold")

    # Shared colour bar
    cbar_ax = fig.add_axes([0.945, 0.04, 0.015, 0.88])
    sm      = plt.cm.ScalarMappable(cmap=matplotlib.colormaps[CMAP],
                                     norm=global_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Wind Speed  [m/s]", labelpad=10,fontsize=25, fontweight="bold")
    cbar.ax.tick_params(labelsize=25)

    # Save the figure
    os.makedirs(os.path.dirname(os.path.abspath(OUT_FILE)), exist_ok=True)
    fig.savefig(OUT_FILE, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"\n[saved]  {OUT_FILE}")


if __name__ == "__main__":
    main()
