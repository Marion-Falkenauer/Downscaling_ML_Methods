"""Plot of the pixel-wise autocorrelation of the CERRA wind components against the time lag.

The values in ACF are the Pearson autocorrelation r(tau) of the u and v components at 100 m of the
2-year CERRA HR training set (320 x 400, 3-hourly samples), for lags of 0 to 6 time steps
(0 to 18 h) and a sample of 8192 pixels. The figure is saved as acf_convergence-clean.pdf next to
this file. Add entries to ACF and SERIES (number of pixels, colour, line style, marker, marker size)
to compare other pixel-sample sizes.
Author: Marion Falkenauer (with the help of Claude)
"""
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(OUT_DIR, "acf_convergence-clean.pdf")

FREQ_HOURS = 3
lags = np.array([0, 1, 2, 3, 4, 5, 6])
lag_hours = lags * FREQ_HOURS

ACF = {
    "U-component": {
        8_192: [1, 0.8954, 0.7989, 0.7161, 0.6498, 0.5978, 0.5608],
    },
    "V-component": {
        8_192: [1, 0.8753, 0.7667, 0.6738, 0.5994, 0.5424, 0.5019],
    },
}

SERIES = [
    (8_192, "#0A0A0A", ":", "D", 10),
]


def plot_acf_convergence():
    """Save the autocorrelation figure, one panel per wind component.

    Reads the module constants ACF, SERIES, lags and lag_hours and writes OUT_PDF.
    """
    # Figure style
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

    fig, axes = plt.subplots(
        2, 1,
        figsize=(10, 9),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )

    # One panel per component, one line per pixel-sample size
    for ax, (comp, series_dict) in zip(axes, ACF.items()):
        for n_pix, color, ls, marker, ms in SERIES:
            r_vals = np.array(series_dict[n_pix])
            ax.plot(
                lag_hours, r_vals,
                linestyle=ls,
                marker=marker,
                markersize=ms,
                color=color,
                linewidth=2.0,
                markerfacecolor="black",
                markeredgewidth=1.8,
                label=f"{n_pix:,} px",
                zorder=3,
            )

        ax.set_xlim(lag_hours[0], lag_hours[-1] + 1.2)
        ax.set_ylim(0.45, 1.1)
        ax.set_ylabel(r"$\rho(\tau)$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_title(f"High Resolution {comp}", pad=8)

    # Lag labels on the bottom panel only
    axes[1].set_xticks(lag_hours)
    axes[1].set_xticklabels([f"τ={k}\n({k * FREQ_HOURS} h)" for k in lags])
    axes[1].set_xlabel("Lag  [hours]")

    plt.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {OUT_PDF}")


if __name__ == "__main__":
    plot_acf_convergence()
