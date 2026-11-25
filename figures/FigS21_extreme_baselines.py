from pathlib import Path
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from PIL import Image
from figure_paths import DATA, RESULTS, WORK, FIGURES, setting
FIG_DPI = 800
SHOW_FIGURE = True

WORK_DIR = RESULTS / 'FigS21_extreme_baselines'
def export_s21(fig):
    stem = 'FigS21'
    fig.savefig(FIGURES / f'{stem}.png', dpi=FIG_DPI, facecolor='white')
    with Image.open(FIGURES / f'{stem}.png') as image:
        image.convert('RGB').save(FIGURES / f'{stem}.tiff', compression='tiff_lzw', dpi=(FIG_DPI, FIG_DPI))
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

levels = [850, 925, 1000]
percentiles = [90, 95, 99]
leads = [1, 2, 3, 4, 5, 6]

pairs = ["DA - Persistence", "DA - Ridge"]
rows = [
    (pairs[0], "DA − Pers.", "horizontal", "Horizontal wind"),
    (pairs[0], "DA − Pers.", "vertical",   r"$|\omega|$"),
    (pairs[1], "DA − Ridge", "horizontal", "Horizontal wind"),
    (pairs[1], "DA − Ridge", "vertical",   r"$|\omega|$"),
]

with xr.open_dataset(WORK_DIR / "FigS21_scores.nc") as ds:
    gain = ds["ETS_gain"].sel(
        comparison=pairs,
        quantity=["horizontal", "vertical"],
        level=levels,
        percentile=percentiles,
        lead_hour=leads,
    ).load()

finite = gain.values[np.isfinite(gain.values)]
if finite.size == 0:
    raise ValueError("选出的两组 ETS 差值全部缺失，请检查 FigS21_scores.nc。")

limit = float(np.max(np.abs(finite)))
if limit == 0:
    limit = 0.01  # 若所有差值恰好为零，所有格子均显示为白色。

norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
cmap = LinearSegmentedColormap.from_list(
    "ets_blue_white_red",
    ["#82AED3", "#FFFFFF", "#DF928C"],
    N=257,
)
cmap.set_bad("#EEEEEE")  # 缺失值用浅灰色，与零值区分。

with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 12}):
    fig, axes = plt.subplots(4, 3, figsize=(12.6, 11.0))
    fig.subplots_adjust(
        left=0.087, right=0.895, bottom=0.066, top=0.958,
        wspace=0.25, hspace=0.49,
    )

    try:
        for i, (pair, row_label, quantity, title) in enumerate(rows):
            for j, level in enumerate(levels):
                ax = axes[i, j]

                values = gain.sel(comparison=pair, quantity=quantity, level=level)
                mat = values.transpose("percentile", "lead_hour").values

                im = ax.imshow(
                    np.ma.masked_invalid(mat),
                    cmap=cmap,
                    norm=norm,
                    origin="upper",
                    aspect="auto",
                    interpolation="nearest",
                )

                panel = chr(ord("a") + 3 * i + j)
                ax.set_title(f"({panel})  {title}, {level} hPa", fontsize=13, pad=6)
                ax.set_xticks(range(len(leads)), labels=leads)
                ax.set_yticks(
                    range(len(percentiles)),
                    labels=[f"P{q}" for q in percentiles],
                )
                ax.tick_params(labelsize=11.5, length=3, pad=3)

                if j == 0:
                    ax.set_ylabel(row_label, fontsize=13, fontweight="bold", labelpad=8)
                if i == 3:
                    ax.set_xlabel("Lead time (h)", fontsize=12.5)

                for (r, c), v in np.ndenumerate(mat):
                    label = f"{v:.3f}" if np.isfinite(v) else "—"
                    if label == "-0.000":
                        label = "0.000"
                    ax.text(
                        c, r, label,
                        ha="center", va="center",
                        fontsize=10.8, color="#151515",
                    )

        cax = fig.add_axes([0.918, 0.175, 0.014, 0.67])
        cb = fig.colorbar(im, cax=cax, ticks=np.linspace(-limit, limit, 5))
        cb.ax.set_title("ΔETS", fontsize=13, pad=8)
        cb.ax.tick_params(labelsize=11)

        export_s21(fig)
        if SHOW_FIGURE:
            plt.show()
    finally:
        plt.close(fig)
