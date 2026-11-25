from figure_paths import DATA, RESULTS, WORK, FIGURES, setting
from pathlib import Path
import json
import warnings
import importlib.util
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, FuncFormatter, ScalarFormatter
LEADS = (1, 2, 3, 4, 5, 6)
FONT_SCALE = 0.8
FIGSIZE = (12.6, 6.9)
USE_CARTOPY = True

SAVE_DPI=800
OUTPUT_DIR=FIGURES
METRICS_PATH=RESULTS/'da_transunet_residual_vs_baseline_spatial_metrics_1to6h.nc'

COLOR_QUANTILE = 0.98  # 仍偏淡可改为0.95；越小，主体区域颜色越浓

if not 0 < COLOR_QUANTILE <= 1:
    raise ValueError("COLOR_QUANTILE 必须在 (0, 1] 范围内。")

with xr.open_dataset(METRICS_PATH, decode_timedelta=False) as _saved:
    plot_ds = _saved[["delta_r_map", "delta_rmspe_map"]].load()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_ccrs, _coastlines = None, None
if USE_CARTOPY:
    try:
        import cartopy.crs as ccrs
        from cartopy.io import shapereader

        _coast_path = shapereader.natural_earth(
            resolution="50m",
            category="physical",
            name="coastline"
        )
        _coast_reader = shapereader.Reader(_coast_path)
        _coastlines = list(_coast_reader.geometries())
        _coast_reader.close()
        _ccrs = ccrs

    except Exception as exc:
        print(
            f"使用普通经纬度坐标绘图"
            f"（海岸线不可用：{type(exc).__name__}）。"
        )


def draw_six_panel_map(dataset, variable, colorbar_label, file_stem):
    data = dataset[variable].sel(
        lead_time=list(LEADS)
    ).transpose("lead_time", "latitude", "longitude")

    values = np.asarray(data.values)
    finite = np.isfinite(values)

    if not np.any(finite):
        raise ValueError(f"{variable}: 没有可绘制的有效格点。")

    absolute = np.abs(values[finite])
    limit = float(np.quantile(absolute, COLOR_QUANTILE))

    if limit <= 0:
        limit = float(absolute.max()) or 1e-6

    saturated = 100.0 * np.mean(absolute > limit)

    print(
        f"{variable}: 原始范围 "
        f"[{values[finite].min():.6g}, {values[finite].max():.6g}]；"
        f"色标范围 ±{limit:.6g}；"
        f"超出色标范围的格点占比 {saturated:.2f}%"
    )

    boundaries = np.linspace(-limit, limit, 25)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    lon = data.longitude.values
    lat = data.latitude.values

    if len(lon) < 2 or len(lat) < 2:
        raise ValueError("空间填色图需要至少2×2个格点。")

    lon_ticks = MaxNLocator(nbins=4, min_n_ticks=3).tick_values(
        float(lon.min()), float(lon.max())
    )
    lat_ticks = MaxNLocator(nbins=4, min_n_ticks=3).tick_values(
        float(lat.min()), float(lat.max())
    )

    lon_ticks = lon_ticks[
        (lon_ticks >= lon.min()) & (lon_ticks <= lon.max())
    ]
    lat_ticks = lat_ticks[
        (lat_ticks >= lat.min()) & (lat_ticks <= lat.max())
    ]

    style = {
        "font.family": "DejaVu Sans",
        "font.size": 18 * FONT_SCALE,
        "axes.titlesize": 20 * FONT_SCALE,
        "axes.labelsize": 18 * FONT_SCALE,
        "xtick.labelsize": 16 * FONT_SCALE,
        "ytick.labelsize": 16 * FONT_SCALE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
    }

    subplot_kw = (
        {"projection": _ccrs.PlateCarree()}
        if _ccrs is not None else {}
    )

    with plt.rc_context(style):
        fig, axes = plt.subplots(
            2, 3, figsize=FIGSIZE, subplot_kw=subplot_kw
        )
        fig.subplots_adjust(
            left=0.073,
            right=0.978,
            bottom=0.215,
            top=0.937,
            wspace=0.27,
            hspace=0.30
        )

        for i, (ax, h) in enumerate(zip(axes.flat, LEADS)):
            z = values[i]

            plot_kw = (
                {"transform": _ccrs.PlateCarree()}
                if _ccrs is not None else {}
            )

            mappable = ax.contourf(
                lon,
                lat,
                np.ma.masked_invalid(z),
                levels=boundaries,
                cmap="RdBu",
                norm=norm,
                extend="both",
                corner_mask=False,
                **plot_kw
            )

            if _ccrs is not None:
                ax.set_extent(
                    [lon.min(), lon.max(), lat.min(), lat.max()],
                    crs=_ccrs.PlateCarree()
                )
                ax.add_geometries(
                    _coastlines,
                    crs=_ccrs.PlateCarree(),
                    facecolor="none",
                    edgecolor="0.25",
                    linewidth=0.55
                )
                ax.set_xticks(lon_ticks, crs=_ccrs.PlateCarree())
                ax.set_yticks(lat_ticks, crs=_ccrs.PlateCarree())

            else:
                ax.set_xlim(lon.min(), lon.max())
                ax.set_ylim(lat.min(), lat.max())
                ax.set_aspect(
                    1 / np.cos(np.deg2rad(float(np.mean(lat))))
                )
                ax.set_xticks(lon_ticks)
                ax.set_yticks(lat_ticks)

            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, pos: f"{x:g}°E")
            )
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda y, pos: f"{y:g}°N")
            )
            ax.tick_params(length=3.5, pad=3)
            ax.set_title(
                f"({chr(97 + i)})  {h} h",
                loc="left",
                pad=7
            )

        cax = fig.add_axes([0.22, 0.083, 0.60, 0.027])
        cb = fig.colorbar(
            mappable,
            cax=cax,
            orientation="horizontal",
            extend="both",
            ticks=np.linspace(-limit, limit, 5)
        )

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        formatter.set_useOffset(False)

        cb.formatter = formatter
        cb.update_ticks()
        cb.set_label(colorbar_label, labelpad=6)
        cb.ax.tick_params(labelsize=16 * FONT_SCALE)
        cb.ax.xaxis.get_offset_text().set_fontsize(15 * FONT_SCALE)

        for extension in ("png", "pdf"):
            path = OUTPUT_DIR / f"{file_stem}.{extension}"
            fig.savefig(
                path,
                dpi=SAVE_DPI,
                bbox_inches="tight",
                facecolor="white"
            )
            print(f"图像已保存：{path}")

        plt.show()
        plt.close(fig)


draw_six_panel_map(
    plot_ds,
    "delta_r_map",
    r"$\Delta R = R_{\mathrm{residual}} - R_{\mathrm{baseline}}$",
    "Fig_residual_vs_baseline_delta_R_1to6h"
)

draw_six_panel_map(
    plot_ds,
    "delta_rmspe_map",
    r"$\Delta\mathrm{RMSPE} = "
    r"\mathrm{RMSPE}_{\mathrm{baseline}} - "
    r"\mathrm{RMSPE}_{\mathrm{residual}}$",
    "Fig_residual_vs_baseline_delta_RMSPE_1to6h"
)
