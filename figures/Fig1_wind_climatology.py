from figure_paths import DATA, RESULTS, WORK, FIGURES, setting

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from cartopy.mpl.ticker import (
    LongitudeFormatter,
    LatitudeFormatter,
)


DATA_DIR = DATA

CLIM_NC = os.path.join(
    RESULTS if (RESULTS / "Yangtze_low_level_wind_speed_climatology_2001_2023.nc").exists() else DATA,
    "Yangtze_low_level_wind_speed_climatology_2001_2023.nc"
)

SHP_PATH = setting("YANGTZE_BOUNDARY_FILE", DATA / "country1.shp")
OUT_DIR = FIGURES / "Figure1"
os.makedirs(OUT_DIR, exist_ok=True)

FIG1A_PNG = os.path.join(
    OUT_DIR,
    "Fig1a_mean_low_level_wind_speed.png"
)
FIG1A_PDF = os.path.join(
    OUT_DIR,
    "Fig1a_mean_low_level_wind_speed.pdf"
)
FIG1A_TIF = os.path.join(
    OUT_DIR,
    "Fig1a_mean_low_level_wind_speed.tif"
)

FIG1B_PNG = os.path.join(
    OUT_DIR,
    "Fig1b_std_low_level_wind_speed.png"
)
FIG1B_PDF = os.path.join(
    OUT_DIR,
    "Fig1b_std_low_level_wind_speed.pdf"
)
FIG1B_TIF = os.path.join(
    OUT_DIR,
    "Fig1b_std_low_level_wind_speed.tif"
)


LON_MIN, LON_MAX = 108.0, 126.0
LAT_MIN, LAT_MAX = 24.0, 36.0


if not os.path.exists(CLIM_NC):
    raise FileNotFoundError(
        f"找不到图1所需的空间统计文件：\n{CLIM_NC}\n\n"
        "请先运行“残差预测-长江流域风速范围计算.ipynb”。"
    )

with xr.open_dataset(CLIM_NC) as ds:
    required_variables = [
        "vertically_averaged_wind_speed_mean",
        "vertically_averaged_wind_speed_std",
    ]

    for variable_name in required_variables:
        if variable_name not in ds:
            raise KeyError(
                f"文件中缺少变量：{variable_name}\n"
                f"当前变量：{list(ds.data_vars)}"
            )

    mean_da = ds[
        "vertically_averaged_wind_speed_mean"
    ].load()

    std_da = ds[
        "vertically_averaged_wind_speed_std"
    ].load()

lat = np.asarray(mean_da["lat"].values)
lon = np.asarray(mean_da["lon"].values)

mean_map = np.asarray(
    mean_da.values,
    dtype=np.float32,
)

std_map = np.asarray(
    std_da.values,
    dtype=np.float32,
)

if mean_map.shape != std_map.shape:
    raise ValueError(
        f"均值图与标准差图形状不同："
        f"{mean_map.shape} != {std_map.shape}"
    )

print("空间场读取完成：")
print("lat.shape =", lat.shape)
print("lon.shape =", lon.shape)
print("mean_map.shape =", mean_map.shape)
print("std_map.shape =", std_map.shape)

print(
    "Fig. 1a 数值范围：",
    float(np.nanmin(mean_map)),
    "至",
    float(np.nanmax(mean_map)),
    "m/s",
)

print(
    "Fig. 1b 数值范围：",
    float(np.nanmin(std_map)),
    "至",
    float(np.nanmax(std_map)),
    "m/s",
)


boundary_gdf = None

if os.path.exists(SHP_PATH):
    try:
        import geopandas as gpd

        boundary_gdf = gpd.read_file(SHP_PATH)

        if boundary_gdf.crs is None:
            print(
                "Warning: 边界文件没有 CRS，"
                "暂按 EPSG:4326 处理。"
            )
            boundary_gdf = boundary_gdf.set_crs(
                epsg=4326
            )
        else:
            boundary_gdf = boundary_gdf.to_crs(
                epsg=4326
            )

        try:
            boundary_gdf = boundary_gdf.cx[
                LON_MIN - 3.0:LON_MAX + 3.0,
                LAT_MIN - 3.0:LAT_MAX + 3.0,
            ]
        except Exception:
            pass

        print("已读取本地边界文件：", SHP_PATH)

    except Exception as error:
        print("本地边界文件读取失败，将使用 Cartopy 边界：")
        print(error)
        boundary_gdf = None

else:
    print(
        "未找到本地边界文件，将使用 Cartopy 边界：",
        SHP_PATH,
    )

plt.rcParams["font.family"] = "Arial"

plt.rcParams["font.size"] = 22

plt.rcParams["axes.labelsize"] = 22
plt.rcParams["axes.titlesize"] = 24

plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20

plt.rcParams["legend.fontsize"] = 20

plt.rcParams["axes.linewidth"] = 1.8
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["ytick.major.width"] = 1.5
plt.rcParams["xtick.major.size"] = 6
plt.rcParams["ytick.major.size"] = 6

plt.rcParams["mathtext.fontset"] = "stix"

def add_map_background(ax):
    """添加海岸线、国界和经纬度网格。"""

    ax.set_extent(
        [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
        crs=ccrs.PlateCarree(),
    )

    ax.set_facecolor("white")

    if boundary_gdf is not None and len(boundary_gdf) > 0:
        boundary_gdf.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="black",
            linewidth=1.2,
            zorder=4,
        )

    ax.coastlines(
        resolution="50m",
        linewidth=1.2,
        color="black",
        zorder=5,
    )

    if boundary_gdf is None:
        ax.add_feature(
            cfeature.BORDERS.with_scale("50m"),
            edgecolor="black",
            linewidth=1.0,
            facecolor="none",
            zorder=4,
        )

    gridlines = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.9,
        color="gray",
        alpha=0.50,
        linestyle="--",
        zorder=6,
    )

    gridlines.top_labels = False
    gridlines.right_labels = False

    gridlines.xlocator = mticker.FixedLocator(
        [108, 111, 114, 117, 120, 123, 126]
    )

    gridlines.ylocator = mticker.FixedLocator(
        [24, 26, 28, 30, 32, 34, 36]
    )

    gridlines.xformatter = LongitudeFormatter(
        zero_direction_label=False
    )

    gridlines.yformatter = LatitudeFormatter()

    gridlines.xlabel_style = {
        "size": 20,
        "weight": "normal",
    }

    gridlines.ylabel_style = {
        "size": 20,
        "weight": "normal",
    }


def plot_single_panel(
    field,
    panel_label,
    cmap,
    colorbar_label,
    out_png,
    out_pdf,
    out_tif,
):
    """Draw and save one climatology panel."""

    valid_values = field[np.isfinite(field)]

    if valid_values.size == 0:
        raise ValueError(
            f"{panel_label} 对应数据全部为NaN。"
        )

    vmin = float(np.nanmin(valid_values))
    vmax = float(np.nanmax(valid_values))

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    fig = plt.figure(
        figsize=(8.8, 6.6),
        dpi=300,
    )

    ax = plt.axes(
        projection=ccrs.PlateCarree()
    )

    mesh = ax.pcolormesh(
        lon,
        lat,
        field,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
        zorder=1,
    )

    add_map_background(ax)

    ax.text(
        0.018,
        0.978,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=30,
        fontweight="bold",
        zorder=10,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
            "pad": 3.0,
        },
    )

    colorbar = fig.colorbar(
        mesh,
        ax=ax,
        orientation="vertical",
        pad=0.035,
        fraction=0.055,
        shrink=0.90,
    )

    colorbar.set_label(
        colorbar_label,
        fontsize=22,
        labelpad=12,
    )

    colorbar.ax.tick_params(
        labelsize=20,
        width=1.4,
        length=5,
    )

    colorbar.outline.set_linewidth(1.4)

    plt.tight_layout()

    fig.savefig(
        out_png,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.08,
    )

    fig.savefig(
        out_pdf,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.08,
    )

    fig.savefig(
        out_tif,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.08,
    )

    plt.show()
    plt.close(fig)

    print("\n已保存：")
    print(out_png)
    print(out_pdf)
    print(out_tif)


plot_single_panel(
    field=mean_map,
    panel_label="(a)",
    cmap="viridis",
    colorbar_label=r"Mean wind speed (m s$^{-1}$)",
    out_png=FIG1A_PNG,
    out_pdf=FIG1A_PDF,
    out_tif=FIG1A_TIF,
)


plot_single_panel(
    field=std_map,
    panel_label="(b)",
    cmap="magma",
    colorbar_label=r"Standard deviation (m s$^{-1}$)",
    out_png=FIG1B_PNG,
    out_pdf=FIG1B_PDF,
    out_tif=FIG1B_TIF,
)


print("\n全部完成。")
print("图件保存目录：")
print(OUT_DIR)


from contextlib import ExitStack
import pymupdf
from PIL import Image

with ExitStack() as stack:
    panels = [stack.enter_context(pymupdf.open(path)) for path in (FIG1A_PDF, FIG1B_PDF)]
    total_width = sum(panel[0].rect.width for panel in panels)
    height = panels[0][0].rect.height
    scale = (146.40277777777777 / 25.4 * 72) / total_width
    composite = stack.enter_context(pymupdf.open())
    page = composite.new_page(width=total_width * scale, height=height * scale)
    offset = 0.
    for panel in panels:
        width = panel[0].rect.width * scale
        page.show_pdf_page(pymupdf.Rect(offset, 0, offset + width, height * scale), panel, 0)
        offset += width
    composite.save(FIGURES / "Fig1.pdf", garbage=4, deflate=True)
    pixmap = page.get_pixmap(dpi=800, alpha=False)
    with Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples) as image:
        image.save(FIGURES / "Fig1.png", dpi=(800, 800), optimize=True)
        image.save(FIGURES / "Fig1.tiff", dpi=(800, 800), compression="tiff_adobe_deflate")
