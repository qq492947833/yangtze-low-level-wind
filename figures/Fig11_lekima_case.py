from figure_paths import DATA, RESULTS, WORK, FIGURES, setting
from pathlib import Path
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.ticker import MaxNLocator
LEAD_HOUR = 6
CASE_LEVELS = (850,)
CASE_NAME = 'Lekima (2019)'
LANDFALL_BJT = '2019-08-10 01:45:00+08:00'
OFFSETS_HOUR = (-24, -12, 0, 12, 24)
FONT_SCALE = 1.0
QUIVER_TARGET_COLUMNS = 20
QUIVER_KEY_MS = None

inputs = RESULTS / 'task3_typhoon'
with xr.open_dataset(inputs / 'Lekima_48h_6h_forecasts.nc') as dataset:
    case = dataset.load()
selected_times = pd.to_datetime(pd.read_csv(inputs / 'selected_times.csv')['target_utc'], utc=True).dt.tz_localize(None).to_numpy(dtype='datetime64[ns]')
boundary = None
FIGURE_DIR = FIGURES
FIG_DPI = 800
SHOW_FIGURES = False

FIGURE_NAMES = {
    int(level): f'Fig{11+i}'
    for i, level in enumerate(CASE_LEVELS)
}


def save_figure(fig, name):
    if FIG_DPI <= 600:
        raise ValueError('FIG_DPI必须大于600。')
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        FIGURE_DIR / f'{name}.png',
        dpi=FIG_DPI, facecolor='white'
    )
    fig.savefig(
        FIGURE_DIR / f'{name}.tiff',
        dpi=FIG_DPI, facecolor='white',
        pil_kwargs={'compression': 'tiff_lzw'}
    )

    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)
    print(f'已保存：{name}.png、{name}.tiff')


LABEL_SIZE = 18
TICK_SIZE = 16
TITLE_SIZE = 19
TIME_SIZE = 15
PANEL_SIZE = 16

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': LABEL_SIZE,
    'axes.labelsize': LABEL_SIZE,
    'xtick.labelsize': TICK_SIZE,
    'ytick.labelsize': TICK_SIZE,
})

WIND_CMAP = ListedColormap(
    plt.get_cmap('YlOrRd')(np.linspace(.03, .50, 256)),
    name='light_wind'
)

for level in CASE_LEVELS:
    data = case.sel(level=level, time=selected_times)
    lon = data.longitude.values
    lat = data.latitude.values

    finite = np.concatenate([
        data.testy_speed.values.ravel(),
        data.predicty_speed.values.ravel(),
    ])
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError(f'{level} hPa无有效风场。')

    vmax = max(float(finite.max()), 1e-6)
    norm = Normalize(0, vmax)

    fig_w = 7.8
    left, right = 0.85, 0.12
    col_gap, row_gap = 0.20, 0.42
    bottom, top = 2.15, 0.85

    panel_w = (fig_w - left - right - col_gap) / 2
    aspect = 1 / np.cos(np.deg2rad(float(np.mean(lat))))
    panel_h = panel_w * np.ptp(lat) / np.ptp(lon) * aspect
    fig_h = bottom + top + 5 * panel_h + 4 * row_gap

    fig = plt.figure(figsize=(fig_w, fig_h))

    key = (
        float(QUIVER_KEY_MS)
        if QUIVER_KEY_MS is not None
        else max(5., 5. * np.ceil(vmax / 5.))
    )
    if not np.isfinite(key) or key <= 0:
        raise ValueError('QUIVER_KEY_MS必须为正数。')

    scale = max(key / 0.48, vmax / 0.55)

    sx = max(
        1,
        int(np.ceil(len(lon) / min(QUIVER_TARGET_COLUMNS, 16)))
    )
    sy = max(
        1,
        int(round(
            sx * np.ptp(lon) / (len(lon) - 1)
            / (np.ptp(lat) / (len(lat) - 1))
        ))
    )
    xx, yy = np.meshgrid(lon[::sx], lat[::sy])

    for row, time in enumerate(selected_times):
        y0 = bottom + (4 - row) * (panel_h + row_gap)
        stamp = pd.Timestamp(time).tz_localize('UTC')
        offset_hour = (-24, -12, 0, 12, 24)[row]
        offset_label = (
            '0' if offset_hour == 0
            else f'{offset_hour:+d}'.replace('-', '−')
        )

        fig.text(
            (left + (2 * panel_w + col_gap) / 2) / fig_w,
            (y0 + panel_h + 0.08) / fig_h,
            f'{stamp:%d %b %H:%M} UTC  ({offset_label} h)',
            ha='center', va='bottom', fontsize=TIME_SIZE
        )

        for col, kind in enumerate(('testy', 'predicty')):
            x0 = left + col * (panel_w + col_gap)
            ax = fig.add_axes([
                x0 / fig_w,
                y0 / fig_h,
                panel_w / fig_w,
                panel_h / fig_h,
            ])

            d = data.sel(time=time)
            if not np.isfinite(d[f'{kind}_speed'].values).any():
                raise ValueError(
                    f'{level} hPa {time} {kind}: 无有效风场。'
                )

            mesh = ax.pcolormesh(
                lon, lat, d[f'{kind}_speed'],
                cmap=WIND_CMAP, norm=norm,
                shading='auto', rasterized=True
            )
            quiver = ax.quiver(
                xx, yy,
                d[f'{kind}_u'].values[::sy, ::sx],
                d[f'{kind}_v'].values[::sy, ::sx],
                color='black',
                angles='uv',
                scale_units='inches',
                scale=scale,
                width=0.0035,
                headwidth=3.8,
                headlength=5.0,
                minlength=0.
            )

            if boundary is not None:
                boundary.boundary.plot(
                    ax=ax, color='black',
                    linewidth=0.65, zorder=4
                )

            ax.set_xlim(lon.min(), lon.max())
            ax.set_ylim(lat.min(), lat.max())
            ax.set_aspect(aspect)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.tick_params(
                direction='out', length=3, pad=4,
                labelsize=TICK_SIZE,
                labelbottom=(row == 4),
                labelleft=(col == 0)
            )
            ax.grid(False)

            ax.text(
                0.025, 0.965,
                f'({chr(97 + row * 2 + col)})',
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=PANEL_SIZE,
                fontweight='bold', zorder=6,
                bbox=dict(
                    facecolor='white', edgecolor='none',
                    alpha=0.8, pad=1.5
                )
            )

            if row == 4:
                ax.set_xlabel(
                    'Longitude (°E)',
                    fontsize=LABEL_SIZE, labelpad=7
                )

            if row == 0:
                fig.text(
                    (x0 + panel_w / 2) / fig_w,
                    (y0 + panel_h + 0.48) / fig_h,
                    'Reference' if col == 0 else '6-h forecast',
                    ha='center', va='bottom',
                    fontsize=TITLE_SIZE, fontweight='bold'
                )
                if col == 1:
                    key_ax = ax
                    key_quiver = quiver

    maps_middle = bottom + (5 * panel_h + 4 * row_gap) / 2
    fig.text(
        0.14 / fig_w,
        maps_middle / fig_h,
        'Latitude (°N)',
        rotation=90,
        ha='center', va='center',
        fontsize=LABEL_SIZE
    )

    key_ax.quiverkey(
        key_quiver,
        0.43,
        1.22 / fig_h,
        key,
        f'{key:g} m s$^{{-1}}$',
        coordinates='figure',
        labelpos='E',
        labelsep=0.10,
        fontproperties={'size': LABEL_SIZE}
    )

    cax = fig.add_axes([
        left / fig_w,
        0.76 / fig_h,
        (2 * panel_w + col_gap) / fig_w,
        0.18 / fig_h,
    ])
    cb = fig.colorbar(mesh, cax=cax, orientation='horizontal')
    cb.locator = MaxNLocator(5)
    cb.update_ticks()
    cb.ax.tick_params(
        labelsize=TICK_SIZE, length=3, pad=4
    )
    cb.set_label(
        'Horizontal wind speed (m s$^{-1}$)',
        fontsize=LABEL_SIZE, labelpad=6
    )

    save_figure(fig, FIGURE_NAMES[int(level)])
