from pathlib import Path
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from PIL import Image
from figure_paths import DATA, RESULTS, WORK, FIGURES, setting
COMPONENTS = ('u', 'v', 'w')
LEVELS = np.array([850., 925., 1000.])
FONT_SCALE = 1.0

inputs = RESULTS / 'best_model_spatial_figures'
with xr.open_dataset(inputs / 'plotted_spatial_metrics.nc') as dataset:
    m1 = dataset[['R', 'RMSPE']].load()
    m2 = dataset[['RMSE', 'MAE']].load()
    latitude, longitude = dataset.latitude.values, dataset.longitude.values
with xr.open_dataset(inputs / 'mean_horizontal_wind_6h.nc') as dataset:
    a = [dataset[name].sel(level=LEVELS).transpose('level', 'latitude', 'longitude').values
         for name in ['test_u', 'test_v', 'pred_u', 'pred_v']]
boundary = None
FIGURE_DIR = FIGURES
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator, ScalarFormatter
FIG_DPI = 800
SHOW_FIGURES = False


def save_figure(fig, stem):
    if FIG_DPI <= 600:
        raise ValueError('FIG_DPI必须大于600。')
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for suffix in ('.png', '.tiff'):
        options = (
            {'pil_kwargs': {'compression': 'tiff_lzw'}}
            if suffix == '.tiff' else {}
        )
        fig.savefig(
            FIGURE_DIR / (stem + suffix),
            dpi=FIG_DPI,
            bbox_inches='tight',
            facecolor='white',
            **options
        )

    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)
    print(f'已保存：{stem}.png、{stem}.tiff')


_fs = float(globals().get('FONT_SCALE', 1.0))
_tick_size = 14 * _fs
_label_size = 14 * _fs
_title_size = 15 * _fs

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 14 * _fs,
    'axes.titlesize': _title_size,
    'axes.labelsize': _label_size,
    'xtick.labelsize': _tick_size,
    'ytick.labelsize': _tick_size,
})

_component_label = {'u': 'u', 'v': 'v', 'w': r'$\omega$'}
_lat = np.asarray(latitude, dtype=float)
_lon = np.asarray(longitude, dtype=float)
_aspect = 1 / np.cos(np.deg2rad(_lat.mean()))
_map_ratio = np.ptp(_lat) / np.ptp(_lon) * _aspect


def _task1_map_grid(nrows, vector=False):
    width, left, right, gap = 10.8, .76, .18, .26
    panel_w = (width - left - right - 2 * gap) / 3
    panel_h = panel_w * _map_ratio
    bottom, top = 1.30, (.70 if vector else .36)
    row_gap = .37 if vector else 1.06

    height = bottom + nrows * panel_h + (nrows - 1) * row_gap + top
    fig = plt.figure(figsize=(width, height), layout=None)
    axes = np.empty((nrows, 3), dtype=object)

    for r in range(nrows):
        for c in range(3):
            x = left + c * (panel_w + gap)
            y = bottom + (nrows - 1 - r) * (panel_h + row_gap)
            ax = fig.add_axes([
                x / width, y / height,
                panel_w / width, panel_h / height
            ])

            if globals().get('boundary') is not None:
                boundary.boundary.plot(
                    ax=ax, color='0.2', linewidth=.55
                )

            ax.set(
                xlim=(_lon.min(), _lon.max()),
                ylim=(_lat.min(), _lat.max())
            )
            ax.set_aspect(_aspect, adjustable='box')
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.tick_params(
                labelsize=_tick_size, pad=3,
                labelbottom=(r == nrows - 1),
                labelleft=(c == 0)
            )

            if r == nrows - 1:
                ax.set_xlabel('Longitude (°E)', labelpad=5)
            if c == 0:
                ax.set_ylabel('Latitude (°N)', labelpad=5)
            axes[r, c] = ax

    return fig, axes


def _task1_bar(fig, image, left_ax, right_ax, label, bottom_row=False):
    p, q = left_ax.get_position(), right_ax.get_position()
    height = fig.get_figheight()
    y = p.y0 - (.72 if bottom_row else .25) / height

    cax = fig.add_axes([
        p.x0, y, q.x1 - p.x0, .12 / height
    ])
    cb = fig.colorbar(image, cax=cax, orientation='horizontal')

    wide = (q.x1 - p.x0) * fig.get_figwidth() > 4
    locator = MaxNLocator(
        nbins=7 if wide else 5,
        min_n_ticks=6 if wide else 4
    )
    lo, hi = image.norm.vmin, image.norm.vmax
    ticks = locator.tick_values(lo, hi)
    tol = (hi - lo) * 1e-9
    ticks = ticks[(ticks >= lo - tol) & (ticks <= hi + tol)]
    cb.set_ticks(ticks)

    formatter = ScalarFormatter(useOffset=False, useMathText=True)
    formatter.set_powerlimits((-3, 4))
    cb.formatter = formatter
    cb.update_ticks()
    cb.ax.tick_params(labelsize=12 * _fs, pad=2)
    cb.ax.xaxis.get_offset_text().set_fontsize(12 * _fs)

    labels = cb.ax.get_xticklabels()
    labels[0].set_horizontalalignment('left')
    labels[-1].set_horizontalalignment('right')
    cb.set_label(label, fontsize=12 * _fs, labelpad=3)
    return cb


def _task1_plot_maps(maps, names, stem):
    fig, axes = _task1_map_grid(2)

    for r, metric in enumerate(names):
        images = []

        for c, component in enumerate(COMPONENTS):
            ax = axes[r, c]
            z = maps[metric].sel(component=component).values

            peers = COMPONENTS if metric in ('R', 'RMSPE') else (
                ('u', 'v') if component != 'w' else ('w',)
            )
            values = np.concatenate([
                maps[metric].sel(component=p).values.ravel()
                for p in peers
            ])
            values = values[np.isfinite(values)]
            if not values.size:
                raise ValueError(f'{metric}/{component}: 无可绘制值。')

            lo, hi = (
                (max(-1., min(.9, np.floor(values.min() * 10) / 10)), 1.)
                if metric == 'R'
                else (0., float(values.max()))
            )
            hi = max(hi, lo + 1e-8)

            im = ax.pcolormesh(
                _lon, _lat, z,
                shading='auto', rasterized=True,
                cmap='RdYlBu_r' if metric == 'R' else 'viridis',
                vmin=lo, vmax=hi
            )
            ax.set_title(
                f'({chr(97 + 3*r + c)}) {metric} — '
                f'{_component_label[component]}',
                pad=7
            )
            images.append(im)

        if metric in ('R', 'RMSPE'):
            _task1_bar(
                fig, images[0], axes[r, 0], axes[r, 2],
                metric, r == 1
            )
        else:
            _task1_bar(
                fig, images[0], axes[r, 0], axes[r, 1],
                metric + ' (m s⁻¹)', r == 1
            )
            _task1_bar(
                fig, images[2], axes[r, 2], axes[r, 2],
                metric + ' (Pa s⁻¹)', r == 1
            )

    save_figure(fig, stem)


def _task1_plot_vectors():
    tu, tv, pu, pv = np.asarray(a)
    pairs = [(tu, tv), (pu, pv), (tu - pu, tv - pv)]
    magnitudes = [np.hypot(u, v) for u, v in pairs]

    light_cmap = ListedColormap(
        plt.get_cmap('YlOrRd')(np.linspace(.03, .50, 256)),
        name='light_wind'
    )

    main_max = max(
        float(np.nanmax(magnitudes[0])),
        float(np.nanmax(magnitudes[1])),
        1e-8
    )
    error_peak = float(np.nanmax(magnitudes[2]))
    error_max = max(error_peak, 1e-8)

    base_key = max(1., float(np.ceil(main_max / 3)))
    main_scale, main_key = base_key * 9 / 1.70, base_key * 3

    if error_peak > 0:
        error_scale = error_peak / (.38 * 1.45)
        target = error_peak / 2
        unit = 10. ** np.floor(np.log10(target))
        error_key = unit * next(
            value for value in (1., 2., 5., 10.)
            if value >= target / unit
        )
    else:
        error_scale, error_key = 1., 0.

    step = max(
        1,
        int(np.ceil(len(_lon) / globals().get('ARROWS_ACROSS', 16)))
    )
    xx, yy = np.meshgrid(_lon, _lat)

    fig, axes = _task1_map_grid(len(LEVELS), vector=True)
    images, key_quivers = [], [None, None]
    titles = ('Reference', 'Forecast', 'Ref. − forecast')

    for r, level in enumerate(LEVELS):
        for c, ((u, v), mag) in enumerate(zip(pairs, magnitudes)):
            ax = axes[r, c]
            im = ax.pcolormesh(
                _lon, _lat, mag[r],
                shading='auto', rasterized=True,
                vmin=0,
                vmax=error_max if c == 2 else main_max,
                cmap=light_cmap
            )

            uq = u[r, ::step, ::step]
            vq = v[r, ::step, ::step]
            zero = (uq == 0) & (vq == 0)

            q = ax.quiver(
                xx[::step, ::step], yy[::step, ::step],
                np.ma.masked_where(zero, uq),
                np.ma.masked_where(zero, vq),
                angles='uv', scale_units='inches',
                scale=error_scale if c == 2 else main_scale,
                width=.0032, color='k',
                minlength=0, zorder=3
            )

            group = int(c == 2)
            if key_quivers[group] is None:
                key_quivers[group] = q

            ax.set_title(
                f'({chr(97 + 3*r + c)}) {int(level)} hPa — {titles[c]}',
                fontsize=13 * _fs, pad=7
            )
            if r == len(LEVELS) - 1:
                images.append(im)

    for q, speed, left_ax, right_ax in (
        (key_quivers[0], main_key, axes[0, 0], axes[0, 1]),
        (key_quivers[1], error_key, axes[0, 2], axes[0, 2])
    ):
        p = left_ax.get_position()
        s = right_ax.get_position()
        key_ax = fig.add_axes([
            p.x0,
            1 - .44 / fig.get_figheight(),
            s.x1 - p.x0,
            .28 / fig.get_figheight()
        ])
        key_ax.set_axis_off()
        center = (p.x0 + s.x1) / 2

        if speed > 0:
            key_ax.quiverkey(
                q,
                center - .22 / fig.get_figwidth(),
                1 - .30 / fig.get_figheight(),
                speed, f'{speed:g} m s⁻¹',
                coordinates='figure',
                labelpos='E', labelsep=.10,
                fontproperties={'size': 14 * _fs}
            )
        else:
            key_ax.text(
                .5, .5, '0 m s⁻¹',
                transform=key_ax.transAxes,
                ha='center', va='center',
                fontsize=14 * _fs, color='k'
            )

    _task1_bar(
        fig, images[0], axes[-1, 0], axes[-1, 1],
        'Magnitude of mean horizontal wind (m s⁻¹)', True
    )
    _task1_bar(
        fig, images[2], axes[-1, 2], axes[-1, 2],
        'Mean difference magnitude (m s⁻¹)', True
    )
    save_figure(fig, 'Fig9')


_task1_plot_maps(m1, ('R', 'RMSPE'), 'Fig8')
_task1_plot_maps(m2, ('RMSE', 'MAE'), 'FigS14')
_task1_plot_vectors()
