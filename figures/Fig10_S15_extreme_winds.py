from figure_paths import DATA, RESULTS, WORK, FIGURES, setting
from pathlib import Path
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.patches import Patch
LEADS = (1, 2, 3, 4, 5, 6)
LEVELS = np.array([850., 925., 1000.])
RADAR_DIVISORS = None
QUANTILES = (0.90, 0.95, 0.99)

inputs = RESULTS / 'extreme_ETS_radar'
summary = pd.read_csv(inputs / 'extreme_metrics_ETS_by_lead_level.csv')
radar = pd.read_csv(inputs / 'radar_unscaled_values.csv')
FIGURE_DIR = FIGURES
FIG_DPI = 800
SHOW_FIGURES = False
RADAR_FIGURE = 'Fig10'
ETS_FIGURE = 'FigS15'


def save_figure(fig, name):
    if FIG_DPI <= 600:
        raise ValueError('FIG_DPI必须大于600。')
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        FIGURE_DIR / f'{name}.png',
        dpi=FIG_DPI, bbox_inches='tight', facecolor='white'
    )
    fig.savefig(
        FIGURE_DIR / f'{name}.tiff',
        dpi=FIG_DPI, bbox_inches='tight', facecolor='white',
        pil_kwargs={'compression': 'tiff_lzw'}
    )

    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)
    print(f'已保存：{name}.png、{name}.tiff')


def nice_divisor(v):
    if not np.isfinite(v) or v < 0:
        raise ValueError('无法缩放非有限或负误差量。')
    if v == 0:
        return 1.

    base = 10. ** np.floor(np.log10(v))
    return next(
        x * base for x in (1., 2., 5., 10.)
        if x * base >= v * (1 - 1e-12)
    )


def plot_extreme_radar(radar_frame):
    data = radar_frame.copy()
    data['1-R'] = 1 - data['R']

    keys = ('1-R', 'RMSPE', 'RMSE', 'MAE')
    colours = ('#69a6d5', '#e7ac67', '#a98dc5')
    angles = np.arange(4) * np.pi / 2
    closed = np.r_[angles, angles[0]]

    fig, axes = plt.subplots(
        1, 2, figsize=(11.4, 4.9),
        subplot_kw={'projection': 'polar'}
    )
    fig.subplots_adjust(
        left=.115, right=.885, bottom=.18, top=.83, wspace=.70
    )

    positions = [
        ((.5, 1), (0, 6), 'center', 'bottom'),
        ((1, .5), (6, 0), 'left', 'center'),
        ((.5, 0), (0, -6), 'center', 'top'),
        ((0, .5), (-6, 0), 'right', 'center'),
    ]
    titles = (
        '(a) Horizontal wind speed',
        '(b) Vertical motion ω'
    )

    for ax, wind, title in zip(
        axes, ('horizontal', 'vertical'), titles
    ):
        part = data.loc[data.wind == wind]
        rows = [
            part.loc[np.isclose(part['quantile'], q)]
            for q in QUANTILES
        ]
        if any(len(row) != 1 for row in rows):
            raise ValueError(
                f'{wind}: radar中每个分位数应恰有一行。'
            )

        values = np.stack([
            row.iloc[0][list(keys)].to_numpy(dtype=float)
            for row in rows
        ])
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(
                f'{wind}: 雷达图指标必须为有限非负数。'
            )

        div = {
            key: (
                float(RADAR_DIVISORS[wind][key])
                if RADAR_DIVISORS is not None
                else nice_divisor(float(part[key].max()))
            )
            for key in keys
        }
        if any(not np.isfinite(v) or v <= 0 for v in div.values()):
            raise ValueError('缩放除数必须为正有限值。')

        values = values / np.array([div[key] for key in keys])
        top = max(1., np.ceil(values.max() * 5) / 5)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, top)
        ax.set_xticks(angles)
        ax.set_xticklabels([])

        radial_ticks = np.linspace(0, top, 6)[1:]
        ax.set_yticks(radial_ticks)
        ax.set_yticklabels([])
        ax.grid(color='#b8b8b8', alpha=.35, linewidth=.8)

        for colour, values_q in zip(colours, values):
            v = np.r_[values_q, values_q[0]]
            ax.plot(
                closed, v, color=colour,
                lw=1.8, marker='o', ms=3.5, zorder=3
            )
            ax.fill(
                closed, v, color=colour, alpha=.18, zorder=2
            )

        for radius in radial_ticks:
            ax.text(
                np.pi / 4, radius, f'{radius:g}',
                transform=ax.transData,
                ha='left', va='center',
                fontsize=9.5, color='#333333', zorder=10,
                bbox=dict(
                    facecolor='white', edgecolor='none', pad=.6
                )
            )

        labels = [
            f'(1−R) / {div["1-R"]:g}',
            f'RMSPE / {div["RMSPE"]:g}',
            f'RMSE / {div["RMSE"]:g}',
            f'MAE / {div["MAE"]:g}',
        ]
        for label, (xy, offset, ha, va) in zip(labels, positions):
            ax.annotate(
                label, xy=xy, xycoords='axes fraction',
                xytext=offset, textcoords='offset points',
                ha=ha, va=va, fontsize=10.5,
                annotation_clip=False, clip_on=False
            )

        ax.set_title(
            title, fontsize=12, fontweight='bold', pad=30
        )
        ax.legend(
            handles=[
                Patch(
                    facecolor=c, edgecolor=c, alpha=.5,
                    label=f'P{q * 100:g}'
                )
                for c, q in zip(colours, QUANTILES)
            ],
            loc='upper center',
            bbox_to_anchor=(.5, -.12),
            ncol=3, frameon=False, fontsize=10,
            columnspacing=1.6, handlelength=1.8
        )

    return fig


def plot_extreme_ets(summary_frame):
    winds = ('horizontal', 'vertical')
    panels = {}

    for wind in winds:
        for lev in LEVELS:
            part = summary_frame.loc[
                (summary_frame.wind == wind)
                & (summary_frame.level_hpa == lev)
            ]
            panels[wind, lev] = (
                part.pivot(
                    index='quantile',
                    columns='lead_hour',
                    values='ETS'
                )
                .reindex(index=QUANTILES, columns=LEADS)
                .to_numpy(dtype=float)
            )

    all_values = np.concatenate([
        z.ravel() for z in panels.values()
    ])
    finite = all_values[np.isfinite(all_values)]
    if not finite.size:
        raise ValueError('当前六个面板没有有限ETS值。')

    vmin = float(finite.min())
    vmax = float(finite.max())
    norm = Normalize(vmin=vmin, vmax=vmax)

    cmap = ListedColormap(
        plt.get_cmap('Blues')(np.linspace(.05, .55, 256)),
        name='light_ets'
    )
    cmap.set_bad('#eeeeee')

    fig, axes = plt.subplots(
        2, len(LEVELS), figsize=(11.4, 5.5),
        squeeze=False, layout='constrained'
    )

    for ri, wind in enumerate(winds):
        for ci, lev in enumerate(LEVELS):
            ax = axes[ri, ci]
            z = panels[wind, lev]

            im = ax.imshow(
                np.ma.masked_invalid(z),
                norm=norm, cmap=cmap, aspect='auto'
            )
            ax.set_xticks(range(len(LEADS)), LEADS)
            ax.set_yticks(
                range(len(QUANTILES)),
                [f'P{q * 100:g}' for q in QUANTILES]
            )
            ax.set_xlabel('Lead time (h)')
            ax.set_title(
                f'{"Horizontal" if wind == "horizontal" else "|ω|"}, '
                f'{lev:g} hPa',
                fontsize=12, pad=7
            )

            for (i, j), val in np.ndenumerate(z):
                ax.text(
                    j, i,
                    f'{val:.2f}' if np.isfinite(val) else 'NA',
                    ha='center', va='center',
                    fontsize=9.5, color='black'
                )

    if vmin < vmax:
        fig.colorbar(
            im, ax=axes,
            label='ETS (larger is better)',
            shrink=.86, pad=.025,
            ticks=np.linspace(vmin, vmax, 6),
            format='%.2f'
        )
    else:
        fig.legend(
            handles=[
                Patch(facecolor=cmap(0.), label=f'ETS = {vmin:g}')
            ],
            loc='outside right center', frameon=False
        )

    return fig


save_figure(plot_extreme_radar(radar), RADAR_FIGURE)
save_figure(plot_extreme_ets(summary), ETS_FIGURE)
