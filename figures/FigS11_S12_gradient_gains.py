from figure_paths import DATA, RESULTS, WORK, FIGURES, setting
from pathlib import Path
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
LEADS = tuple(range(1, 7))
COMPONENTS = ('u', 'v', 'w')
LEVELS = np.array([850., 925., 1000.])
FIG_DPI = 800
FONT_SCALE = 1.0

FIGURE_DIR = FIGURES
SHOW_FIGURES = False
gain_df = pd.read_csv(RESULTS / 'gradient_comparison_figures' / 'gradient_gains_by_component_level.csv')
if FIG_DPI <= 600:
    raise ValueError('FIG_DPI必须高于600。')
if tuple(LEADS) != (1, 2, 3, 4, 5, 6):
    raise ValueError('本版S11/S12布局固定为1–6 h。')
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
COLORS = {850: '#0072B2', 925: '#D55E00', 1000: '#009E73'}
MARKERS = {'u': 'o', 'v': '^', 'w': 's'}
LABELS = {'u': r'$u$', 'v': r'$v$', 'w': r'$\omega$'}
X_MULTIPLIER = 1000.0
Y_MULTIPLIER = 1000.0
finite_x = gain_df.loc[np.isfinite(gain_df.delta_R), 'delta_R'].to_numpy() * X_MULTIPLIER
finite_y = gain_df.loc[np.isfinite(gain_df.delta_RMSPE), 'delta_RMSPE'].to_numpy() * Y_MULTIPLIER
if not len(finite_x) or not len(finite_y):
    raise ValueError('没有可绘制的有效增益，不能以0替代缺失值。')
x_extent = max(float(np.max(np.abs(finite_x))), 1e-6) * 1.14
y_extent = max(float(np.max(np.abs(finite_y))), 1e-6) * 1.14


def check_figure_text_bounds(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    artists = list(fig.texts) + list(fig.legends)
    for ax in fig.axes:
        artists.extend([ax.title, ax._left_title, ax._right_title])
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        artists.extend(label for value, label in zip(ax.get_xticks(), ax.get_xticklabels()) if xmin <= value <= xmax)
        artists.extend(label for value, label in zip(ax.get_yticks(), ax.get_yticklabels()) if ymin <= value <= ymax)
    for artist in artists:
        if not artist.get_visible():
            continue
        box = artist.get_window_extent(renderer)
        if box.width > 0 and box.height > 0 and (box.x0 < -1 or box.y0 < -1 or box.x1 > width+1 or box.y1 > height+1):
            raise RuntimeError('图中文字或图例超出画布。请减小FONT_SCALE或调整图幅/边距，不能直接导出裁切图片。')


def save_png_tiff(fig, stem):
    png = FIGURE_DIR / f'{stem}.png'
    tif = FIGURE_DIR / f'{stem}.tiff'
    png_tmp = FIGURE_DIR / f'.{stem}.tmp.png'
    tif_tmp = FIGURE_DIR / f'.{stem}.tmp.tiff'
    check_figure_text_bounds(fig)
    fig.savefig(png_tmp, dpi=FIG_DPI, facecolor='white', format='png')
    with Image.open(png_tmp) as im:
        im.convert('RGB').save(tif_tmp, format='TIFF', compression='tiff_lzw', dpi=(FIG_DPI, FIG_DPI))
    with Image.open(png_tmp) as p, Image.open(tif_tmp) as t:
        if p.size != t.size or not np.array_equal(np.asarray(p.convert('RGB')), np.asarray(t.convert('RGB'))):
            raise RuntimeError(f'{stem}: PNG与TIFF像素不一致。')
        if min(p.info.get('dpi', (0, 0))) <= 600 or min(t.info.get('dpi', (0, 0))) <= 600:
            raise RuntimeError(f'{stem}: 导出DPI不满足要求。')
        size = p.size
    os.replace(png_tmp, png)
    os.replace(tif_tmp, tif)
    print(f'{stem}: {size[0]}×{size[1]} pixels，{FIG_DPI} dpi；PNG/TIFF像素相同。')
    return [str(png), str(tif)]


figure_paths = []
style = {
    'font.family': 'DejaVu Sans', 'font.size': 10.0 * FONT_SCALE,
    'axes.titlesize': 11.5 * FONT_SCALE, 'axes.labelsize': 11.5 * FONT_SCALE,
    'xtick.labelsize': 10.5 * FONT_SCALE, 'ytick.labelsize': 10.5 * FONT_SCALE,
    'legend.fontsize': 10 * FONT_SCALE, 'axes.unicode_minus': True,
    'axes.linewidth': 0.75, 'mathtext.fontset': 'dejavusans',
}
with plt.rc_context(style):
    for stage, stem in (('final', 'FigS11'), ('first_stage', 'FigS12')):
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.2), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.105, right=0.98, bottom=0.10, top=0.855, wspace=0.19, hspace=0.29)
        for i, (lead, ax) in enumerate(zip(LEADS, axes.flat)):
            part = gain_df[(gain_df.stage == stage) & (gain_df.lead_hour == lead)]
            ax.axhline(0, color='0.35', linewidth=0.8, linestyle='--', zorder=1)
            ax.axvline(0, color='0.35', linewidth=0.8, linestyle='--', zorder=1)
            ax.grid(True, color='0.92', linewidth=0.5, zorder=0)
            missing = 0
            for _, row in part.iterrows():
                x, y = row.delta_R * X_MULTIPLIER, row.delta_RMSPE * Y_MULTIPLIER
                if np.isfinite(x) and np.isfinite(y):
                    ax.scatter(x, y, s=42, marker=MARKERS[row.component], color=COLORS[int(row.level_hpa)],
                               edgecolors='white', linewidths=0.45, alpha=0.95, zorder=3)
                else:
                    missing += 1
            if len(part) == 9 and np.isfinite(part[['delta_R', 'delta_RMSPE']].to_numpy()).all():
                ax.scatter(part.delta_R.mean()*X_MULTIPLIER, part.delta_RMSPE.mean()*Y_MULTIPLIER,
                           marker='D', s=50, color='black', edgecolors='white', linewidths=0.55, zorder=4)
            if missing:
                print(f'{stem}, {lead} h: {missing}个单元至少一项指标未定义；未用0或其他值替代。')
            ax.set_title(f'({chr(97+i)}) {lead} h', loc='left', pad=6)
            ax.set_xlim(-x_extent, x_extent)
            ax.set_ylim(-y_extent, y_extent)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, symmetric=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, symmetric=True))
            ax.tick_params(direction='out', length=3, pad=2)
        component_handles = [Line2D([], [], marker=MARKERS[c], color='none', markerfacecolor='0.35',
                                    markeredgecolor='white', markersize=6.5, label=LABELS[c]) for c in COMPONENTS]
        component_handles += [Line2D([], [], marker='D', color='none', markerfacecolor='black',
                                     markersize=6.5, label='Nine-unit mean')]
        pressure_handles = [Line2D([], [], marker='o', color='none', markerfacecolor=COLORS[int(level)],
                                   markeredgecolor='white', markersize=7, label=f'{int(level)} hPa') for level in LEVELS]
        fig.legend(handles=component_handles, loc='upper center', bbox_to_anchor=(0.53, 0.995),
                   ncol=4, frameon=False, handletextpad=0.35, columnspacing=1.5)
        fig.legend(handles=pressure_handles, loc='upper center', bbox_to_anchor=(0.53, 0.951),
                   ncol=3, frameon=False, handletextpad=0.35, columnspacing=2.0)
        fig.supxlabel(r'$\Delta R$ ($\times 10^{-3}$)', y=0.025, fontsize=11.5*FONT_SCALE)
        fig.supylabel(r'$\Delta\mathrm{RMSPE}$ ($\times 10^{-3}$)', x=0.012, fontsize=11.5*FONT_SCALE)
        figure_paths.extend(save_png_tiff(fig, stem))
        if SHOW_FIGURES:
            plt.show()
        plt.close(fig)
