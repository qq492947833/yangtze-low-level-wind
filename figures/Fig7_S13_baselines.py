from figure_paths import DATA, RESULTS, WORK, FIGURES, setting
from pathlib import Path
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.ticker import MaxNLocator
FIG_DPI = 800
FONT_SCALE = 1.0
LEADS = tuple(range(1, 7))
COMPONENTS = ('u', 'v', 'w')
MODEL_LABELS = {'B': 'DA-TransUNet', 'P': 'Persistence', 'Ridge': 'Ridge'}
curves = pd.read_csv(RESULTS / 'fig7_s13_alpha1e8' / 'Fig7_S13_alpha1e8_curve_values.csv')
def plot_comparison(curves, names):
    if tuple(names) not in (('R', 'RMSPE'), ('RMSE', 'MAE')):
        raise ValueError('仅支持图7的R/RMSPE和图S13的RMSE/MAE。')
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 14 * FONT_SCALE,
        'axes.titlesize': 15 * FONT_SCALE, 'axes.labelsize': 14 * FONT_SCALE,
        'xtick.labelsize': 13 * FONT_SCALE, 'ytick.labelsize': 13 * FONT_SCALE,
        'savefig.facecolor': 'white', 'axes.unicode_minus': True,
    })
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.0), layout='constrained')
    fig.get_layout_engine().set(w_pad=.04, h_pad=.04, wspace=.025, hspace=.035,
                                rect=(0., .065, 1., .935))
    styles = [('B', '#2368a2', 'o', '-'),
              ('P', '#cf7237', 's', '-'),
              ('Ridge', '#39844e', '^', '--')]
    labels = {'u': 'u', 'v': 'v', 'w': r'$\omega$'}
    for r, metric in enumerate(names):
        for c, component in enumerate(COMPONENTS):
            ax = axes[r, c]
            for group, color, marker, linestyle in styles:
                selected = curves.loc[(curves.group == group) & (curves.component == component)]
                selected = selected.set_index('lead_hour').reindex(LEADS)
                values = selected[metric].to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError(f'{metric}/{component}/{group}: 曲线有未定义值，请检查H盘指标表和掩膜。')
                ax.plot(LEADS, values, color=color, marker=marker, linestyle=linestyle,
                        lw=2., ms=5., label=MODEL_LABELS[group])
            ax.set_title(f'({chr(97 + 3*r + c)}) {metric} — {labels[component]}', pad=7)
            ax.set_xticks(LEADS)
            ax.set_xlabel('Forecast lead time (h)')
            unit = r'Pa s$^{-1}$' if component == 'w' else r'm s$^{-1}$'
            ax.set_ylabel(f'{metric} ({unit})' if metric in ('RMSE', 'MAE') else metric)
            ax.set_xlim(.8, 6.2)
            ax.margins(y=.09)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.grid(alpha=.18, linewidth=.7)
            ax.spines[['top', 'right']].set_visible(False)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', bbox_to_anchor=(.5, .004),
               ncol=3, frameon=False, fontsize=12 * FONT_SCALE,
               handlelength=2.5, columnspacing=2.)
    return fig


for stem, names in [("Fig7",("R","RMSPE")), ("FigS13",("RMSE","MAE"))]:
    fig=plot_comparison(curves,names)
    fig.savefig(FIGURES / f'{stem}.png', dpi=FIG_DPI, facecolor='white', bbox_inches='tight', pad_inches=.06)
    with Image.open(FIGURES / f'{stem}.png') as image:
        image.convert('RGB').save(FIGURES / f'{stem}.tiff', compression='tiff_lzw', dpi=(FIG_DPI, FIG_DPI))
    plt.close(fig)
