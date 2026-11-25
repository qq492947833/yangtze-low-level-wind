from figure_paths import DATA, RESULTS, WORK, FIGURES, setting
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

METRICS_NC = RESULTS / 'da_transunet_1hour_pinn_comparison_metrics.nc'
OUTPUT_PREFIX = FIGURES / 'Fig_S2_PINN_1hour_ablation'
DPI = 800

experiment_ids = [
    'no_pinn',
    'full_pinn',
    'no_continuity',
    'no_vorticity',
    'no_shear'
]
labels = [
    'No PINN',
    'Full PINN',
    'w/o\nContinuity',
    'w/o\nVorticity',
    'w/o\nShear'
]
components = ['u', 'v', 'w']
component_titles = [r'$u$', r'$v$', r'$\omega$']
colors = ['#86919D', '#2866A5', '#379380', '#C39335', '#BC5962']
metric_names = [
    'pearson_r_mean_all_levels',
    'rmspe_mean_all_levels'
]
ylabels = ['Pearson correlation (R)', 'RMSPE']

if not METRICS_NC.is_file():
    raise FileNotFoundError(f'未找到指标文件：{METRICS_NC}')

with xr.open_dataset(METRICS_NC, engine='netcdf4') as saved:
    missing = [name for name in metric_names if name not in saved]
    if missing:
        raise KeyError(f'指标文件缺少变量：{missing}')

    plot_ds = saved[metric_names].sel(
        experiment=experiment_ids,
        component=components
    ).load()

style = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
    'font.size': 17,
    'axes.titlesize': 21,
    'axes.labelsize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 16,
    'axes.linewidth': 1.1,
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
}

with plt.rc_context(style):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=120)
    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.115,
        top=0.94,
        wspace=0.32,
        hspace=0.45
    )
    positions = np.arange(len(experiment_ids))

    for row, metric in enumerate(metric_names):
        for col, component in enumerate(components):
            ax = axes[row, col]
            values = (
                plot_ds[metric]
                .sel(component=component)
                .transpose('experiment')
                .values.astype(float)
            )

            finite = np.isfinite(values)
            if not finite.all():
                bad = [
                    experiment_ids[i]
                    for i in np.flatnonzero(~finite)
                ]
                plt.close(fig)
                raise ValueError(
                    f'{metric}/{component} 存在非有限值，请核对指标：{bad}'
                )

            low, high = float(values.min()), float(values.max())
            span = high - low
            if span == 0:
                span = max(abs(high) * 0.01, 1.e-4)

            bottom = low - 0.18 * span
            top = high + 0.30 * span
            if row == 1:
                bottom = max(0.0, bottom)

            ax.set_ylim(bottom, top)
            ax.bar(
                positions,
                values - bottom,
                bottom=bottom,
                width=0.64,
                color=colors,
                edgecolor='white',
                linewidth=0.8,
                zorder=3
            )

            for x, value in zip(positions, values):
                ax.annotate(
                    f'{value:.5f}',
                    xy=(x, value),
                    xytext=(0, 6),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    fontsize=14,
                    bbox=(dict(facecolor='white', edgecolor='none', alpha=1.0, pad=0.2)
                          if f'{value:.5f}' in {'0.04219', '0.03912', '0.04490', '0.69860'} else None)
                )

            letter = 'abcdef'[row * 3 + col]
            ax.set_title(
                f'({letter}) {component_titles[col]}',
                loc='left',
                pad=13
            )
            ax.set_ylabel(ylabels[row], labelpad=10)
            ax.set_xticks(positions)
            ax.set_xticklabels(
                labels,
                rotation=20,
                ha='right',
                rotation_mode='anchor'
            )
            ax.tick_params(axis='x', length=0, pad=9)
            ax.tick_params(axis='y', direction='out', length=4, pad=5)

            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            ax.set_axisbelow(True)
            ax.grid(
                axis='y',
                color='#D9D9D9',
                linewidth=0.7,
                alpha=0.6
            )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.margins(x=0.055)

    for suffix in ['.png', '.pdf']:
        output = OUTPUT_PREFIX.with_suffix(suffix)
        fig.savefig(
            output,
            dpi=DPI,
            bbox_inches='tight',
            pad_inches=0.15
        )
        print('已保存：', output)

    plt.close('all') if False else None
    plt.close(fig)
