from figure_paths import DATA, RESULTS, WORK, FIGURES, setting
from pathlib import Path
import string
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter


OUTPUT_DIR = FIGURES
OUTPUT_PREFIX = 'da_transunet_1hour_pinn_seed_metrics_v1'

FONT_SCALE = 0.8
FIGURE_DPI = 800
SHOW_FIGURES = False

W_EXTERNAL_SCALE = -0.1

SEEDS = (42, 2026)
EXPERIMENTS = ('no_pinn', 'no_shear')
COMPONENTS = ('u', 'v', 'w')
LEVELS = (850, 925, 1000)
METRICS = ('pearson_r', 'rmse', 'rmspe')

NAMES = {
    'no_pinn': 'No PINN',
    'no_shear': 'w/o Shear',
}
COLORS = {
    'no_pinn': '#727272',
    'no_shear': '#C43B3B',
}
TITLES = {
    'u': r'$u$',
    'v': r'$v$',
    'w': r'$\omega$',
}
SEED_STYLES = {
    42: dict(linestyle='-', marker='o'),
    2026: dict(linestyle='--', marker='s'),
}


summary_path = (RESULTS / 'da_transunet_1hour_pinn_seed_metrics_v1_summary.csv')

if not summary_path.is_file():
    raise FileNotFoundError(f'未找到已保存的指标文件：{summary_path}')

if FONT_SCALE <= 0 or FIGURE_DPI <= 0:
    raise ValueError('FONT_SCALE 和 FIGURE_DPI 必须大于 0。')

if not np.isfinite(W_EXTERNAL_SCALE) or W_EXTERNAL_SCALE == 0:
    raise ValueError('W_EXTERNAL_SCALE 必须是有限非零数。')

summary = pd.read_csv(summary_path, encoding='utf-8-sig')

required = {
    'seed', 'experiment', 'component', 'scope', 'level_hpa',
    *METRICS,
}
missing = required.difference(summary.columns)
if missing:
    raise ValueError(f'指标文件缺少列：{sorted(missing)}')

for column in ('seed', 'level_hpa', *METRICS):
    summary[column] = pd.to_numeric(summary[column], errors='raise')

summary = summary.loc[
    summary.seed.isin(SEEDS)
    & summary.experiment.isin(EXPERIMENTS)
    & summary.component.isin(COMPONENTS)
].copy()

summary.loc[summary.component.eq('w'), 'rmse'] /= abs(W_EXTERNAL_SCALE)


def validated_scope(scope):
    frame = summary.loc[summary.scope.eq(scope)].copy()

    keys = ['seed', 'experiment', 'component']
    dimensions = [SEEDS, EXPERIMENTS, COMPONENTS]

    if scope == 'all_levels':
        if frame.level_hpa.notna().any():
            raise ValueError('all_levels 行的 level_hpa 应为空。')
    else:
        frame = frame.loc[frame.level_hpa.isin(LEVELS)]
        keys += ['level_hpa']
        dimensions += [LEVELS]

    if frame.duplicated(keys).any():
        raise ValueError(
            f'{scope} 存在重复记录，请检查原始 summary.csv。'
        )

    expected = pd.MultiIndex.from_product(dimensions, names=keys)
    frame = frame.set_index(keys)

    missing_rows = expected.difference(frame.index)
    if len(missing_rows):
        raise ValueError(f'{scope} 缺少记录：{list(missing_rows)}')

    frame = frame.reindex(expected)

    for metric in METRICS:
        values = frame[metric].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]

        if np.isinf(values).any():
            raise ValueError(f'{scope}/{metric} 中存在无穷大。')

        if metric == 'pearson_r' and np.any(np.abs(finite) > 1 + 1e-10):
            raise ValueError(f'{scope} 中存在超出 [-1, 1] 的相关系数。')

        if metric != 'pearson_r' and np.any(finite < 0):
            raise ValueError(f'{scope}/{metric} 中存在负误差。')

        if np.isnan(values).any():
            warnings.warn(
                f'{scope}/{metric} 存在 NaN，将保留缺失，不填 0。'
            )

    return frame


def metric_limits(values, metric):
    """依据当前子图的数据范围设置纵轴，保留适量边距。"""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return (-1, 1) if metric == 'pearson_r' else (0, 1)

    low, high = float(values.min()), float(values.max())

    if metric == 'pearson_r':
        pad = max((high - low) * 0.22, 1e-5)
        return max(-1, low - pad), min(1, high + pad)

    pad = max(
        (high - low) * 0.22,
        max(abs(low), abs(high)) * 1e-4,
        1e-12,
    )
    return max(0, low - pad), high + pad


def style_axis(ax, values, metric, component, panel):
    ax.set_title(
        f'({string.ascii_lowercase[panel]}) {TITLES[component]}',
        pad=12,
    )

    if metric == 'pearson_r':
        ylabel = 'Pearson R'
    elif metric == 'rmse':
        ylabel = (
            r'RMSE (Pa s$^{-1}$)'
            if component == 'w'
            else r'RMSE (m s$^{-1}$)'
        )
    else:
        ylabel = 'RMSPE'

    ax.set_ylabel(ylabel, labelpad=9)
    ax.set_ylim(*metric_limits(values, metric))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))

    formatter = ScalarFormatter(useOffset=False, useMathText=True)
    formatter.set_powerlimits((-3, 4))
    ax.yaxis.set_major_formatter(formatter)

    ax.grid(axis='y', color='#DEDEDE', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=5, width=1.1, pad=7)


def save_figure(fig, name):
    for extension in ('png', 'pdf'):
        destination = OUTPUT_DIR / f'{name}.{extension}'
        fig.savefig(
            destination,
            dpi=FIGURE_DPI,
            facecolor='white',
            bbox_inches='tight',
            pad_inches=0.15,
        )
        print(f'已保存：{destination}')

    if SHOW_FIGURES:
        plt.close('all') if False else None

    plt.close(fig)


all_levels = validated_scope('all_levels')
by_level = validated_scope('level')

plot_rc = {
    'font.family': 'DejaVu Sans',
    'font.size': 26 * FONT_SCALE,
    'axes.titlesize': 29 * FONT_SCALE,
    'axes.labelsize': 27 * FONT_SCALE,
    'xtick.labelsize': 24 * FONT_SCALE,
    'ytick.labelsize': 24 * FONT_SCALE,
    'legend.fontsize': 24 * FONT_SCALE,
    'axes.linewidth': 1.1,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}


with plt.rc_context(plot_rc):

    fig, axes = plt.subplots(3, 3, figsize=(16.8, 14.0))
    fig.subplots_adjust(
        left=0.105, right=0.98, bottom=0.09, top=0.885,
        wspace=0.52, hspace=0.45,
    )

    offsets = {'no_pinn': -0.065, 'no_shear': 0.065}

    for row, metric in enumerate(METRICS):
        for col, component in enumerate(COMPONENTS):
            ax = axes[row, col]
            values = []

            for si, seed in enumerate(SEEDS):
                y = np.array([
                    all_levels.loc[(seed, e, component), metric]
                    for e in EXPERIMENTS
                ], dtype=float)

                x = np.array([
                    si + offsets[e] for e in EXPERIMENTS
                ])
                values.extend(y)

                if np.isfinite(y).all():
                    ax.plot(
                        x, y,
                        color='#B8B8B8',
                        linewidth=2.2,
                        zorder=2,
                    )

                for e, xx, yy in zip(EXPERIMENTS, x, y):
                    if np.isfinite(yy):
                        ax.scatter(
                            xx, yy,
                            s=115,
                            color=COLORS[e],
                            edgecolor='white',
                            linewidth=0.9,
                            zorder=3,
                        )

            style_axis(
                ax, values, metric, component, row * 3 + col
            )
            ax.set_xlim(-0.35, 1.35)
            ax.set_xticks([0, 1], [str(s) for s in SEEDS])

            if row == 2:
                ax.set_xlabel('Random seed', labelpad=10)

    handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='none',
            markersize=11,
            markerfacecolor=COLORS[e],
            markeredgecolor='white',
            label=NAMES[e],
        )
        for e in EXPERIMENTS
    ]

    fig.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.54, 0.99),
        ncol=2,
        frameon=False,
        handletextpad=0.5,
        columnspacing=2.4,
    )

    save_figure(
        fig,
        'Fig_S6_PINN_paired_seeds_test_metrics_omega',
    )

    fig, axes = plt.subplots(3, 3, figsize=(16.8, 14.0))
    fig.subplots_adjust(
        left=0.105, right=0.98, bottom=0.09, top=0.865,
        wspace=0.52, hspace=0.45,
    )

    level_x = np.arange(len(LEVELS))

    for row, metric in enumerate(METRICS):
        for col, component in enumerate(COMPONENTS):
            ax = axes[row, col]
            values = []

            for e in EXPERIMENTS:
                for seed in SEEDS:
                    y = np.array([
                        by_level.loc[
                            (seed, e, component, lev), metric
                        ]
                        for lev in LEVELS
                    ], dtype=float)

                    values.extend(y)

                    ax.plot(
                        level_x, y,
                        color=COLORS[e],
                        linewidth=2.3,
                        markersize=8,
                        markeredgewidth=0.9,
                        markeredgecolor='white',
                        zorder=3,
                        **SEED_STYLES[seed],
                    )

            style_axis(
                ax, values, metric, component, row * 3 + col
            )
            ax.set_xlim(-0.15, 2.15)
            ax.set_xticks(
                level_x, [str(lev) for lev in LEVELS]
            )

            if row == 2:
                ax.set_xlabel('Pressure level (hPa)', labelpad=10)

    handles = [
        Line2D(
            [0], [0],
            color=COLORS[e],
            linewidth=2.3,
            markersize=9,
            markeredgecolor='white',
            markeredgewidth=0.9,
            label=f'{NAMES[e]}, seed {seed}',
            **SEED_STYLES[seed],
        )
        for e in EXPERIMENTS
        for seed in SEEDS
    ]

    fig.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.54, 0.995),
        ncol=2,
        frameon=False,
        handlelength=2.1,
        handletextpad=0.6,
        columnspacing=1.6,
        labelspacing=0.5,
    )

    save_figure(
        fig,
        'Fig_S7_PINN_paired_seeds_test_metrics_by_level_omega',
    )


print(f'读取指标文件：{summary_path}')
print(
    f'omega 的 RMSE 已乘以 {1 / abs(W_EXTERNAL_SCALE):g}，'
    '单位为 Pa s^-1；R、RMSPE 保持不变。'
)
