from figure_paths import DATA, RESULTS, WORK, FIGURES, setting
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter

FONT_SCALE = 0.8          # 如需继续放大，改为 1.1
FIGURE_SUFFIX = '_largefont'  # 改为 '' 可覆盖原图

_FIG_EXPERIMENTS = [
    'no_pinn', 'full_pinn', 'no_continuity',
    'no_vorticity', 'no_shear'
]

_FIG_LABELS = {
    'no_pinn': 'No\nPINN',
    'full_pinn': 'Full\nPINN',
    'no_continuity': 'w/o\nContinuity',
    'no_vorticity': 'w/o\nVorticity',
    'no_shear': 'w/o\nShear',
}

_FIG_COLORS = dict(zip(
    _FIG_EXPERIMENTS,
    ['#86919D', '#2866A5', '#379380', '#C39335', '#BC5962']
))

_FIG_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 26 * FONT_SCALE,
    'axes.titlesize': 29 * FONT_SCALE,
    'axes.labelsize': 27 * FONT_SCALE,
    'xtick.labelsize': 24 * FONT_SCALE,
    'ytick.labelsize': 24 * FONT_SCALE,
    'legend.fontsize': 23 * FONT_SCALE,
    'axes.linewidth': 1.3,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',
}


def _fig_clean_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=6, width=1.3, pad=7)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#DADADA', linewidth=0.7, alpha=0.65)


def _fig_finite_range(values, lower_bound=None, upper_bound=None,
                      include=None, fraction=0.13):
    data = np.asarray(values, dtype=np.float64).ravel()
    data = data[np.isfinite(data)]

    if include is not None:
        data = np.r_[data, np.asarray(include, dtype=np.float64).ravel()]

    if data.size == 0:
        return None

    low, high = float(data.min()), float(data.max())
    span = high - low

    if span <= max(abs(low), abs(high), 1e-300) * 1e-10:
        span = max(abs(low), abs(high), 1e-5) * 0.10

    pad = fraction * span
    low, high = low - pad, high + pad

    if lower_bound is not None:
        low = max(low, lower_bound)
    if upper_bound is not None:
        high = min(high, upper_bound)
    if high <= low:
        high = low + max(abs(low), 1.0) * 0.05

    return low, high


def _fig_values(ds, variable, experiments, metric, primary_block=None,
                aggregation='all_levels'):
    array = ds[variable].sel(
        experiment=experiments,
        aggregation=aggregation,
        metric=metric
    )

    if 'block_hours' in array.dims:
        array = array.sel(block_hours=primary_block)

    return np.asarray(array.transpose('experiment').values, dtype=float)


def _fig_vertical_interval(ax, x, estimate, lo, hi, color, cap=0.075):
    if np.isfinite(lo) and np.isfinite(hi):
        ax.vlines(x, lo, hi, color=color, linewidth=2.0, zorder=3)
        ax.hlines(
            [lo, hi], x - cap, x + cap,
            color=color, linewidth=1.7, zorder=3
        )

    if np.isfinite(estimate):
        ax.scatter(
            x, estimate, s=115, color=color,
            edgecolor='white', linewidth=0.75, zorder=4
        )


def _fig_horizontal_interval(ax, y, estimate, lo, hi, color, cap=0.09):
    if np.isfinite(lo) and np.isfinite(hi):
        ax.hlines(y, lo, hi, color=color, linewidth=2.5, zorder=3)
        ax.vlines(
            [lo, hi], y - cap, y + cap,
            color=color, linewidth=1.8, zorder=3
        )

    if np.isfinite(estimate):
        ax.scatter(
            estimate, y, s=125, color=color,
            edgecolor='white', linewidth=0.75, zorder=4
        )


def _fig_save(fig, output_dir, stem, dpi, show):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for extension in ('png', 'pdf'):
        path = output_dir / f'{stem}{FIGURE_SUFFIX}.{extension}'
        fig.savefig(
            path, dpi=dpi, bbox_inches='tight', facecolor='white'
        )
        paths[extension] = str(path)

    if show:
        plt.close('all') if False else None

    plt.close(fig)
    return paths


def draw_physics_figures(summary, bootstrap, output_dir, dpi=600,
                         primary_block=72, show=True):
    experiments = [
        e for e in _FIG_EXPERIMENTS
        if e in set(map(str, summary.experiment.values))
    ]

    if 'no_pinn' not in experiments or len(experiments) < 2:
        raise ValueError('S3/S4 至少需要 No PINN 和一个 PINN 实验。')
    if 'block_hours' not in bootstrap.coords:
        raise ValueError('Bootstrap 结果缺少 block_hours 坐标。')
    if primary_block not in np.asarray(bootstrap.block_hours.values):
        raise ValueError(f'已保存结果中没有 {primary_block} h 时间块区间。')

    missing = set(experiments) - set(map(str, bootstrap.experiment.values))
    if missing:
        raise ValueError(f'Bootstrap 结果缺少实验：{sorted(missing)}')

    output = {}

    specifications = [
        (
            'cont_pred_rms',
            '(a) Continuity\nresidual',
            r'RMS (s$^{-1}$)', 0, None
        ),
        (
            'cont_error_rmse',
            '(b) Continuity-residual\nerror',
            r'RMSE (s$^{-1}$)', 0, None
        ),
        (
            'vorticity_rmse',
            '(c) Relative-vorticity\nerror',
            r'RMSE (s$^{-1}$)', 0, None
        ),
        (
            'vorticity_r',
            '(d) Relative-vorticity\ncorrelation',
            'Pearson $r$', -1, 1
        ),
        (
            'vorticity_std_ratio',
            '(e) Relative-vorticity\nvariability',
            'Standard-deviation\nratio', 0, None
        ),
        (
            'shear_rmse',
            '(f) Vertical-shear\nerror',
            r'RMSE (m s$^{-1}$ Pa$^{-1}$)', 0, None
        ),
    ]

    with plt.rc_context(_FIG_STYLE):
        fig, axes = plt.subplots(
            2, 3, figsize=(19, 13.2), layout='constrained'
        )
        fig.get_layout_engine().set(
            w_pad=0.12, h_pad=0.16, wspace=0.08, hspace=0.10
        )

        for ax, (metric, title, ylabel, floor, ceiling) in zip(
            axes.ravel(), specifications
        ):
            estimate = _fig_values(
                summary, 'value', experiments, metric
            )
            low = _fig_values(
                bootstrap, 'ci_low', experiments, metric, primary_block
            )
            high = _fig_values(
                bootstrap, 'ci_high', experiments, metric, primary_block
            )
            all_values = np.r_[estimate, low, high]

            for position, eid in enumerate(experiments):
                _fig_vertical_interval(
                    ax, position, estimate[position],
                    low[position], high[position], _FIG_COLORS[eid]
                )

            if metric == 'cont_pred_rms':
                reference = _fig_values(
                    summary, 'value', experiments, 'cont_true_rms'
                )
                ax.scatter(
                    np.arange(len(experiments)), reference,
                    marker='x', s=115, linewidth=2,
                    color='#555555', zorder=5,
                    label='ERA5 reference'
                )
                all_values = np.r_[all_values, reference]
                ax.legend(frameon=False, loc='best', handlelength=1)

            include = [1.0] if metric == 'vorticity_std_ratio' else None
            if include is not None:
                ax.axhline(
                    1.0, color='#666666', linestyle='--', linewidth=1.3
                )

            limits = _fig_finite_range(
                all_values, floor, ceiling, include
            )
            if limits is not None:
                ax.set_ylim(*limits)

            ax.set_xlim(-0.5, len(experiments) - 0.5)
            ax.set_xticks(
                np.arange(len(experiments)),
                [_FIG_LABELS[e] for e in experiments]
            )
            ax.tick_params(axis='x', labelrotation=35)

            for tick in ax.get_xticklabels():
                tick.set_ha('right')

            ax.set_title(title, loc='left', pad=14)
            ax.set_ylabel(ylabel)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

            formatter = ScalarFormatter(
                useMathText=True, useOffset=False
            )
            formatter.set_powerlimits((-3, 4))
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.get_offset_text().set_fontsize(24 * FONT_SCALE)
            _fig_clean_axis(ax)

        output['S3'] = _fig_save(
            fig, output_dir,
            'Fig_S3_PINN_physical_consistency', dpi, show
        )

        variants = [e for e in experiments if e != 'no_pinn']
        specifications = [
            ('u_rmse', '(a) Zonal wind', 'RMSE reduction (%)'),
            ('v_rmse', '(b) Meridional wind', 'RMSE reduction (%)'),
            (
                'omega_rmse', '(c) Pressure vertical\nvelocity',
                'RMSE reduction (%)'
            ),
            (
                'cont_pred_rms', '(d) Continuity residual',
                'Residual RMS\nreduction (%)'
            ),
            (
                'vorticity_rmse', '(e) Relative vorticity',
                'RMSE reduction (%)'
            ),
            (
                'shear_rmse', '(f) Vertical shear',
                'RMSE reduction (%)'
            ),
        ]

        fig, axes = plt.subplots(
            2, 3, figsize=(19, 12.8), layout='constrained'
        )
        fig.get_layout_engine().set(
            w_pad=0.12, h_pad=0.16, wspace=0.08, hspace=0.10
        )

        for ax, (metric, title, xlabel) in zip(
            axes.ravel(), specifications
        ):
            estimate = _fig_values(
                bootstrap, 'relative_reduction_pct',
                variants, metric, primary_block
            )
            low = _fig_values(
                bootstrap, 'relative_reduction_ci_low',
                variants, metric, primary_block
            )
            high = _fig_values(
                bootstrap, 'relative_reduction_ci_high',
                variants, metric, primary_block
            )

            for position, eid in enumerate(variants):
                _fig_horizontal_interval(
                    ax, position, estimate[position],
                    low[position], high[position], _FIG_COLORS[eid]
                )

            limits = _fig_finite_range(
                np.r_[estimate, low, high], include=[0.0]
            )
            if limits is not None:
                ax.set_xlim(*limits)

            ax.axvline(
                0, color='#666666', linestyle='--', linewidth=1.3
            )
            ax.set_yticks(
                np.arange(len(variants)),
                [_FIG_LABELS[e] for e in variants]
            )
            ax.set_ylim(len(variants) - 0.45, -0.55)
            ax.set_xlabel(xlabel, labelpad=30)
            ax.set_title(title, loc='left', pad=15)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

            formatter = ScalarFormatter(
                useMathText=True, useOffset=False
            )
            formatter.set_powerlimits((-3, 4))
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.get_offset_text().set_fontsize(24 * FONT_SCALE)
            _fig_clean_axis(ax)

        output['S4'] = _fig_save(
            fig, output_dir,
            'Fig_S4_PINN_block_bootstrap', dpi, show
        )

    return output


def draw_gradient_figure(gradient_df, output_dir, dpi=600, show=True):
    required = {
        'experiment', 'sample', 'term',
        'loss_ratio', 'gradient_ratio', 'cosine_to_data'
    }

    if gradient_df is None or len(gradient_df) == 0:
        warnings.warn('没有可用的梯度诊断结果，未生成 S5。')
        return None

    missing = required - set(gradient_df.columns)
    if missing:
        raise ValueError(f'梯度结果缺少字段：{sorted(missing)}')

    data = gradient_df.copy()
    experiments = [
        e for e in _FIG_EXPERIMENTS
        if e in set(data.experiment.astype(str)) and e != 'no_pinn'
    ]
    terms = ['continuity', 'vorticity', 'shear', 'physics_total']

    if not experiments:
        warnings.warn('没有可用的 PINN checkpoint 诊断结果。')
        return None

    display_rows = []
    for eid in experiments:
        for term in terms:
            group = data[
                (data.experiment == eid) & (data.term == term)
            ]
            for metric in [
                'loss_ratio', 'gradient_ratio', 'cosine_to_data'
            ]:
                values = pd.to_numeric(
                    group[metric], errors='coerce'
                ).to_numpy(dtype=float)
                values = values[np.isfinite(values)]

                if len(values) == 0:
                    continue

                q25, median, q75 = np.quantile(
                    values, [0.25, 0.5, 0.75]
                )
                display_rows.append(dict(
                    experiment=eid, term=term, metric=metric,
                    q25=q25, median=median, q75=q75, n=len(values)
                ))

    display = pd.DataFrame(display_rows)
    if display.empty:
        warnings.warn('梯度诊断结果均未定义，未生成 S5。')
        return None

    offsets = (
        np.linspace(-0.14, 0.14, len(experiments))
        if len(experiments) > 1 else [0.0]
    )
    specifications = [
        (
            'loss_ratio', '(a) Weighted physical\nloss',
            'Physical / data loss', True
        ),
        (
            'gradient_ratio', '(b) Parameter-gradient\nnorm',
            'Physical / data\ngradient norm', True
        ),
        (
            'cosine_to_data', '(c) Gradient\ndirection',
            'Cosine similarity to\ndata gradient', False
        ),
    ]

    with plt.rc_context(_FIG_STYLE):
        fig, axes = plt.subplots(
            1, 3, figsize=(19, 7.4), layout='constrained'
        )
        fig.get_layout_engine().set(
            w_pad=0.12, h_pad=0.16, wspace=0.08, hspace=0.10
        )

        for ax, (metric, title, ylabel, logarithmic) in zip(
            axes, specifications
        ):
            for offset, eid in zip(offsets, experiments):
                for position, term in enumerate(terms):
                    row = display[
                        (display.experiment == eid)
                        & (display.term == term)
                        & (display.metric == metric)
                    ]
                    if row.empty:
                        continue

                    row = row.iloc[0]
                    median = float(row['median'])
                    low, high = float(row.q25), float(row.q75)

                    if logarithmic:
                        if median <= 0:
                            continue
                        if low <= 0 or high <= 0:
                            low, high = np.nan, np.nan

                    _fig_vertical_interval(
                        ax, position + offset, median, low, high,
                        _FIG_COLORS[eid], cap=0.045
                    )

            if logarithmic:
                ax.set_yscale('log')
            else:
                ax.set_ylim(-1.06, 1.06)
                ax.axhline(
                    0, color='#666666', linestyle='--', linewidth=1.3
                )
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

            ax.set_xlim(-0.5, len(terms) - 0.5)
            ax.set_xticks(
                np.arange(len(terms)),
                ['Continuity', 'Vorticity', 'Shear', 'Total']
            )
            ax.tick_params(axis='x', labelrotation=35)
            for tick in ax.get_xticklabels():
                tick.set_ha('right')

            ax.set_title(title, loc='left', pad=15)
            ax.set_ylabel(ylabel)
            _fig_clean_axis(ax)

        handles = [
            Line2D(
                [0], [0], marker='o', linestyle='none',
                markersize=11, color=_FIG_COLORS[e],
                label=_FIG_LABELS[e].replace('\n', ' ')
            )
            for e in experiments
        ]
        axes[0].legend(handles=handles, frameon=False, loc='best')

        paths = _fig_save(
            fig, output_dir,
            'Fig_S5_PINN_gradient_diagnostics', dpi, show
        )

    return {'S5': paths, 'display_summary': display}


OUTPUT_DIR = FIGURES
PREFIX = (RESULTS / 'da_transunet_1hour_pinn_physics')
FIGURE_DPI = 800
PRIMARY_BLOCK = 72


def _read_saved_nc(name, suffix):
    path = Path(globals().get(name, str(PREFIX) + suffix))
    if not path.is_file():
        raise FileNotFoundError(f'找不到已保存结果：{path}')
    with xr.open_dataset(path) as ds:
        return ds.load()


if not isinstance(globals().get('physical_summary'), xr.Dataset):
    physical_summary = _read_saved_nc('SUMMARY_NC', '_summary.nc')

if not isinstance(globals().get('bootstrap_results'), xr.Dataset):
    bootstrap_results = _read_saved_nc(
        'BOOTSTRAP_NC', '_block_bootstrap.nc'
    )

if not isinstance(globals().get('gradient_plot_data'), pd.DataFrame):
    gradient_plot_data = pd.DataFrame()
    gradient_path = Path(globals().get(
        'GRADIENT_NC', str(PREFIX) + '_parameter_gradients.nc'
    ))

    if gradient_path.is_file():
        with xr.open_dataset(gradient_path) as ds:
            gradient_results = ds.load()

        verified = gradient_results.experiment.values[
            gradient_results.status.values == 'verified'
        ]

        if len(verified):
            names = [
                'loss_ratio_to_data', 'gradient_ratio_to_data',
                'cosine_to_data', 'trusted_sample'
            ]
            table = (
                gradient_results[names]
                .sel(experiment=verified)
                .to_dataframe()
                .reset_index()
            )

            gradient_plot_data = table.loc[
                (table.trusted_sample == 1) & (table.term != 'data')
            ].copy()

            gradient_plot_data = gradient_plot_data.rename(columns={
                'loss_ratio_to_data': 'loss_ratio',
                'gradient_ratio_to_data': 'gradient_ratio'
            })
            gradient_plot_data['term'] = (
                gradient_plot_data['term']
                .replace({'physics': 'physics_total'})
            )


figure_results = draw_physics_figures(
    physical_summary,
    bootstrap_results,
    OUTPUT_DIR,
    dpi=FIGURE_DPI,
    primary_block=PRIMARY_BLOCK,
    show=False
)

gradient_figure = draw_gradient_figure(
    gradient_plot_data,
    OUTPUT_DIR,
    dpi=FIGURE_DPI,
    show=False
)

if gradient_figure is None:
    print('S3/S4 已重绘；S5 缺少已通过核对的梯度结果，请检查代码框 7 的输出。')
else:
    print('S3、S4、S5 大字体版本已保存。')

print('保存目录：', OUTPUT_DIR)
print('文件格式：PNG（%s dpi）和 PDF' % FIGURE_DPI)
print('文件后缀：', FIGURE_SUFFIX)
