from script.pipeline_manager import PipelineMemory, pipeline_task
from script.graphics.common import *
import os
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib.lines import Line2D
from tueplots import bundles, fontsizes

from typing import Dict, Any, List

plt.rcParams.update(bundles.icml2024())
fontsizes.icml2024()


root = 'experiments/tmscm_sym/ablation'


def add_flag(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
):
    arguments = metadata['arguments']
    df['dataset'] = arguments['dataset']
    df['exogenous_distribution'] = arguments['exogenous_distribution']
    df['markovian'] = 1 if arguments['markovian'] else 0
    df['solution_mapping'] = arguments['solution_mapping']
    df['causal_ordered'] = 1 if arguments['causal_ordered'] else 0
    df['triangular'] = 1 if arguments['triangular'] else 0


@pipeline_task('post_tmscm_sym_ablation')
def post_tmscm_sym_ablation(pipeline_memory: PipelineMemory, has_done: bool):
    val_df_filepath = 'script/graphics/cache/tmscm_sym_ablation_val.csv'
    test_df_filepath = 'script/graphics/cache/tmscm_sym_ablation_test.csv'

    # If csv file has been created
    if os.path.exists(val_df_filepath) and os.path.exists(test_df_filepath):
        val_df = pd.read_csv(val_df_filepath)
        test_df = pd.read_csv(test_df_filepath)
        pipeline_memory.set('tmscm_sym_ablation_val', val_df)
        pipeline_memory.set('tmscm_sym_ablation_test', test_df)
        return

    # Otherwise, create the complete csv
    _, dirs, _ = next(os.walk(root))
    val_df = []
    test_df = []
    for dir in dirs:
        val_filepath = os.path.join(root, dir, 'val-result.csv')
        val_df_i = pd.read_csv(val_filepath)
        test_filepath = os.path.join(root, dir, 'test-result.csv')
        test_df_i = pd.read_csv(test_filepath)
        meta_filepath = os.path.join(root, dir, 'metadata')
        metadata = th.load(meta_filepath, weights_only=True)
        add_flag(val_df_i, metadata=metadata)
        add_flag(test_df_i, metadata=metadata)
        val_df.append(val_df_i)
        test_df.append(test_df_i)
    val_df = pd.concat(val_df)
    test_df = pd.concat(test_df)
    val_df.to_csv(val_df_filepath)
    test_df.to_csv(test_df_filepath)
    pipeline_memory.set('tmscm_sym_ablation_val', val_df)
    pipeline_memory.set('tmscm_sym_ablation_test', test_df)


tmpl_appendix_sym_1 = """
\\begin{{table}}[t]
\\caption{{Write your description here.}}
\\label{{sample-table}}
\\vskip 0.15in
\\begin{{center}}
\\begin{{small}}
\\begin{{sc}}
\\begin{{tabular}}{{llcccccccc}}
\\toprule
 &  & \\multicolumn{{3}}{{c}}{{Barbell}} & \\multicolumn{{3}}{{c}}{{Stair}} \\\\
      \\cmidrule(lr){{3-5}}                \\cmidrule(lr){{6-8}}
Method &  & Obs$_{{\\text{{WD}}}}$ & Ctf$_{{\\text{{RMSE}}}}$ & Ctf$_{{\\text{{WD}}}}$ & Obs$_{{\\text{{WD}}}}$ & Ctf$_{{\\text{{RMSE}}}}$ & Ctf$_{{\\text{{WD}}}}$ \\\\
\\midrule
\\multirow{{3}}{{*}}{{DNME}} & -     & {} & {} & {} & {} & {} & {} & \\\\
                             & w/o O & {} & {} & {} & {} & {} & {} & \\\\
                             & w/o M & {} & {} & {} & {} & {} & {} & \\\\
\\midrule
\\multirow{{3}}{{*}}{{TNME}} & -     & {} & {} & {} & {} & {} & {} & \\\\
                             & w/o O & {} & {} & {} & {} & {} & {} & \\\\
                             & w/o M & {} & {} & {} & {} & {} & {} & \\\\
\\midrule
\\multirow{{4}}{{*}}{{CMSM}} & -     & {} & {} & {} & {} & {} & {} & \\\\
                             & w/o O & {} & {} & {} & {} & {} & {} & \\\\
                             & w/o M & {} & {} & {} & {} & {} & {} & \\\\
                             & w/o T & {} & {} & {} & {} & {} & {} & \\\\
\\midrule
\\multirow{{4}}{{*}}{{TVSM}} & -     & {} & {} & {} & {} & {} & {} & \\\\
                             & w/o O & {} & {} & {} & {} & {} & {} & \\\\
                             & w/o M & {} & {} & {} & {} & {} & {} & \\\\
                             & w/o T & {} & {} & {} & {} & {} & {} & \\\\
\\bottomrule
\\end{{tabular}}
\\end{{sc}}
\\end{{small}}
\\end{{center}}
\\vskip -0.1in
\\end{{table}}
"""


@pipeline_task('tmscm_sym_ablation_table_1')
def tmscm_sym_ablation_table_1(
    pipeline_memory: PipelineMemory,
    has_done: bool,
):
    table_filepath = 'script/graphics/tables/ablation_appendix_sym_1.tex'
    if os.path.exists(table_filepath):
        return

    df = pipeline_memory.get('tmscm_sym_ablation_test')

    values = np.zeros((14, 12))
    row, col = 0, 0
    # Calculate values
    for method in ['dnme', 'tnme', 'cmsm', 'tvsm']:
        for ablation in [None, 'causal_ordered', 'markovian', 'triangular']:
            if method in ['dnme', 'tnme'] and ablation == 'triangular':
                continue
            for dataset in ['barbell', 'stair']:
                cond = {
                    'dataset': dataset,
                    'solution_mapping': method,
                    'causal_ordered': 1,
                    'markovian': 1,
                    'triangular': 1,
                }
                if ablation is not None:
                    cond[ablation] = 0
                for metric in ['obs_wd', 'ctf_rmse', 'ctf_wd']:
                    mean, ci95 = mean_ci95_by(
                        df[df[metric] < 1000],
                        column=metric,
                        conditions=cond,
                    )
                    # Set values
                    values[row, col] = mean
                    values[row, col + 1] = ci95
                    col += 2
                    if col == 12:
                        row, col = row + 1, 0

    # Mark best
    top = 0
    is_best = np.zeros_like(values, dtype=bool)
    for i in [3, 3, 4, 4]:
        ranged = values[top:top + i, :]
        min_indices = np.argmin(ranged, axis=0) + top
        for col, min_row in enumerate(min_indices):
            is_best[min_row, col] = True
        top += i

    # Set format
    format_values = []
    for i in range(values.shape[0]):
        for j in range(values.shape[1] // 2):
            mean, ci95 = values[i, j * 2], values[i, j * 2 + 1]
            bold = is_best[i, j * 2]
            format_val = format_mean_ci95(mean, ci95, bold=bold)
            format_values.append(format_val)

    tab = tmpl_appendix_sym_1.format(*format_values)
    with open(table_filepath, 'w+', encoding='utf-8') as f:
        f.write(tab)


tmpl_appendix_sym_2 = """
\\begin{{table}}[t]
\\caption{{Write your description here.}}
\\label{{sample-table}}
\\vskip 0.15in
\\begin{{center}}
\\begin{{small}}
\\begin{{sc}}
\\begin{{tabular}}{{llcccccccc}}
\\toprule
 &  & \\multicolumn{{3}}{{c}}{{Fork}} & \\multicolumn{{3}}{{c}}{{Backdoor}} \\\\
      \\cmidrule(lr){{3-5}}                \\cmidrule(lr){{6-8}}
Method & & Obs$_{{\\text{{WD}}}}$ & Ctf$_{{\\text{{RMSE}}}}$ & Ctf$_{{\\text{{WD}}}}$ & Obs$_{{\\text{{WD}}}}$ & Ctf$_{{\\text{{RMSE}}}}$ & Ctf$_{{\\text{{WD}}}}$ \\\\
\\midrule
\\multirow{{3}}{{*}}{{DNME}} & -     & {} & {} & {} & {} & {} & {} & \\\\
                             & w/o O & {} & {} & {} & {} & {} & {} & \\\\
                             & w/o M & {} & {} & {} & {} & {} & {} & \\\\
\\midrule
\\multirow{{3}}{{*}}{{TNME}} & -     & {} & {} & {} & {} & {} & {} & \\\\
                             & w/o O & {} & {} & {} & {} & {} & {} & \\\\
                             & w/o M & {} & {} & {} & {} & {} & {} & \\\\
\\midrule
\\multirow{{4}}{{*}}{{CMSM}} & -     & {} & {} & {} & {} & {} & {} & \\\\
                             & w/o O & {} & {} & {} & {} & {} & {} & \\\\
                             & w/o M & {} & {} & {} & {} & {} & {} & \\\\
                             & w/o T & {} & {} & {} & {} & {} & {} & \\\\
\\midrule
\\multirow{{4}}{{*}}{{TVSM}} & -     & {} & {} & {} & {} & {} & {} & \\\\
                             & w/o O & {} & {} & {} & {} & {} & {} & \\\\
                             & w/o M & {} & {} & {} & {} & {} & {} & \\\\
                             & w/o T & {} & {} & {} & {} & {} & {} & \\\\
\\bottomrule
\\end{{tabular}}
\\end{{sc}}
\\end{{small}}
\\end{{center}}
\\vskip -0.1in
\\end{{table}}
"""


@pipeline_task('tmscm_sym_ablation_table_2')
def tmscm_sym_ablation_table_2(
    pipeline_memory: PipelineMemory,
    has_done: bool,
):
    table_filepath = 'script/graphics/tables/ablation_appendix_sym_2.tex'
    if os.path.exists(table_filepath):
        return

    df = pipeline_memory.get('tmscm_sym_ablation_test')

    values = np.zeros((14, 12))
    row, col = 0, 0
    # Calculate values
    for method in ['dnme', 'tnme', 'cmsm', 'tvsm']:
        for ablation in [None, 'causal_ordered', 'markovian', 'triangular']:
            if method in ['dnme', 'tnme'] and ablation == 'triangular':
                continue
            for dataset in ['fork', 'backdoor']:
                cond = {
                    'dataset': dataset,
                    'solution_mapping': method,
                    'causal_ordered': 1,
                    'markovian': 1,
                    'triangular': 1,
                }
                if ablation is not None:
                    cond[ablation] = 0
                for metric in ['obs_wd', 'ctf_rmse', 'ctf_wd']:
                    mean, ci95 = mean_ci95_by(
                        df[df[metric] < 1000],
                        column=metric,
                        conditions=cond,
                    )
                    # Set values
                    values[row, col] = mean
                    values[row, col + 1] = ci95
                    col += 2
                    if col == 12:
                        row, col = row + 1, 0

    # Mark best
    top = 0
    is_best = np.zeros_like(values, dtype=bool)
    for i in [3, 3, 4, 4]:
        ranged = values[top:top + i, :]
        min_indices = np.argmin(ranged, axis=0) + top
        for col, min_row in enumerate(min_indices):
            is_best[min_row, col] = True
        top += i

    # Set format
    format_values = []
    for i in range(values.shape[0]):
        for j in range(values.shape[1] // 2):
            mean, ci95 = values[i, j * 2], values[i, j * 2 + 1]
            bold = is_best[i, j * 2]
            format_val = format_mean_ci95(mean, ci95, bold=bold)
            format_values.append(format_val)

    tab = tmpl_appendix_sym_2.format(*format_values)
    with open(table_filepath, 'w+', encoding='utf-8') as f:
        f.write(tab)


palette = "tab10"


def plot_gpr(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    alpha_noise: float = 0.2,
    n_restarts_optimizer: int = 5,
) -> plt.Figure:
    groups = df[hue_col].unique()
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    x_min, x_max = df[x_col].min(), df[x_col].max()
    x_pred = np.linspace(x_min, x_max, 200).reshape(-1, 1)
    palettes = sns.color_palette(palette, n_colors=len(groups))
    group_colors = {grp: palettes[i] for i, grp in enumerate(groups)}
    pred_list = []
    for grp in groups:
        df_grp = df[df[hue_col] == grp]
        X = df_grp[x_col].values.reshape(-1, 1)
        y = df_grp[y_col].values
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha_noise,
            n_restarts_optimizer=n_restarts_optimizer)
        gp.fit(X, y)
        y_pred, sigma = gp.predict(x_pred, return_std=True)
        ci = 1.96 * sigma
        y_lower = y_pred - ci
        y_upper = y_pred + ci
        df_pred = pd.DataFrame({
            x_col: x_pred.ravel(),
            "y_pred": y_pred,
            "y_lower": y_lower,
            "y_upper": y_upper,
            hue_col: grp
        })
        pred_list.append(df_pred)
    df_pred_all = pd.concat(pred_list, ignore_index=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=group_colors,
        alpha=0.2,
        ax=ax
    )
    for grp in groups:
        df_grp_pred = df_pred_all[df_pred_all[hue_col] == grp]
        color = group_colors[grp]
        ax.plot(df_grp_pred[x_col], df_grp_pred["y_pred"], color=color)
        ax.fill_between(
            df_grp_pred[x_col],
            df_grp_pred["y_lower"],
            df_grp_pred["y_upper"],
            color=color,
            alpha=0.2
        )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_ylim(bottom=0, top=None)
    plt.legend([], [], frameon=False)
    return fig


def plot_sliding_window(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    window_width: float = None,
    n_pred: int = 200,
) -> plt.Figure:
    groups = df[hue_col].unique()
    x_min, x_max = df[x_col].min(), df[x_col].max()
    if window_width is None:
        window_width = (x_max - x_min) / 10
    x_pred = np.linspace(x_min, x_max, n_pred)
    palettes = sns.color_palette(palette, n_colors=len(groups))
    group_colors = {grp: palettes[i] for i, grp in enumerate(groups)}
    pred_list = []
    for grp in groups:
        sub = df[df[hue_col] == grp]
        X = sub[x_col].values
        y = sub[y_col].values
        means = []
        stds = []
        for x0 in x_pred:
            w = np.exp(-0.5 * ((X - x0) / window_width) ** 2)
            w_sum = w.sum()
            mu = (w * y).sum() / w_sum
            var = (w * (y - mu) ** 2).sum() / w_sum
            means.append(mu)
            stds.append(np.sqrt(var))
        df_pred = pd.DataFrame({
            x_col:     x_pred,
            "y_pred":  np.array(means),
            "y_lower": np.array(means) - np.array(stds),
            "y_upper": np.array(means) + np.array(stds),
            hue_col:   grp
        })
        pred_list.append(df_pred)
    df_pred_all = pd.concat(pred_list, ignore_index=True)
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=group_colors,
        alpha=0.2,
        ax=ax
    )
    for grp in groups:
        dfp = df_pred_all[df_pred_all[hue_col] == grp]
        c = group_colors[grp]
        ax.plot(dfp[x_col], dfp["y_pred"], color=c, linewidth=2)
        ax.fill_between(
            dfp[x_col],
            dfp["y_lower"],
            dfp["y_upper"],
            color=c,
            alpha=0.2
        )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_ylim(bottom=0, top=None)
    plt.legend([], [], frameon=False)
    return fig


def plot_poly_regression(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    degree: int = 3,
    n_pred: int = 200,
) -> plt.Figure:
    groups = df[hue_col].unique()
    x_min, x_max = df[x_col].min(), df[x_col].max()
    x_pred = np.linspace(x_min, x_max, n_pred)

    def make_design(x):
        return np.vander(x, N=degree+1, increasing=True)
    palettes = sns.color_palette(palette, n_colors=len(groups))
    group_colors = {g: palettes[i] for i, g in enumerate(groups)}
    pred_list = []
    for grp in groups:
        sub = df[df[hue_col] == grp]
        X = sub[x_col].values
        y = sub[y_col].values
        X_design = make_design(X)
        model = sm.OLS(y, X_design).fit()
        Xp_design = make_design(x_pred)
        pred = model.get_prediction(Xp_design).summary_frame(alpha=0.05)
        pred_list.append(pd.DataFrame({
            x_col: x_pred,
            "y_pred": pred["mean"],
            "y_lower": pred["mean_ci_lower"],
            "y_upper": pred["mean_ci_upper"],
            hue_col: grp
        }))
    df_pred_all = pd.concat(pred_list, ignore_index=True)
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=group_colors,
        alpha=0.2,
        ax=ax
    )
    for grp in groups:
        dfg = df_pred_all[df_pred_all[hue_col] == grp]
        c = group_colors[grp]
        ax.plot(dfg[x_col], dfg["y_pred"], color=c)
        ax.fill_between(
            dfg[x_col],
            dfg["y_lower"],
            dfg["y_upper"],
            color=c,
            alpha=0.2
        )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_ylim(bottom=0)
    plt.legend([], [], frameon=False)
    return fig


def plot_legend(groups: List[str] = []) -> plt.Figure:
    palettes = sns.color_palette(palette, n_colors=len(groups))
    group_colors = {grp: palettes[i] for i, grp in enumerate(groups)}
    fig, ax = plt.subplots(figsize=(2, 0.5))
    handles = []
    for grp in groups:
        color = group_colors[grp]
        handles.append(
            Line2D(
                xdata=[],
                ydata=[],
                lw=1.5,
                linestyle="-",
                color=color,
                label="\\rmfamily " + grp,
            ))
    L = ax.legend(handles=handles, ncols=4, frameon=False)
    ax.axis("off")
    plt.setp(L.texts, family='Times New Roman')
    return fig


@pipeline_task('tmscm_sym_convergence_figure')
def tmscm_sym_convergence_figure(
    pipeline_memory: PipelineMemory,
    has_done: bool,
    curve_type: str = 'win'  # or 'gpr', 'poly'
):
    df = pipeline_memory.get('tmscm_sym_ablation_val')
    dirpath = 'script/graphics/figures/convergence_main_text_sym_' + curve_type
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    def _plot_legend():
        # Draw legend figure
        legend_filepath = os.path.join(dirpath, f'legend.pdf')
        if os.path.exists(legend_filepath):
            return
        fig = plot_legend(groups=[
            'TM-SCM',
            'w/o O',
            'w/o M',
            'w/o T',
        ])
        plt.tight_layout()
        fig.savefig(legend_filepath)
        plt.clf()
        plt.close()

        print('Figure :', legend_filepath, 'done.')

    def _plot_subfig(dataset, method):
        figure_filepath = os.path.join(dirpath, f'{dataset}_{method}.pdf')
        if os.path.exists(figure_filepath):
            return

        ablation_samples = {}
        for ablation in ['none', 'causal_ordered', 'markovian', 'triangular']:
            if method in ['dnme', 'tnme'] and ablation == 'triangular':
                continue

            cond = {
                'dataset': dataset,
                'solution_mapping': method,
                'causal_ordered': 1,
                'markovian': 1,
                'triangular': 1,
            }

            if not ablation == 'none':
                cond[ablation] = 0
            curve = select(df, cond)[['obs_wd', 'ctf_rmse']]
            curve = curve[curve['obs_wd'] < 100]
            curve = curve[curve['ctf_rmse'] < 100]
            ablation_samples[ablation] = curve

        # Combine samples and remove outliers
        samples = combine_result(ablation_samples, 'ablation')
        samples = remove_outliers(samples, 'obs_wd')
        samples = remove_outliers(samples, 'ctf_rmse')

        # Draw sub figures
        fig_func = {
            'gpr': plot_gpr,
            'win': plot_sliding_window,
            'poly': plot_poly_regression,
        }[curve_type]
        fig = fig_func(
            samples,
            x_col="obs_wd",
            y_col="ctf_rmse",
            hue_col="ablation",
        )
        plt.tight_layout()
        fig.savefig(figure_filepath)
        plt.clf()
        plt.close()

        print('Figure :', figure_filepath, 'done.')

    # Start drawing
    with warnings.catch_warnings(action="ignore"):
        _plot_legend()
    for dataset in ['barbell', 'stair', 'fork', 'backdoor']:
        for method in ['dnme', 'tnme', 'cmsm', 'tvsm']:
            with warnings.catch_warnings(action="ignore"):
                sns.set_style("whitegrid", {'axes.grid': False})
                _plot_subfig(dataset, method)
