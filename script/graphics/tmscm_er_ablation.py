import os
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tueplots import bundles, fontsizes
from typing import Dict, Any

plt.rcParams.update(bundles.icml2024())
fontsizes.icml2024()

from script.graphics.common import *
from script.pipeline_manager import PipelineMemory, pipeline_task

root = 'experiments/tmscm_er/ablation'


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


@pipeline_task('post_tmscm_er_ablation')
def post_tmscm_er_ablation(pipeline_memory: PipelineMemory, has_done: bool):
    val_df_filepath = 'script/graphics/cache/tmscm_er_ablation_val.csv'
    test_df_filepath = 'script/graphics/cache/tmscm_er_ablation_test.csv'

    # If csv file has been created
    if os.path.exists(val_df_filepath) and os.path.exists(test_df_filepath):
        val_df = pd.read_csv(val_df_filepath)
        test_df = pd.read_csv(test_df_filepath)
        pipeline_memory.set('tmscm_er_ablation_val', val_df)
        pipeline_memory.set('tmscm_er_ablation_test', test_df)
        return

    # Otherwise, create the complete csv
    _, dirs, _ = next(os.walk(root))
    val_df = []
    test_df = []
    for dir in dirs:
        try:
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
        except:
            pass
    val_df = pd.concat(val_df)
    test_df = pd.concat(test_df)
    val_df.to_csv(val_df_filepath)
    test_df.to_csv(test_df_filepath)
    pipeline_memory.set('tmscm_er_ablation_val', val_df)
    pipeline_memory.set('tmscm_er_ablation_test', test_df)


tmpl_main_text_er = """
\\begin{{table}}[t]
\\caption{{Write your description here.}}
\\label{{sample-table}}
\\vskip 0.15in
\\begin{{center}}
\\begin{{small}}
\\begin{{sc}}
\\begin{{tabular}}{{llcccc}}
\\toprule
Method &  & ER-Diag-50 & ER-Tril-50 \\\\
\\midrule
\\multirow{{3}}{{*}}{{DNME}} & -     & {} & {} \\\\
                             & w/o O & {} & {} \\\\
                             & w/o M & {} & {} \\\\
\\midrule
\\multirow{{3}}{{*}}{{TNME}} & -     & {} & {} \\\\
                             & w/o O & {} & {} \\\\
                             & w/o M & {} & {} \\\\
\\midrule
\\multirow{{4}}{{*}}{{CMSM}} & -     & {} & {} \\\\
                             & w/o O & {} & {} \\\\
                             & w/o M & {} & {} \\\\
                             & w/o T & {} & {} \\\\
\\midrule
\\multirow{{4}}{{*}}{{TVSM}} & -     & {} & {} \\\\
                             & w/o O & {} & {} \\\\
                             & w/o M & {} & {} \\\\
                             & w/o T & {} & {} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{sc}}
\\end{{small}}
\\end{{center}}
\\vskip -0.1in
\\end{{table}}
"""


@pipeline_task('tmscm_er_ablation_table_1')
def tmscm_er_ablation_table_1(
    pipeline_memory: PipelineMemory,
    has_done: bool,
):
    table_filepath = 'script/graphics/tables/ablation_main_text_er.tex'
    if os.path.exists(table_filepath):
        return

    df = pipeline_memory.get('tmscm_er_ablation_test')

    values = np.zeros((14, 4))
    row, col = 0, 0
    # Calculate values
    for method in ['dnme', 'tnme', 'cmsm', 'tvsm']:
        for ablation in [None, 'causal_ordered', 'markovian', 'triangular']:
            if method in ['dnme', 'tnme'] and ablation == 'triangular':
                continue
            for dataset in ['diag', 'tril']:
                cond = {
                    'dataset': dataset,
                    'solution_mapping': method,
                    'causal_ordered': 1,
                    'markovian': 1,
                    'triangular': 1,
                }
                if ablation is not None:
                    cond[ablation] = 0
                for metric in ['ctf_rmse']:
                    mean, ci95 = mean_ci95_by(
                        df[df[metric] < 1000],
                        column=metric,
                        conditions=cond,
                    )
                    # Set values
                    values[row, col] = mean
                    values[row, col + 1] = ci95
                    col += 2
                    if col == 4:
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

    tab = tmpl_main_text_er.format(*format_values)
    with open(table_filepath, 'w+', encoding='utf-8') as f:
        f.write(tab)


tmpl_appendix_er = """
\\begin{{table}}[t]
\\caption{{Write your description here.}}
\\label{{sample-table}}
\\vskip 0.15in
\\begin{{center}}
\\begin{{small}}
\\begin{{sc}}
\\begin{{tabular}}{{llcccccccc}}
\\toprule
 &  & \\multicolumn{{3}}{{c}}{{ER-Diag-50}} & \\multicolumn{{3}}{{c}}{{ER-Tril-50}} \\\\
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


@pipeline_task('tmscm_er_ablation_table_2')
def tmscm_er_ablation_table_2(
    pipeline_memory: PipelineMemory,
    has_done: bool,
):
    table_filepath = 'script/graphics/tables/ablation_appendix_er.tex'
    if os.path.exists(table_filepath):
        return

    df = pipeline_memory.get('tmscm_er_ablation_test')

    values = np.zeros((14, 12))
    row, col = 0, 0
    # Calculate values
    for method in ['dnme', 'tnme', 'cmsm', 'tvsm']:
        for ablation in [None, 'causal_ordered', 'markovian', 'triangular']:
            if method in ['dnme', 'tnme'] and ablation == 'triangular':
                continue
            for dataset in ['diag', 'tril']:
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

    tab = tmpl_appendix_er.format(*format_values)
    with open(table_filepath, 'w+', encoding='utf-8') as f:
        f.write(tab)
