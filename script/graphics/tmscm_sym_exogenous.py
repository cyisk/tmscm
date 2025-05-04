import os
import torch as th
import matplotlib.pyplot as plt
import pandas as pd
from tueplots import bundles, fontsizes
from typing import Dict, Any

plt.rcParams.update(bundles.icml2024())
fontsizes.icml2024()

from script.graphics.common import *
from script.pipeline_manager import PipelineMemory, pipeline_task

root = 'experiments/tmscm_sym/exogenous'


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


@pipeline_task('post_tmscm_sym_exogenous')
def post_tmscm_sym_exogenous(pipeline_memory: PipelineMemory, has_done: bool):
    val_df_filepath = 'script/graphics/cache/tmscm_sym_exogenous_val.csv'
    test_df_filepath = 'script/graphics/cache/tmscm_sym_exogenous_test.csv'

    # If csv file has been created
    if os.path.exists(val_df_filepath) and os.path.exists(test_df_filepath):
        val_df = pd.read_csv(val_df_filepath)
        test_df = pd.read_csv(test_df_filepath)
        pipeline_memory.set('tmscm_sym_exogenous_val', val_df)
        pipeline_memory.set('tmscm_sym_exogenous_test', test_df)
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
    pipeline_memory.set('tmscm_sym_exogenous_val', val_df)
    pipeline_memory.set('tmscm_sym_exogenous_test', test_df)


tmpl_appendix_sym = """
\\begin{{table}}[t]
\\caption{{Write your description here.}}
\\label{{sample-table}}
\\vskip 0.15in
\\begin{{center}}
\\begin{{small}}
\\begin{{sc}}
\\begin{{tabular}}{{llcccccccccc}}
\\toprule
 &  & \\multicolumn{{2}}{{c}}{{Barbell}} & \\multicolumn{{2}}{{c}}{{Stair}} & \\multicolumn{{2}}{{c}}{{Fork}} & \\multicolumn{{2}}{{c}}{{Backdoor}} \\\\
      \\cmidrule(lr){{3-4}}                \\cmidrule(lr){{5-6}}              \\cmidrule(lr){{7-8}}             \\cmidrule(lr){{9-10}}
Method & Dist & Obs$_{{\\text{{WD}}}}$ & Ctf$_{{\\text{{RMSE}}}}$ & Obs$_{{\\text{{WD}}}}$ & Ctf$_{{\\text{{RMSE}}}}$ & Obs$_{{\\text{{WD}}}}$ & Ctf$_{{\\text{{RMSE}}}}$ & Obs$_{{\\text{{WD}}}}$ & Ctf$_{{\\text{{RMSE}}}}$ \\\\
\\midrule
\\multirow{{3}}{{*}}{{DNME}} & N   & {} & {} & {} & {} & {} & {} & {} & {} \\\\
                             & GMM & {} & {} & {} & {} & {} & {} & {} & {} \\\\
                             & NF  & {} & {} & {} & {} & {} & {} & {} & {} \\\\
\\midrule
\\multirow{{3}}{{*}}{{TNME}} & N   & {} & {} & {} & {} & {} & {} & {} & {} \\\\
                             & GMM & {} & {} & {} & {} & {} & {} & {} & {} \\\\
                             & NF  & {} & {} & {} & {} & {} & {} & {} & {} \\\\
\\midrule
\\multirow{{3}}{{*}}{{CMSM}} & N   & {} & {} & {} & {} & {} & {} & {} & {} \\\\
                             & GMM & {} & {} & {} & {} & {} & {} & {} & {} \\\\
                             & NF  & {} & {} & {} & {} & {} & {} & {} & {} \\\\
\\midrule
\\multirow{{3}}{{*}}{{TVSM}} & N   & {} & {} & {} & {} & {} & {} & {} & {} \\\\
                             & GMM & {} & {} & {} & {} & {} & {} & {} & {} \\\\
                             & NF  & {} & {} & {} & {} & {} & {} & {} & {} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{sc}}
\\end{{small}}
\\end{{center}}
\\vskip -0.1in
\\end{{table}}
"""


@pipeline_task('tmscm_sym_exogenous_table')
def tmscm_sym_exogenous_table(
    pipeline_memory: PipelineMemory,
    has_done: bool,
):
    table_filepath = 'script/graphics/tables/exogenous_appendix_sym.tex'
    if os.path.exists(table_filepath):
        return

    df = pipeline_memory.get('tmscm_sym_exogenous_test')

    values = []
    for method in ['dnme', 'tnme', 'cmsm', 'tvsm']:
        for exogenous in ['n', 'gmm', 'nf']:
            for dataset in ['barbell', 'stair', 'fork', 'backdoor']:
                cond = {
                    'dataset': dataset,
                    'solution_mapping': method,
                    'exogenous_distribution': exogenous,
                    'causal_ordered': 1,
                    'markovian': 1,
                    'triangular': 1,
                }
                for metric in ['obs_wd', 'ctf_rmse']:
                    mean, ci95 = mean_ci95_by(
                        df[df[metric] < 1000],
                        column=metric,
                        conditions=cond,
                    )
                    val = format_mean_ci95(mean, ci95)
                    values.append(val)

    tab = tmpl_appendix_sym.format(*values)
    with open(table_filepath, 'w+', encoding='utf-8') as f:
        f.write(tab)
