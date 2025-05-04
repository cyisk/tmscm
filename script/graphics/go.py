from script.pipeline_manager import PipelineManager

from script.graphics.tmscm_sym_ablation import post_tmscm_sym_ablation, tmscm_sym_ablation_table_1, tmscm_sym_ablation_table_2, tmscm_sym_convergence_figure
from script.graphics.tmscm_sym_exogenous import post_tmscm_sym_exogenous, tmscm_sym_exogenous_table
from script.graphics.tmscm_er_ablation import post_tmscm_er_ablation, tmscm_er_ablation_table_1, tmscm_er_ablation_table_2

# Initialize pipeline
work_dirpath = 'script/graphics/cache'
pipeline = PipelineManager(
    work_dirpath=work_dirpath,
    initial_metadata={},
    clean=False,
    verbose=True,
)

# Setting up pipeline
pipeline.add_task(post_tmscm_sym_ablation)
pipeline.add_task(tmscm_sym_ablation_table_1)
pipeline.add_task(tmscm_sym_ablation_table_2)
pipeline.add_task(tmscm_sym_convergence_figure)
pipeline.add_task(post_tmscm_sym_exogenous)
pipeline.add_task(tmscm_sym_exogenous_table)
pipeline.add_task(post_tmscm_er_ablation)
pipeline.add_task(tmscm_er_ablation_table_1)
pipeline.add_task(tmscm_er_ablation_table_2)

# Go
pipeline.go()
