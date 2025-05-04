from script.pipeline_manager import PipelineManager

from script.tmscm_sym.init import initialize_experiment_metadata
from script.tmscm_sym.config_save import config_save
from script.tmscm_sym.dataset import load_dataset
from script.tmscm_sym.width_search import width_search
from script.tmscm_sym.trainval import trainval
from script.tmscm_sym.test import test
from script.tmscm_sym.post import post

# Initialize metadata
metadata = initialize_experiment_metadata()
work_dirpath = metadata['work_dirpath']
seeds = metadata['config']['pipeline']['run_seeds']
print('\n' + metadata['flag'])

# Initialize pipeline
pipeline = PipelineManager(
    work_dirpath=work_dirpath,
    initial_metadata=metadata,
    clean=False,
    verbose=True,
)

# Setting up pipeline
pipeline.add_task(config_save)
pipeline.add_task(load_dataset)
pipeline.add_task(width_search)
for seed in seeds:
    pipeline.add_task(trainval(seed))
    pipeline.add_task(test(seed))
pipeline.add_task(post)

# Go
pipeline.go()
