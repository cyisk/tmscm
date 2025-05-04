from dataset.tmscm_er import min_scm_id, max_scm_id
from script.pipeline_manager import PipelineManager

from script.tmscm_er.init import initialize_experiment_metadata
from script.tmscm_er.config_save import config_save, config_reload
from script.tmscm_er.dataset import dataset
from script.tmscm_er.width_search import width_search
from script.tmscm_er.trainval import trainval
from script.tmscm_er.test import test
from script.tmscm_er.post import post

# Initialize metadata
metadata = initialize_experiment_metadata()
work_dirpath = metadata['work_dirpath']
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
for scm_id in range(min_scm_id, max_scm_id + 1):
    pipeline.add_task(dataset(scm_id))
    pipeline.add_task(width_search)
    pipeline.add_task(trainval(scm_id))
    pipeline.add_task(test(scm_id))
    pipeline.add_task(config_reload)
pipeline.add_task(post)

# Go
pipeline.go()
