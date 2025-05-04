import os

os.environ["KEOPS_VERBOSE"] = "0"

import torch as th
from copy import deepcopy
from lightning import seed_everything

from common import *
from dataset.tmscm_er import *
from script.tmscm_er.common import *
from script.pipeline_manager import PipelineMemory, pipeline_task

th.set_float32_matmul_precision('medium')


def test(scm_id: int = 0):

    @pipeline_task(f"test[scm_id={scm_id}]")
    def test_once(pipeline_memory: PipelineMemory, has_done: bool):
        # Get from memory
        work_dirpath = pipeline_memory.get('work_dirpath')
        config = pipeline_memory.get('metadata')['config']
        run_dirpath = get_run_dirpath(work_dirpath, config['pipeline'], scm_id)
        test_dataloader = pipeline_memory.get('datasets')['test_dataloader']

        if not has_done:
            # Set seed to 0
            seed_everything(seed=0)

            # Get trainer
            scheduler_kwargs = get_full_scheduler_kwargs(
                run_dirpath, config['scheduler'])
            trainer = get_trainer(scheduler_kwargs)

            # Load model from checkpoint
            iota: Iota = test_dataloader.dataset.iota
            model_kwargs = deepcopy(config['model'])
            model = load_model(iota, model_kwargs, run_dirpath)

            # Train
            trainer.test(model, test_dataloader)

            # Clean cache
            del trainer
            del model
            pipeline_memory.gc()

    return test_once
