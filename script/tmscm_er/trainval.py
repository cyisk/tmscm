import os
import shutil
import torch as th
from copy import deepcopy
from lightning import seed_everything

os.environ["KEOPS_VERBOSE"] = "0"

from common import *
from dataset.tmscm_er import *
from model.solution_mappings.tvsm import change_ode_method
from script.tmscm_er.common import *
from script.pipeline_manager import PipelineMemory, pipeline_task

th.set_float32_matmul_precision('medium')


def trainval(scm_id: int = 0):

    @pipeline_task(f"trainval[scm_id={scm_id}]")
    def trainval_once(pipeline_memory: PipelineMemory, has_done: bool):
        # Get from memory
        work_dirpath = pipeline_memory.get('work_dirpath')
        config = pipeline_memory.get('metadata')['config']
        run_dirpath = get_run_dirpath(work_dirpath, config['pipeline'], scm_id)
        train_dataloader = pipeline_memory.get('datasets')['train_dataloader']
        val_dataloader = pipeline_memory.get('datasets')['val_dataloader']

        if not has_done:
            # Clear run_dirpath
            if os.path.exists(run_dirpath):
                shutil.rmtree(run_dirpath)

            # Set seed to 0
            seed_everything(seed=0)

            # Get trainer
            scheduler_kwargs = get_full_scheduler_kwargs(
                run_dirpath, config['scheduler'])
            trainer = get_trainer(scheduler_kwargs)

            # Get model
            iota: Iota = train_dataloader.dataset.iota
            model_kwargs = deepcopy(config['model'])
            model = get_model(iota, model_kwargs)

            # Train
            can_try_again = True
            while can_try_again:
                try:
                    can_try_again = False
                    trainer.fit(model, train_dataloader, val_dataloader)
                except AssertionError as e:
                    args = pipeline_memory.get('metadata')['arguments']
                    m = args['solution_mapping']
                    if m == 'tvsm':  # Underflow ODEINT
                        can_try_again = 'underflow in dt' in str(e)

            # Clean cache
            del trainer
            del model
            pipeline_memory.gc()

    return trainval_once
