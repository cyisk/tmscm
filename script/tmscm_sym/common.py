import os
from copy import deepcopy
from lightning import Trainer
from typing import Dict, Any

from common import *
from dataset.tmscm_sym import *
from model.proxy_scm import LightningProxySCM


def get_run_dirpath(
    work_dirpath: str,
    pipeline_kwargs: Dict[str, Any],
    seed: int = 0,
):
    # Get work dirpath for this run
    run_subdirpath = pipeline_kwargs['run_subdirpath']
    run_subdirpath = run_subdirpath.replace(r'${seed}', str(seed))
    run_dirpath = os.path.join(work_dirpath, run_subdirpath)
    return run_dirpath


def get_full_scheduler_kwargs(
    run_dirpath: str,
    scheduler_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    # Get scheduler kwargs with dirpath
    scheduler_kwargs_ = deepcopy(scheduler_kwargs)
    scheduler_kwargs_['val_scheduler']['dirpath'] = run_dirpath
    scheduler_kwargs_['checkpoint_scheduler']['dirpath'] = run_dirpath
    scheduler_kwargs_['test_scheduler']['dirpath'] = run_dirpath
    return scheduler_kwargs_


def get_trainer(scheduler_kwargs: Dict[str, Any]) -> Trainer:
    trainer_kwargs = scheduler_kwargs['trainer']
    val_scheduler_kwargs = scheduler_kwargs['val_scheduler']
    checkpoint_scheduler_kwargs = scheduler_kwargs['checkpoint_scheduler']
    test_scheduler_kwargs = scheduler_kwargs['test_scheduler']

    # Validation scheduler
    val_scheduler = ValidationScheduler(**val_scheduler_kwargs)

    # Checkpoint scheduler
    checkpoint_scheduler = CheckpointScheduler(**checkpoint_scheduler_kwargs)

    # Test scheduler
    test_scheduler = TestScheduler(**test_scheduler_kwargs)

    # Trainer
    return Trainer(
        callbacks=[
            val_scheduler,
            checkpoint_scheduler,
            test_scheduler,
        ],
        **trainer_kwargs,
    )


def get_model(iota: Iota, model_kwargs: Dict[str, Any]) -> LightningProxySCM:
    # Model arguments
    exogenous_distribution_kwargs = model_kwargs['exogenous_distribution']
    solution_mapping_kwargs = model_kwargs['solution_mapping']
    optimizer_kwargs = model_kwargs['optimizer']

    # Model
    return LightningProxySCM(
        iota=iota,
        exogenous_distribution_kwargs=exogenous_distribution_kwargs,
        solution_mapping_kwargs=solution_mapping_kwargs,
        optimizer_kwargs=optimizer_kwargs,
    )


def load_model(
    iota: Iota,
    model_kwargs: Dict[str, Any],
    run_dirpath: str,
) -> LightningProxySCM:
    # Find checkpoint file
    files = os.listdir(run_dirpath)
    ckpt_files = [file for file in files if file.endswith(".ckpt")]
    assert len(ckpt_files) > 0, \
        f"Error: Checkpoint file not found in {run_dirpath}."
    ckpt_filepath = os.path.join(run_dirpath, ckpt_files[0])

    # Model arguments
    exogenous_distribution_kwargs = model_kwargs['exogenous_distribution']
    solution_mapping_kwargs = model_kwargs['solution_mapping']
    optimizer_kwargs = model_kwargs['optimizer']

    # Load model
    return LightningProxySCM.load_from_checkpoint(
        checkpoint_path=ckpt_filepath,
        iota=iota,
        exogenous_distribution_kwargs=exogenous_distribution_kwargs,
        solution_mapping_kwargs=solution_mapping_kwargs,
        optimizer_kwargs=optimizer_kwargs,
    )
