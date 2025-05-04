from torch.utils.data import DataLoader
from typing import Dict, Any

from dataset.tmscm_er import *
from script.pipeline_manager import PipelineMemory, pipeline_task


def get_datasets(
    dataset_kwargs: Dict[str, Any],
    scm_id: int,
) -> Dict[str, Any]:
    dataset_name = dataset_kwargs['name']

    # Datasets
    train_dataset, val_dataset, test_dataset = {
        'diag': tmscm_er_diag_i,
        'tril': tmscm_er_tril_i,
    }[dataset_name](scm_id=scm_id)

    # Train dataloader
    train_dataloader_kwargs = dataset_kwargs['train']
    if train_dataloader_kwargs['batch_size'] == -1:
        train_dataloader_kwargs['batch_size'] = len(train_dataset)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        **train_dataloader_kwargs,
    )

    # Validation dataloader
    val_dataloader_kwargs = dataset_kwargs['val']
    if val_dataloader_kwargs['batch_size'] == -1:
        val_dataloader_kwargs['batch_size'] = len(val_dataset)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        **val_dataloader_kwargs,
    )

    # Test dataloader
    test_dataloader_kwargs = dataset_kwargs['test']
    if test_dataloader_kwargs['batch_size'] == -1:
        test_dataloader_kwargs['batch_size'] = len(test_dataset)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        **test_dataloader_kwargs,
    )

    return {
        'train_dataset': train_dataset,
        'train_dataloader': train_dataloader,
        'val_dataset': val_dataset,
        'val_dataloader': val_dataloader,
        'test_dataset': test_dataset,
        'test_dataloader': test_dataloader,
    }


def dataset(scm_id: int = 0):

    @pipeline_task(f"load_dataset[scm_id={scm_id}]")
    def load_dataset_once(pipeline_memory: PipelineMemory, has_done: bool):
        # Load datasets and initialize dataloaders according to config
        config = pipeline_memory.get('metadata')['config']
        datasets = get_datasets(config['dataset'], scm_id)

        # Set datasets into pipeline memory
        pipeline_memory.set('datasets', datasets)

    return load_dataset_once
