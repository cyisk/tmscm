from torch.utils.data import DataLoader
from typing import Dict, Any

from dataset.tmscm_sym import *
from script.pipeline_manager import PipelineMemory, pipeline_task


def get_datasets(dataset_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    dataset_name = dataset_kwargs['name']

    # Train dataset and dataloader
    train_dataset = {
        'barbell': barbell_train,
        'stair': stair_train,
        'fork': fork_train,
        'backdoor': backdoor_train,
    }[dataset_name]
    train_dataloader_kwargs = dataset_kwargs['train']
    if train_dataloader_kwargs['batch_size'] == -1:
        train_dataloader_kwargs['batch_size'] = len(train_dataset)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        **train_dataloader_kwargs,
    )

    # Validation dataset and dataloader
    val_dataset = {
        'barbell': barbell_val,
        'stair': stair_val,
        'fork': fork_val,
        'backdoor': backdoor_val,
    }[dataset_name]
    val_dataloader_kwargs = dataset_kwargs['val']
    if val_dataloader_kwargs['batch_size'] == -1:
        val_dataloader_kwargs['batch_size'] = len(val_dataset)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        **val_dataloader_kwargs,
    )

    # Test dataset and dataloader
    test_dataset = {
        'barbell': barbell_test,
        'stair': stair_test,
        'fork': fork_test,
        'backdoor': backdoor_test,
    }[dataset_name]
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


@pipeline_task('load_dataset')
def load_dataset(pipeline_memory: PipelineMemory, has_done: bool):
    # Load datasets and initialize dataloaders according to config
    config = pipeline_memory.get('metadata')['config']
    datasets = get_datasets(config['dataset'])

    # Set datasets into pipeline memory
    pipeline_memory.set('datasets', datasets)
