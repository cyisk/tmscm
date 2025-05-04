import os
from typing import Dict, Any

from script.pipeline_manager import PipelineMemory, pipeline_task
from script.tmscm_er.init import get_experiment_config_string


def save_config_file(work_dirpath: str, metadata: Dict[str, Any]):
    # Save config file
    arguments, flag = metadata['arguments'], metadata['flag']
    config_string = get_experiment_config_string(arguments, flag)
    config_filepath = os.path.join(work_dirpath, 'config.yaml')
    with open(config_filepath, 'w+', encoding='utf-8') as f:
        f.write(config_string)


@pipeline_task('config_save')
def config_save(pipeline_memory: PipelineMemory, has_done: bool):
    work_dirpath = pipeline_memory.get('work_dirpath')
    metadata = pipeline_memory.get('metadata')

    save_config_file(work_dirpath, metadata)


@pipeline_task('config_reload')
def config_reload(pipeline_memory: PipelineMemory, has_done: bool):
    # For width search
    pipeline_memory.get(
        'metadata')['config']['model']['solution_mapping']['width'] = 'auto'
    pipeline_memory.get(
        'metadata')['config']['model']['solution_mapping']['max_width'] = 1024
    pipeline_memory.get('metadata')['config']['model']['solution_mapping'][
        'max_params'] = '512K'
