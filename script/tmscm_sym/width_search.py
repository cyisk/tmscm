import gc
import torch as th
from copy import deepcopy
from torch.utils.data import Dataset
from typing import Dict, Any

from model.proxy_scm import LightningProxySCM
from script.tmscm_sym.common import *
from script.pipeline_manager import PipelineMemory, pipeline_task


def integer_from_string(int_string: str):
    suffixes = {
        'K': 1_000,
        'M': 1_000_000,
        'B': 1_000_000_000,
        'T': 1_000_000_000_000,
    }
    if int_string.isdigit() or 'e' in int_string.lower():
        return int(float(int_string))
    for suffix, multiplier in suffixes.items():
        if int_string.upper().endswith(suffix):
            num_part = int_string[:-len(suffix)]
            return int(float(num_part) * multiplier)


def search_and_update_width(
    iota: Iota,
    model_kwargs: Dict[str, Any],
) -> int:

    def get_width_fixed_model(width: int) -> LightningProxySCM:
        # Try to instantiate a new model with fixed width
        try:
            model_kwargs_ = deepcopy(model_kwargs)
            model_kwargs_['solution_mapping']['width'] = width
            del model_kwargs_['solution_mapping']['max_params']
            del model_kwargs_['solution_mapping']['max_width']
            return get_model(iota, model_kwargs_)
        except:
            return None

    # Get approperaite width from binary search
    left, right = 1, model_kwargs['solution_mapping']['max_width']
    threshold = model_kwargs['solution_mapping']['max_params']
    threshold = integer_from_string(threshold)
    result = None

    # Binary search
    while left <= right:
        mid = (left + right) // 2

        # Instantiate a new model to calculate total parameters
        model = get_width_fixed_model(mid)
        if model:
            module = model.proxy_scm.solution_mapping
            total_params = sum(p.numel() for p in module.parameters())
        else:
            total_params = threshold + 1  # Maybe OOM

        # Binary update
        if total_params <= threshold:
            result = mid
            left = mid + 1
        else:
            right = mid - 1

        # Collect cache to prevent OOM
        del module
        gc.collect()
        th.cuda.empty_cache()

    return result


@pipeline_task('width_search')
def width_search(pipeline_memory: PipelineMemory, has_done: bool):
    if not has_done:
        config = pipeline_memory.get('metadata')['config']
        dataset = pipeline_memory.get('datasets')['train_dataset']

        # Search an appropriate width
        iota = dataset.iota
        width = search_and_update_width(iota, config['model'])

        # Update a fixed width in config in memory
        config['model']['solution_mapping']['width'] = width
        del config['model']['solution_mapping']['max_params']
        del config['model']['solution_mapping']['max_width']
