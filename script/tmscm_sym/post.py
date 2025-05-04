import pandas as pd

from common import *
from script.tmscm_sym.common import *
from script.pipeline_manager import PipelineMemory, pipeline_task


@pipeline_task('test')
def post(pipeline_memory: PipelineMemory, has_done: bool):
    # Get from memory
    work_dirpath = pipeline_memory.get('work_dirpath')
    config = pipeline_memory.get('metadata')['config']

    # Results
    val_results = {}
    test_results = {}

    if not has_done:
        # Load results
        for seed in config['pipeline']['run_seeds']:
            run_dirpath = get_run_dirpath(
                work_dirpath=work_dirpath,
                pipeline_kwargs=config['pipeline'],
                seed=seed,
            )
            val_result = load_result(
                dirpath=run_dirpath,
                filename=config['scheduler']['val_scheduler']['filename'],
            )
            test_result = load_result(
                dirpath=run_dirpath,
                filename=config['scheduler']['test_scheduler']['filename'],
            )
            val_results[seed] = val_result
            test_results[seed] = test_result

        # Combine results
        combined_val_result: pd.DataFrame = combine_result(
            val_results,
            identifier='seed',
            format=config['scheduler']['val_scheduler']['format'],
        )
        combined_test_result: pd.DataFrame = combine_result(
            test_results,
            identifier='seed',
            format=config['scheduler']['test_scheduler']['format'],
        )

        # Save combined results
        combined_val_filepath = os.path.join(
            work_dirpath, config['scheduler']['val_scheduler']['filename'])
        combined_val_result.to_csv(combined_val_filepath)
        combined_test_filepath = os.path.join(
            work_dirpath, config['scheduler']['test_scheduler']['filename'])
        combined_test_result.to_csv(combined_test_filepath)
