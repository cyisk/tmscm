import gc
import os
import shutil
import torch as th
from copy import deepcopy
from typing import List, Dict, Any


class PipelineMemory():

    def __init__(self):
        self.memory = {}

    def get(self, obj_name: str):
        assert obj_name in self.memory, f"Error: Expect {obj_name} in the pipeline memory."
        return self.memory[obj_name]

    def set(self, obj_name: str, obj: Any):
        self.memory[obj_name] = obj

    def remove(self, obj_name: str):
        assert obj_name in self.memory, f"Error: Expect {obj_name} in the pipeline memory."
        del self.memory[obj_name]

    def gc(self):
        gc.collect()
        th.cuda.empty_cache()


class PipelineTask():

    def __init__(
        self,
        task_func,
        task_name: str,
        verbose: bool = False,
    ):
        self.task_func = task_func
        self.task_name = task_name
        self.verbose = verbose

    def __call__(
        self,
        pipeline_memory: PipelineMemory,
        has_done: bool,
    ) -> Dict[str, Any]:
        if self.verbose:
            print(f"Pipeline: {self.task_name} starts.")
        self.task_func(pipeline_memory, has_done)
        if self.verbose:
            print(f"Pipeline: {self.task_name} ends.")


def pipeline_task(
    task_name: str,
    verbose: bool = False,
):

    def _pipeline_task(task_func):
        return PipelineTask(task_func, task_name, verbose)

    return _pipeline_task


class PipelineManager():

    def __init__(
        self,
        work_dirpath: str,
        initial_metadata: Dict[str, Any],
        clean: bool = False,
        verbose: bool = True,
    ):
        self.work_dirpath = work_dirpath
        self.queue: List[PipelineTask] = []
        self.memory = PipelineMemory()
        self.clean = clean
        self.verbose = verbose

        # Create work directory
        self.memory.set('work_dirpath', work_dirpath)
        if not os.path.exists(work_dirpath):
            os.makedirs(work_dirpath)

        # Initialize or load metadata from disk
        metadata_filepath = os.path.join(work_dirpath, 'metadata')
        if not os.path.exists(metadata_filepath):
            metadata = deepcopy(initial_metadata)
            th.save(metadata, metadata_filepath)
        else:
            metadata = th.load(metadata_filepath, weights_only=True)
        if 'done_tasks' not in metadata:
            metadata['done_tasks'] = []

        # Load metadata into memory
        self.memory.set('metadata', metadata)

    def save_metadata(self):
        # Save metadata from memory to dist
        metadata_filepath = os.path.join(self.work_dirpath, 'metadata')
        metadata = self.memory.get('metadata')
        th.save(metadata, metadata_filepath)

    def get_tasks(self, task_name: str) -> List[PipelineTask]:
        return list(
            filter(
                lambda task: task.task_name == task_name,
                self.queue,
            ))

    def add_task(self, task: PipelineTask):
        self.queue.append(task)

    def remove_tasks(self, task_name: str):
        self.queue = list(
            filter(
                lambda task: not task.task_name == task_name,
                self.queue,
            ))

    def go(self):
        done_tasks = self.memory.get('metadata')['done_tasks']
        assert isinstance(done_tasks, list), \
            "Conflist: progress is ruined, try delete whole work directory."

        def after_task(append_task: bool = True):
            # Update progress and save metadata into disk
            if append_task:
                self.memory.get('metadata')['done_tasks'].append(
                    task.task_name)
            self.save_metadata()

        for i, task in enumerate(self.queue):
            if i < len(done_tasks):
                task_name = done_tasks[i]

            # Task is not done
            else:
                if self.verbose:
                    print(f"Pipeline: {task.task_name} is not done, run.")
                task(self.memory, has_done=False)
                after_task(append_task=True)
                continue

            # Record does not exist, rollback
            if not task.task_name == task_name:
                done_tasks = done_tasks[:i]
                self.memory.get('metadata')['done_tasks'] = done_tasks
                if self.verbose:
                    print(
                        f"Pipeline: task {task.task_name} does not match",
                        f"record {task_name}, rollback and run.",
                    )
                task(self.memory, has_done=False)
                after_task(append_task=True)
                continue

            # Task is done
            if self.verbose:
                print(f"Pipeline: {task.task_name} is done, load and skip.")
            task(self.memory, has_done=True)
            after_task(append_task=False)

        if self.verbose:
            print(f"Pipeline: done.")

        # Clean cache
        self.memory.gc()

        # Clean work directory if needed
        if self.clean:
            if self.verbose:
                print(f"Pipeline: all is done, clean work directory.")
            shutil.rmtree(self.work_dirpath)
