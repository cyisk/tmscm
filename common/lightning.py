import os
import torch as th
import pandas as pd
from copy import deepcopy
from lightning import Trainer, LightningModule, Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from typing import List, Dict, Any, Callable

TensorDict = Dict[str, th.Tensor]
ListResult = List[Dict[str, Any]]
DictResult = Dict[str, List[Any]]
ValidationListResult = ListResult
ValidationSaveCallback = Callable[[ValidationListResult], ValidationListResult]
TestListResult = ListResult
TestSaveCallback = Callable[[TestListResult], TestListResult]


def tensor_dict_to_number(tensor_dict: TensorDict):
    """Make tensors in a tensor dictionary saveable.

    Args:
        `tensor_dict` (Dict[str, Tensor]): A tensor dictionary.

    Returns:
        Dict (Dict[str, Tensor | float]): The same tensor dictionary, with saveable tensors.
    """
    tensor_dict = deepcopy(tensor_dict)
    for key in tensor_dict:
        value = tensor_dict[key]
        if isinstance(value, th.Tensor):
            value = value.cpu().clone().detach()  # to cpu
            if value.dim() == 0:  # number
                value = value.item()
        tensor_dict[key] = value
    return tensor_dict


def save_result(
    result: ValidationListResult,
    dirpath: str,
    filename: str,
    format: str = "csv",
):
    """Save validation results to disk.

    Args:
        `result` (Dict[str, Tensor]): The validation results.
        `dirpath` (str): The dirpath of the saved file.
        `firname` (str): The filename of the saved file.
        `format` (str): 'csv', 'dict' or 'list'.
    """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    dict_result = {k: [v[k] for v in result] for k in result[0]}
    filepath = os.path.join(dirpath, filename)
    if format == "dict":
        th.save(dict_result, filepath)
    elif format == "list":
        th.save(result, filepath)
    elif format == "csv":
        df = pd.DataFrame(dict_result)
        df.to_csv(filepath)


def load_result(
    dirpath: str,
    filename: str,
    format: str = "csv",
):
    """Save validation results from disk.

    Args:
        `dirpath` (str): The dirpath of the saved file.
        `firname` (str): The filename of the saved file.
        `format` (str): 'csv', 'dict' or 'list'.

    Returns:
        Dict[str, Tensor]: The validation results.
    """
    filepath = os.path.join(dirpath, filename)
    if format == "dict":
        return th.load(filepath, weights_only=True)
    elif format == "list":
        return th.load(filepath, weights_only=True)
    elif format == "csv":
        return pd.read_csv(filepath)


def combine_result(
    results: Dict[str, List[ListResult | DictResult | pd.DataFrame]],
    identifier: str = "id",
    format: str = "csv",
):
    """Combine a dictionary of validation results into one.

    Args:
        `results` (Dict[str, List[ListResult | DictResult | pd.DataFrame]]): A dictionary of validation results. The keys will be used as identifiers.
        `identifier` (str): The new identifier name.
        `format` (str): 'csv', 'dict' or 'list'.

    Returns:
        ListResult | DictResult | pd.DataFrame: The combined validation results.

    ---
    Let `identifier=id`. For `format=dict`:
    ```python
    results = {'key1': {'a': [...], 'b': [...]},
        'key2': {'a': [...], 'b': [...]}}
    => combined = {'id': ['key1', ..., 'key2', ....], 'a': [...], 'b': [...]}
    ```

    For `format=list`:
    ```python
    results = {'key1': [{'a': ..., 'b': ...}, ...],
        'key2': [{'a': ..., 'b': ...}, ...]}
    => combined = [{'id': 'key1', 'a': ..., 'b': ...}, ...]
    ```

    For `format=csv`, a new column named `id` will be added.
    """
    if format == "dict":
        first = results[next(iter(results))]
        combined = {key: [] for key in first}
        for key in first:
            combined[key] = [
                results[id][key][i] for i in range(len(first))
                for id in results
            ]
        ids = [id for i in range(len(first)) for id in results]
        combined[identifier] = ids
        return combined
    elif format == "list":
        first = results[next(iter(results))]
        combined = []
        for id in results:
            for row in results[id]:
                row[identifier] = id
                combined.append(row)
    else:
        dfs = []
        for id in results:
            df = results[id].copy()
            df[identifier] = id
            dfs.append(df)
        return pd.concat(dfs)


class ValidationScheduler(Callback):
    """
    The Validation Scheduler.
    """

    def __init__(
        self,
        interval: int = 1,
        interval_unit: str = 'epoch',
        interval_mul: float = 1.0,
        max_interval: int = 1,
        max_interval_unit: int = 'epoch',
        initial: bool = True,
        dirpath: str = None,
        filename: str = None,
        format: str = "csv",
        on_before_save_result: ValidationSaveCallback = None,
    ):
        """A lightning callback that controls the validation schedule.

        Args:
            `interval` (int): The interval of validation.
            `interval_unit` (str): 'step' or 'epoch'.
            `interval_mul` (float): Exponential interval increasing or decreasing.
            `max_interval` (int): The maximum interval of validation.
            `max_interval_unit` (str): 'step' or 'epoch'.
            `initial` (bool): Validate before training.
            `dirpath` (str, optional): The dirpath of the saved validation results. If not given, then no file saved.
            `firname` (str, optional): The filename of the saved validation results. If not given, then no file saved.
            `format` (str): 'csv', 'dict' or 'list'.
            `on_before_save_result` (ValidationSaveCallback): A callback function before saving validation results, allowing validation results as input, and output modified validation results.
        """
        super().__init__()

        # Parameter list
        self.interval = interval
        self.interval_unit = interval_unit
        self.interval_mul = float(interval_mul)
        self.max_interval = max_interval
        self.max_interval_unit = max_interval_unit
        self.initial = initial
        self.dirpath = dirpath
        self.filename = filename
        self.format = format
        self.on_before_save_result = on_before_save_result

        # Temporarily store the original property
        self._previous_limit_val_batches = 1.0

        # Validation results and counters
        self._val_result = []
        self._current_interval = float(interval)
        self._current_steps = 0
        self._current_epochs = 0

    def can_validate(
        self,
        stage: str = 'step_end',
    ) -> bool:
        # Check if validation is available at current step and update counters
        if stage == 'step_end':
            if self.interval_unit == 'step' and self._current_steps >= self._current_interval:
                self._current_steps = self._current_epochs = 0
                self._current_interval *= self.interval_mul
                return True
            if self.max_interval_unit == 'step' and self._current_steps >= self.max_interval:
                self._current_steps = self._current_epochs = 0
                return True

        # Check if validation is available at current epoch and update counters
        elif stage == 'epoch_end':
            if self.interval_unit == 'epoch' and self._current_epochs >= self._current_interval:
                self._current_epochs = self._current_steps = 0
                self._current_interval *= self.interval_mul
                return True
            if self.max_interval_unit == 'epoch' and self._current_epochs >= self.max_interval:
                self._current_epochs = self._current_steps = 0
                return True

        # validation is not available
        return False

    def validate_once(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ):
        # Current epoch and step
        epoch = pl_module.current_epoch
        step = pl_module.global_step

        # Enable validation and validate
        trainer.limit_val_batches = 1.0
        val_loop = trainer.fit_loop.epoch_loop.val_loop
        first_loop_iter = trainer._logger_connector._first_loop_iter
        trainer._logger_connector._first_loop_iter = False
        val_result_rows = val_loop.run()
        trainer._logger_connector._first_loop_iter = first_loop_iter
        trainer.limit_val_batches = 0.0

        # Add auxillary information; value conversion
        val_result_rows_new = []
        for batch_idx, row in enumerate(val_result_rows):
            row = tensor_dict_to_number(row)
            row['epoch'] = epoch
            row['step'] = step
            row['batch'] = batch_idx
            val_result_rows_new.append(row)

        self._val_result.extend(val_result_rows_new)

    def on_fit_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        *args,
        **kwargs,
    ):
        # Disable automatic validation
        self._previous_limit_val_batches = trainer.limit_val_batches
        trainer.val_check_interval = 0
        trainer.limit_val_batches = 0.0

        # Initial validation
        if self.initial:
            self.validate_once(trainer, pl_module)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        *args,
        **kwargs,
    ):
        # Update counter
        self._current_steps += 1

        # Validate if available
        if self.can_validate(stage='step_end'):
            self.validate_once(trainer, pl_module)

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        *args,
        **kwargs,
    ):
        # Update counter
        self._current_epochs += 1

        # Validate if available
        if self.can_validate(stage='epoch_end'):
            self.validate_once(trainer, pl_module)

    def on_fit_end(
        self,
        trainer: Trainer,
        *args,
        **kwargs,
    ):
        # Save result if filename is given
        if self.dirpath is not None and self.filename is not None:
            if self.on_before_save_result is not None:
                self._val_result = self.on_before_save_result(self._val_result)
            save_result(
                result=self._val_result,
                dirpath=self.dirpath,
                filename=self.filename,
                format=self.format,
            )

        # Reset
        trainer.limit_val_batches = self._previous_limit_val_batches
        self._val_result = []
        self._current_interval = self.interval
        self._current_steps = 0
        self._current_epochs = 0


class CheckpointScheduler(ModelCheckpoint):
    """
    The Checkpoint Scheduler, see `lightning.pytorch.callbacks.ModelCheckpoint`
    """
    pass


class TestScheduler(Callback):
    """
    The Test Scheduler.
    """

    def __init__(
        self,
        dirpath: str = None,
        filename: str = None,
        format: str = "csv",
        on_before_save_result: ValidationSaveCallback = None,
    ):
        """A lightning callback that controls the test schedule.

        Args:
            `dirpath` (str, optional): The dirpath of the saved validation results. If not given, then no file saved.
            `firname` (str, optional): The filename of the saved validation results. If not given, then no file saved.
            `format` (str): 'csv', 'dict' or 'list'.
            `on_before_save_result` (ValidationSaveCallback): A callback function before saving test results, allowing test results as input, and output modified test results.
        """
        super().__init__()

        # Parameter list
        self.dirpath = dirpath
        self.filename = filename
        self.format = format
        self.on_before_save_result = on_before_save_result

    def on_test_epoch_end(
        self,
        trainer: Trainer,
        *args,
        **kwargs,
    ):
        test_result = [tensor_dict_to_number(trainer.callback_metrics)]

        # Save result if filename is given
        if self.dirpath is not None and self.filename is not None:
            if self.on_before_save_result is not None:
                test_result = self.on_before_save_result(test_result)
            save_result(
                result=test_result,
                dirpath=self.dirpath,
                filename=self.filename,
                format=self.format,
            )
