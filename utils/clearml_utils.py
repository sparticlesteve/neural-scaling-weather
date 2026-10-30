"""ClearML reporting for the trainer: scalars, summary values and checkpoint models.

Active only on world rank 0 of a process running inside a ClearML task
(train.py attaches to the task when a ClearML agent launches it).
"""

import logging
import os

try:
    from clearml import InputModel, OutputModel, Task
except ImportError:
    Task = None


def get_clearml_reporter(cfg, world_rank):
    """Return a ClearMLReporter, or None when ClearML is off or there is no task"""
    if world_rank != 0 or Task is None or not cfg.logging.get("log_to_clearml", True):
        return None
    task = Task.current_task()
    if task is None:
        return None
    logging.info(f"Reporting to ClearML task {task.id}")
    return ClearMLReporter(task, cfg.run_tag)


def file_uri(path):
    return "file://" + os.path.abspath(path)


class ClearMLReporter:
    def __init__(self, task, run_tag):
        self.task = task
        self.logger = task.get_logger()
        self.run_tag = run_tag

    def report_scalars(self, logs, iteration):
        """Report one row of training logs, grouping related series into one plot"""
        for key, value in logs.items():
            value = float(value.item() if hasattr(value, "item") else value)
            if key in ("train_loss", "val_loss", "best_val_loss"):
                title, series = "loss", key
            elif key.startswith("val_rmse_"):
                title, series = "val_rmse", key[len("val_rmse_") :]
            elif key.endswith("_time"):
                title, series = "time", key
            else:
                title, series = key, key
            self.logger.report_scalar(title, series, value=value, iteration=iteration)

    def report_summary(self, values):
        """Report single values (final loss, FLOPs, ...) shown in the task's summary table"""
        for name, value in values.items():
            self.logger.report_single_value(name, float(value))

    def register_checkpoint(self, path, iteration, metadata=None):
        """Register a checkpoint file as an output model without uploading it.

        With tensor or spatial parallelism the file is the logical path that
        restore_checkpoint expands into per-rank shards.
        """
        model = OutputModel(
            task=self.task,
            name=f"{self.run_tag} iter{iteration}",
            framework="PyTorch",
            tags=[f"iter{iteration}"],
        )
        model.update_weights(register_uri=file_uri(path), iteration=iteration, auto_delete_file=False)
        model.set_metadata("iteration", str(iteration), "int")
        for key, value in (metadata or {}).items():
            model.set_metadata(key, str(value))

    def connect_input_checkpoint(self, path):
        """Record the checkpoint a run branches from, reusing its registered model if any"""
        model = InputModel.import_model(weights_url=file_uri(path), name=os.path.basename(path), framework="PyTorch")
        self.task.connect(model, name="branch_from")
