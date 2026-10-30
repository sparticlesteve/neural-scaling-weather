import os, sys, time
import hydra
import torch
import wandb
import logging
import torch.distributed as dist
from utils.trainer import Trainer


def init_clearml():
    """Attach to the ClearML task when a ClearML agent runs this script.

    This must run before hydra.main: ClearML then applies the task's Hydra/*
    parameters as Hydra overrides. Every rank attaches to the same task; only
    rank 0 reports (see utils/clearml_utils.py).
    """
    if not os.environ.get("CLEARML_TASK_ID"):
        return
    from clearml import Task

    # Checkpoints are registered explicitly, not on every torch.save
    Task.init(auto_connect_frameworks={"pytorch": False})


@hydra.main(version_base=None, config_path="config", config_name="default")
def run(cfg):
    trainer = Trainer(cfg)
    trainer.launch()
    dist.destroy_process_group()
    logging.info("DONE")


if __name__ == "__main__":
    init_clearml()
    run()
