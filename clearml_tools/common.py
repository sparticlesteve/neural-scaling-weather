"""Shared settings and helpers for running this repo's scripts as ClearML tasks.

Tasks run on the Muller Slurm agent ("muller" queue): every Slurm rank runs
the script in the container image, and paths under /mscratch are visible
inside it. Data and experiment dirs default to the dasml scratch there.
"""

from clearml import Task

REPO = "https://github.com/sparticlesteve/neural-scaling-weather.git"
BRANCH = "clearml-scaling"
IMAGE = "docker.io/nersc/pytorch:26.07.01"
PROJECT = "SC26 demo-dev/Scaling laws"
QUEUE = "muller"
TAGS = ["sc26-test", "scaling"]

SCRATCH = "/mscratch/sd/d/dasml/sc26-scaling"
DATA_ROOT = f"{SCRATCH}/data"
EXP_ROOT = f"{SCRATCH}/expts"
SYNTHETIC_DATA = "latlon_1deg_synthetic_6h"

# Pin clearml: the task code was tested with this version
PACKAGES = ["clearml==2.0.2"]

# Hydra overrides that point training at the synthetic test data
SYNTHETIC_DATA_OVERRIDES = {
    "data.root": DATA_ROOT,
    "data.name": SYNTHETIC_DATA,
    "data.dt_scale": 1,  # the synthetic data is already 6-hourly
    "data.train_start_year": 2016,
    "data.valid_years": "[2017]",
    "data.test_years": "[2018]",
    "exp_root": EXP_ROOT,
    "logging.log_to_wandb": False,
}

# The template exports Slurm's rank variables; the image's NCCL net plugin
# setting does not apply inside podman-hpc (submit_batch.sh unsets it too).
SETUP_SCRIPT = """
unset NCCL_NET_PLUGIN
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
"""


def hydra_params(overrides):
    """Map {key: value} Hydra overrides to ClearML's Hydra/* parameter section"""
    return {f"Hydra/{key}": str(value).lower() if isinstance(value, bool) else str(value) for key, value in overrides.items()}


def parse_overrides(args):
    """Parse key=value strings into a dict"""
    overrides = {}
    for arg in args:
        key, sep, value = arg.partition("=")
        if not sep:
            raise ValueError(f"Expected key=value, got {arg!r}")
        overrides[key] = value
    return overrides


def micro_batch_without_accum(batch_size, nodes, tp=1, sp1=1, sp2=1, gpus_per_node=4):
    """Micro-batch size that makes gradient accumulation 1 (one micro-batch per DP rank per step).

    Grad accum > 1 currently misaligns DALI and trainer epochs (alternating short epochs).
    """
    dp = nodes * gpus_per_node // (int(tp) * int(sp1) * int(sp2))
    if batch_size % dp:
        raise ValueError(f"batch size {batch_size} is not divisible by DP size {dp}")
    return batch_size // dp


def set_slurm_resources(task, nodes, ntasks_per_node=4, cpus_per_task=32):
    """User properties read by the Slurm template"""
    task.set_user_properties(num_nodes=nodes, ntasks_per_node=ntasks_per_node, cpus_per_task=cpus_per_task)


def create_train_task(name, overrides, nodes=1, project=PROJECT, tags=(), branch=BRANCH, image=IMAGE):
    """Create a draft task that runs train.py with the given Hydra overrides"""
    task = Task.create(
        project_name=project,
        task_name=name,
        task_type="training",
        repo=REPO,
        branch=branch,
        script="train.py",
        docker=image,
        docker_bash_setup_script=SETUP_SCRIPT,
        packages=PACKAGES,
        add_task_init_call=False,  # train.py attaches to the task itself
    )
    task.set_parameters(hydra_params(overrides))
    set_slurm_resources(task, nodes)
    task.add_tags(list(TAGS) + list(tags))
    return task


def create_script_task(
    name, script, args, project=PROJECT, tags=(), branch=BRANCH, image=IMAGE, task_type="data_processing", add_task_init_call=True
):
    """Create a draft single-process task that runs a repo script.

    args maps argparse dest names to values; ClearML applies them as Args/* parameters.
    Set add_task_init_call=False for scripts that call Task.init themselves.
    """
    task = Task.create(
        project_name=project,
        task_name=name,
        task_type=task_type,
        repo=REPO,
        branch=branch,
        script=script,
        argparse_args=[(key, str(value)) for key, value in args.items()],
        docker=image,
        docker_bash_setup_script=SETUP_SCRIPT,
        packages=PACKAGES,
        add_task_init_call=add_task_init_call,
    )
    set_slurm_resources(task, nodes=1, ntasks_per_node=1, cpus_per_task=32)
    task.add_tags(list(TAGS) + list(tags))
    return task


def describe(task):
    return f"{task.id} {task.name!r} ({task.get_project_name()}) {task.get_output_log_web_page()}"

