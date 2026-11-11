"""Create (and by default enqueue) a ClearML training task for train.py.

Positional arguments are Hydra overrides, applied on top of the synthetic-data
settings and small model in common.py, or on the real ERA5 subset settings with
--real-data. Examples:

    # small model on one node, data parallel
    python clearml_tools/submit_train.py --name swin-dp4 model.embed_dim=128 model.depth=4

    # hybrid parallelism on two nodes: TP=2, SP=(2,2), DP=2
    python clearml_tools/submit_train.py --name swin-hybrid --nodes 2 \\
        parallelism.tp=2 parallelism.sp1=2 parallelism.sp2=2 model.patch_size=2

    # short test on Perlmutter in the debug QOS
    python clearml_tools/submit_train.py --name swin-dp4-pm --system perlmutter --qos debug --time 10
"""

import argparse

from clearml import Task

import common

# A small Swin that trains in about a minute on the synthetic data
SMALL_MODEL = {
    "data.batch_size": 16,
    "data.limit_nsamples": 352,  # a multiple of the batch size; see notes on DALI epochs
    "optimizer": "adamw",
    "optimizer.lr": 1e-3,
    "optimizer.max_iterations": 40,
    "optimizer.warmup_steps": 10,
    "train.log_every": 20,
    "model.patch_size": 4,
    "model.embed_dim": 128,
    "model.depth": 4,
    "model.num_heads": 2,
    "model.window_size": "[9,18]",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="task name; also the run_tag unless overridden")
    parser.add_argument("--run-name", default="dev", help="experiment group (cfg.run_name)")
    parser.add_argument("--project", default=common.PROJECT)
    common.add_system_args(parser)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--qos", help="Slurm QOS, e.g. debug (default: the template's)")
    parser.add_argument("--time", help="Slurm time limit, e.g. 10 or 0:30:00 (default: the template's)")
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--real-data", action="store_true", help="use the real ERA5 subset settings instead of the synthetic data and small model")
    parser.add_argument("--draft", action="store_true", help="create the task without enqueuing it")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, key=value")
    args = parser.parse_args()
    common.apply_system_args(args)

    overrides = dict(common.REAL_DATA_OVERRIDES) if args.real_data else {**common.SYNTHETIC_DATA_OVERRIDES, **SMALL_MODEL}
    overrides.update({"run_name": args.run_name, "run_tag": args.name})
    overrides.update(common.parse_overrides(args.overrides))
    if "parallelism.micro_batch_size" not in overrides and "data.batch_size" in overrides:
        overrides["parallelism.micro_batch_size"] = common.micro_batch_without_accum(
            int(overrides["data.batch_size"]), args.nodes,
            **{k: overrides.get(f"parallelism.{k}", 1) for k in ("tp", "sp1", "sp2")},
        )

    task = common.create_train_task(args.name, overrides, nodes=args.nodes, project=args.project, tags=args.tags)
    common.set_task_properties(task, qos=args.qos, time=args.time)
    if args.draft:
        print("Created", common.describe(task))
    else:
        Task.enqueue(task, queue_name=args.queue)
        print("Enqueued", common.describe(task))


if __name__ == "__main__":
    main()
