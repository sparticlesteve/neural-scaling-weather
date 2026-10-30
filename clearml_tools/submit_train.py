"""Create (and by default enqueue) a ClearML training task for train.py.

Positional arguments are Hydra overrides, applied on top of the synthetic-data
settings in common.py unless --real-data is given. Examples:

    # small model on one node, data parallel
    python clearml_tools/submit_train.py --name swin-dp4 model.embed_dim=128 model.depth=4

    # hybrid parallelism on two nodes: TP=2, SP=(2,2), DP=2
    python clearml_tools/submit_train.py --name swin-hybrid --nodes 2 \\
        parallelism.tp=2 parallelism.sp1=2 parallelism.sp2=2 model.patch_size=2
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
    parser.add_argument("--queue", default=common.QUEUE)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--real-data", action="store_true", help="don't apply the synthetic-data overrides")
    parser.add_argument("--draft", action="store_true", help="create the task without enqueuing it")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, key=value")
    args = parser.parse_args()

    overrides = {} if args.real_data else {**common.SYNTHETIC_DATA_OVERRIDES, **SMALL_MODEL}
    overrides.update({"run_name": args.run_name, "run_tag": args.name})
    overrides.update(common.parse_overrides(args.overrides))

    task = common.create_train_task(args.name, overrides, nodes=args.nodes, project=args.project, tags=args.tags)
    if args.draft:
        print("Created", common.describe(task))
    else:
        Task.enqueue(task, queue_name=args.queue)
        print("Enqueued", common.describe(task))


if __name__ == "__main__":
    main()
