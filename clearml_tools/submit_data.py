"""Submit a ClearML task that writes the synthetic test dataset on the cluster.

    python clearml_tools/submit_data.py [--system muller|perlmutter] [--queue QUEUE]
"""

import argparse

from clearml import Task

import common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=common.PROJECT)
    common.add_system_args(parser)
    parser.add_argument("--root", help="data root (default: the system's)")
    args = parser.parse_args()
    common.apply_system_args(args)
    args.root = args.root or common.DATA_ROOT

    task = common.create_script_task(
        "make-synthetic-era5",
        "scripts/make_synthetic_era5.py",
        {"root": args.root, "name": common.SYNTHETIC_DATA},
        project=args.project,
        tags=["data"],
    )
    Task.enqueue(task, queue_name=args.queue)
    print("Enqueued", common.describe(task))


if __name__ == "__main__":
    main()
