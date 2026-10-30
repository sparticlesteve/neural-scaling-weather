"""Submit a ClearML task that writes the synthetic test dataset on the cluster.

    python clearml_tools/submit_data.py [--queue muller]
"""

import argparse

from clearml import Task

import common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=common.PROJECT)
    parser.add_argument("--queue", default=common.QUEUE)
    parser.add_argument("--root", default=common.DATA_ROOT)
    args = parser.parse_args()

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
