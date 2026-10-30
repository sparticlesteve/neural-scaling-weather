"""Run an isoFLOP scaling campaign as a ClearML pipeline.

Same structure as pretrain_campaign.py + cooldown_campaign.py, as one DAG:

    pretrain <model>  (constant LR, stops at the last checkpoint its cooldowns need)
      -> cooldown <model> to each FLOP budget  (branches from a pretrain checkpoint)
    all cooldowns -> isoflop-analysis

Each cooldown waits only for its own model's pretrain. Cooldown lengths are set
so that each run spends one of the FLOP budgets. The steps run on the cluster
queue. The controller runs in this process (start_locally), so the script
blocks until the pipeline ends; run it in the background to keep working.
It prints the pipeline ID first, for watching the run.

The default is a minimal functional test on the synthetic 1-degree data:

    python clearml_tools/isoflop_pipeline.py --dry-run   # print the plan
    python clearml_tools/isoflop_pipeline.py
"""

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

from clearml import PipelineController

import common

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from cooldown_campaign import cooldown_from_checkpoint_iter  # noqa: E402
from pretrain_campaign import BASE_NODES, HEAD_DIM, build_train_params, parse_scaling_tag  # noqa: E402
from utils.flops_utils import FlopsCalculator  # noqa: E402

# Minimal test campaign on the synthetic data (1-degree grid, 180 x 360)
TEST_MODELS = ("p4-e64-d2-lr1em3", "p4-e128-d4-lr1em3", "p4-e192-d6-lr1em3")
TEST_BUDGETS = (3e13, 1e14)
IMG_SIZE = (180, 360)
N_CHANNELS = 71
N_INVARIANTS = 3


def flops_per_iteration(params: dict, batch_size: int, img_size=IMG_SIZE) -> float:
    """Training FLOPs per optimizer step, as the trainer counts them"""
    window = [int(x) for x in params["model.window_size"].strip("[]").split(",")]
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            window_size=window,
            embed_dim=params["model.embed_dim"],
            num_heads=params["model.num_heads"],
            depth=params["model.depth"],
            patch_size=params["model.patch_size"],
        ),
        parallelism=SimpleNamespace(micro_batch_size=1),
        train=SimpleNamespace(temporal_context_window=1),
    )
    calc = FlopsCalculator(cfg, h=img_size[0], w=img_size[1], c_in=N_CHANNELS + N_INVARIANTS, c_out=N_CHANNELS)
    # TFLOPs per sample, x3 for forward + backward
    return calc.flops() * 1e12 * 3 * batch_size


def build_plan(models, budgets, run_name, checkpoint_step, cooldown_fraction, batch_size, train_start_year):
    """Pretrain and cooldown runs, with the Hydra overrides for each"""
    plan = []
    for base_tag in models:
        params = build_train_params(
            base_tag, 1, 1, max_iterations=0, log_every=checkpoint_step,
            train_start_year=train_start_year, img_size=IMG_SIZE,
        )
        params["data.batch_size"] = batch_size
        params["model.num_heads"] = max(1, params["model.embed_dim"] // HEAD_DIM)
        per_iter = flops_per_iteration(params, batch_size)
        nodes = BASE_NODES // 4  # one node: these models need no model parallelism
        params["parallelism.micro_batch_size"] = common.micro_batch_without_accum(batch_size, nodes)

        cooldowns = []
        for budget in budgets:
            to_iter = round(budget / per_iter)
            from_iter = cooldown_from_checkpoint_iter(to_iter, cooldown_fraction, checkpoint_step)
            if to_iter - from_iter < int(to_iter * cooldown_fraction) or from_iter >= to_iter:
                raise ValueError(f"{base_tag}: budget {budget:.1e} gives too few iterations ({to_iter})")
            cooldowns.append({
                "run_tag": f"{base_tag}-cool{to_iter}-base",
                "budget": budget,
                "params": {
                    **params,
                    "run_name": run_name,
                    "run_tag": f"{base_tag}-cool{to_iter}-base",
                    "optimizer.max_iterations": to_iter,
                    "optimizer.scheduler": "cooldown",
                    "optimizer.cooldown_from_iter": from_iter,
                    "optimizer.cooldown_to_iter": to_iter,
                    "optimizer.cooldown_fraction": cooldown_fraction,
                    "train.branch_from": os.path.join(
                        common.EXP_ROOT, run_name, base_tag, "checkpoints", f"ckpt_iter{from_iter}.tar"
                    ),
                    "logging.clearml_checkpoint_iters": f"[{to_iter}]",
                },
                "from_iter": from_iter,
                "to_iter": to_iter,
            })

        branch_iters = sorted({c["from_iter"] for c in cooldowns})
        plan.append({
            "run_tag": base_tag,
            "nodes": nodes,
            "flops_per_iter": per_iter,
            "params": {
                **params,
                "run_name": run_name,
                "run_tag": base_tag,
                "optimizer.max_iterations": branch_iters[-1],
                "logging.clearml_checkpoint_iters": "[" + ",".join(map(str, branch_iters)) + "]",
            },
            "cooldowns": cooldowns,
        })
    return plan


def print_plan(plan):
    for run in plan:
        print(f"pretrain {run['run_tag']}: {run['params']['optimizer.max_iterations']} iters, "
              f"{run['flops_per_iter']:.2e} FLOP/iter, nodes={run['nodes']}")
        for c in run["cooldowns"]:
            print(f"  cooldown {c['run_tag']}: {c['from_iter']} -> {c['to_iter']} iters, budget {c['budget']:.1e}")


def build_pipeline(plan, args):
    pipe = PipelineController(
        name=args.name,
        project=args.project,
        target_project=args.project,
        version="0.1",
        add_pipeline_tags=False,
        abort_on_failure=False,
    )
    pipe.add_tags(list(common.TAGS))
    pipe.set_default_execution_queue(args.queue)

    # Template tasks the steps clone; the steps override all run-specific settings
    template = common.create_train_task(
        f"{args.name} train template", common.SYNTHETIC_DATA_OVERRIDES, project=args.project, tags=["template"]
    )
    analysis = common.create_script_task(
        f"{args.name} analysis template", "clearml_tools/isoflop_analysis.py", {}, project=args.project,
        tags=["template"], task_type="qc", add_task_init_call=False,
    )

    data_overrides = {**common.SYNTHETIC_DATA_OVERRIDES, "data.limit_nsamples": args.limit_nsamples}
    cooldown_steps = []
    for run in plan:
        pipe.add_step(
            name=run["run_tag"],
            base_task_id=template.id,
            parameter_override={
                **common.hydra_params({**data_overrides, **run["params"]}),
                "properties/num_nodes": str(run["nodes"]),
            },
            cache_executed_step=False,
        )
        for c in run["cooldowns"]:
            pipe.add_step(
                name=c["run_tag"],
                parents=[run["run_tag"]],
                base_task_id=template.id,
                parameter_override={
                    **common.hydra_params({**data_overrides, **c["params"]}),
                    "properties/num_nodes": str(run["nodes"]),
                },
                cache_executed_step=False,
            )
            cooldown_steps.append(c["run_tag"])

    pipe.add_step(
        name="isoflop-analysis",
        parents=cooldown_steps,
        base_task_id=analysis.id,
        cache_executed_step=False,
    )
    return pipe


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="isoflop-campaign", help="pipeline name")
    parser.add_argument("--run-name", default=time.strftime("isoflop-%Y%m%d-%H%M"),
                        help="experiment group (cfg.run_name); also the output directory, so keep it unique "
                             "(default: timestamped)")
    parser.add_argument("--project", default=common.PROJECT)
    parser.add_argument("--queue", default=common.QUEUE)
    parser.add_argument("--models", nargs="+", default=TEST_MODELS, help="tags like p4-e128-d4-lr1em3")
    parser.add_argument("--budgets", nargs="+", type=float, default=TEST_BUDGETS, help="FLOP budgets")
    parser.add_argument("--checkpoint-step", type=int, default=10)
    parser.add_argument("--cooldown-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-start-year", type=int, default=2016)
    parser.add_argument("--limit-nsamples", type=int, default=352, help="multiple of the batch size")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args()

    plan = build_plan(
        args.models, args.budgets, args.run_name, args.checkpoint_step,
        args.cooldown_fraction, args.batch_size, args.train_start_year,
    )
    print_plan(plan)
    if args.dry_run:
        print(json.dumps(plan, indent=1, default=str)[:3000])
        return

    pipe = build_pipeline(plan, args)
    # The controller blocks below until the pipeline ends; print the ID first so it can be watched
    print(f"Pipeline ID: {pipe.id}", flush=True)
    pipe.start_locally(run_pipeline_steps_locally=False)
    print("Pipeline finished")


if __name__ == "__main__":
    main()
