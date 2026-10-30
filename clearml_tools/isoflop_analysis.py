"""IsoFLOP analysis of a scaling campaign, from metrics stored in ClearML.

Like scripts/isoflop.py, but reads each cooldown's final validation loss,
FLOPs and parameter count from its ClearML task instead of logs.txt:

1. group cooldown runs by FLOP budget (runs within 10% of each other),
2. per budget, fit loss as a parabola in log10(params) and take its minimum,
3. fit the compute-optimal size N_opt = a * C^b across budgets.

As the last step of isoflop_pipeline.py it analyses its sibling steps (the
tasks that share its pipeline parent). It can also run by hand:

    python clearml_tools/isoflop_analysis.py --pipeline <pipeline task id>
"""

import argparse

import numpy as np
from clearml import Task

import common

BUDGET_RTOL = 0.1


def collect_runs(pipeline_id):
    """Final metrics of the completed cooldown runs in a pipeline"""
    tasks = Task.get_tasks(task_filter={"parent": pipeline_id, "status": ["completed"]})
    runs = []
    for task in tasks:
        params = task.get_parameters()
        if params.get("Hydra/optimizer.scheduler") != "cooldown":
            continue
        values = task.get_reported_single_values()
        if "flops" not in values:
            continue
        runs.append({
            "task_id": task.id,
            "name": task.name,
            "params_M": values["params_M"],
            "flops": values["flops"],
            "val_loss": values["final_val_loss"],
        })
    return sorted(runs, key=lambda r: r["flops"])


def group_by_budget(runs):
    groups = []
    for run in runs:
        if groups and run["flops"] <= groups[-1][0]["flops"] * (1 + BUDGET_RTOL):
            groups[-1].append(run)
        else:
            groups.append([run])
    return groups


def fit_isoflop(group):
    """Fit loss vs log10(params) with a parabola; return the optimum or None"""
    x = np.log10([r["params_M"] for r in group])
    y = np.array([r["val_loss"] for r in group])
    if len(group) < 3:
        return None
    a, b, c = np.polyfit(x, y, 2)
    if a <= 0:
        return None  # no interior minimum
    x_opt = -b / (2 * a)
    return {"params_M": 10**x_opt, "val_loss": c - b**2 / (4 * a), "coeffs": [a, b, c]}


def analyse(runs, logger=None):
    groups = group_by_budget(runs)
    optima = []
    for group in groups:
        budget = float(np.mean([r["flops"] for r in group]))
        opt = fit_isoflop(group)
        print(f"budget {budget:.2e}: " + ", ".join(f"{r['params_M']:.2f}M -> {r['val_loss']:.4f}" for r in group))
        if opt:
            print(f"  optimum {opt['params_M']:.2f}M params, loss {opt['val_loss']:.4f}")
            optima.append({"budget": budget, **opt})
        else:
            print("  no interior optimum (need 3+ sizes around the minimum)")
        if logger:
            logger.report_scatter2d(
                "isoFLOP curves", f"{budget:.1e} FLOPs", iteration=0,
                scatter=[[np.log10(r["params_M"]), r["val_loss"]] for r in group],
                xaxis="log10(params / 1M)", yaxis="final val loss", mode="lines+markers",
            )

    fit = None
    if len(optima) >= 2:
        b, log_a = np.polyfit(np.log10([o["budget"] for o in optima]), np.log10([o["params_M"] for o in optima]), 1)
        fit = {"exponent": b, "prefactor_M": 10**log_a}
        print(f"N_opt = {10**log_a:.3g}M * C^{b:.3f}")
    else:
        print("scaling-law fit needs optima at 2+ budgets")

    if logger:
        rows = [["task", "params (M)", "FLOPs", "val loss"]]
        rows += [[r["name"], f"{r['params_M']:.3f}", f"{r['flops']:.3e}", f"{r['val_loss']:.5f}"] for r in runs]
        logger.report_table("cooldown runs", "final metrics", iteration=0, table_plot=rows)
        if optima:
            logger.report_scatter2d(
                "compute-optimal size", "isoFLOP minima", iteration=0,
                scatter=[[np.log10(o["budget"]), np.log10(o["params_M"])] for o in optima],
                xaxis="log10(FLOPs)", yaxis="log10(params / 1M)", mode="markers",
            )
        if fit:
            logger.report_single_value("scaling_exponent", fit["exponent"])
            logger.report_single_value("scaling_prefactor_M", fit["prefactor_M"])
        logger.report_single_value("n_runs", len(runs))
        logger.report_single_value("n_budgets", len(groups))
    return {"groups": groups, "optima": optima, "fit": fit}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", default="", help="pipeline task id (default: this task's parent)")
    parser.add_argument("--project", default=common.PROJECT)
    args = parser.parse_args()

    task = Task.init(project_name=args.project, task_name="isoflop-analysis", task_type="qc", reuse_last_task_id=False)
    task.add_tags(list(common.TAGS))
    pipeline_id = args.pipeline or task.parent
    if not pipeline_id:
        parser.error("no --pipeline given and this task has no parent")
    runs = collect_runs(pipeline_id)
    print(f"{len(runs)} completed cooldown runs in pipeline {pipeline_id}")
    analyse(runs, task.get_logger())


if __name__ == "__main__":
    main()
