"""Load runs, cross-check them against LeRobot's metrics, and compare conditions."""

from __future__ import annotations

import json
import shutil
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from pusht_eval.failures import CATEGORIES, classify, episode_metrics
from pusht_eval.perturbations import PerturbationSpec
from pusht_eval.stats import holm, mcnemar_exact, paired_bootstrap_ci, wilson_interval

PACK_NAME = "trajectories.npz"


# -- storage -------------------------------------------------------------------
def pack_trajectories(traj_dir: Path, dest: Path) -> int:
    """Pack per-episode ``seed_*.npz`` files into one compressed archive."""
    arrays = {}
    files = sorted(traj_dir.glob("seed_*.npz"))
    for f in files:
        with np.load(f) as d:
            seed = int(d["seed"])
            for k in d.files:
                arrays[f"s{seed}__{k}"] = d[k]
    np.savez_compressed(dest, **arrays)
    return len(files)


def load_trajectories(run_dir: Path) -> dict[int, dict[str, np.ndarray]]:
    out: dict[int, dict[str, np.ndarray]] = {}
    packed = run_dir / PACK_NAME
    if packed.exists():
        with np.load(packed) as d:
            for key in d.files:
                seed_part, field = key.split("__", 1)
                out.setdefault(int(seed_part[1:]), {})[field] = d[key]
        return out
    for f in sorted((run_dir / "trajectories").glob("seed_*.npz")):
        with np.load(f) as d:
            out[int(d["seed"])] = {k: d[k] for k in d.files}
    return out


def collect_run(output_dir: Path, dest: Path) -> None:
    """Copy LeRobot's eval_info, our manifest and packed trajectories into dest."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("eval_info.json", "run_manifest.json"):
        shutil.copy2(output_dir / name, dest / name)
    traj_dir = output_dir / "trajectories"
    if traj_dir.exists():
        pack_trajectories(traj_dir, dest / PACK_NAME)


# -- loading & checks ----------------------------------------------------------
def load_run(run_dir: Path) -> list[dict]:
    """Per-episode rows for one run, cross-checked against eval_info.json."""
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "eval_info.json").read_text(encoding="utf-8"))["per_task"][0]["metrics"]
    n = manifest["n_episodes"]
    start = manifest["start_seed"]
    if len(metrics["max_rewards"]) != n:
        raise ValueError(f"{run_dir}: eval_info has {len(metrics['max_rewards'])} episodes, manifest says {n}")
    trajs = load_trajectories(run_dir)
    spec = PerturbationSpec(**manifest["perturbation"])
    rows = []
    for i in range(n):
        seed = start + i
        row = {
            "condition": manifest["condition"],
            "kind": spec.kind,
            "level": spec.level,
            "seed": seed,
            "success": bool(metrics["successes"][i]),
            "max_reward": float(metrics["max_rewards"][i]),
            "sum_reward": float(metrics["sum_rewards"][i]),
        }
        traj = trajs.get(seed)
        if traj is not None:
            m = episode_metrics(traj)
            if abs(m["max_reward"] - row["max_reward"]) > 1e-4 or m["success"] != row["success"]:
                raise ValueError(
                    f"{run_dir} seed {seed}: trajectory (max_reward={m['max_reward']:.6f}, success={m['success']}) "
                    f"disagrees with eval_info (max_reward={row['max_reward']:.6f}, success={row['success']})"
                )
            m.pop("seed"), m.pop("success"), m.pop("max_reward")
            row.update(m)
            row["category"] = classify({**m, "success": row["success"]})
        rows.append(row)
    return rows


def load_sweep(root: Path) -> dict[str, list[dict]]:
    runs = {}
    for run_dir in sorted(p for p in root.iterdir() if (p / "eval_info.json").exists()):
        rows = load_run(run_dir)
        runs[rows[0]["condition"]] = rows
    return runs


# -- comparison ----------------------------------------------------------------
def summarize_condition(rows: list[dict]) -> dict:
    n = len(rows)
    k = sum(r["success"] for r in rows)
    lo, hi = wilson_interval(k, n)
    cats = Counter(r.get("category") for r in rows)
    return {
        "condition": rows[0]["condition"],
        "kind": rows[0]["kind"],
        "level": rows[0]["level"],
        "n": n,
        "successes": k,
        "success_rate": k / n,
        "ci_low": lo,
        "ci_high": hi,
        "mean_max_reward": statistics.fmean(r["max_reward"] for r in rows),
        **{f"n_{c}": cats.get(c, 0) for c in CATEGORIES},
    }


def compare_to_baseline(runs: dict[str, list[dict]], baseline: str = "clean") -> list[dict]:
    base = {r["seed"]: r for r in runs[baseline]}
    table, pvals = [], {}
    for cond, rows in runs.items():
        summary = summarize_condition(rows)
        if cond != baseline:
            paired = [(base[r["seed"]], r) for r in rows if r["seed"] in base]
            if len(paired) != len(rows):
                raise ValueError(f"{cond}: seeds do not match the baseline run")
            test = mcnemar_exact([b["success"] for b, _ in paired], [r["success"] for _, r in paired])
            diffs = [r["max_reward"] - b["max_reward"] for b, r in paired]
            lo, hi = paired_bootstrap_ci(diffs)
            summary.update(
                delta_success=summary["success_rate"] - sum(b["success"] for b, _ in paired) / len(paired),
                lost=test["lost"],
                gained=test["gained"],
                mcnemar_p=test["p_value"],
                delta_max_reward=statistics.fmean(diffs),
                delta_max_reward_ci_low=lo,
                delta_max_reward_ci_high=hi,
            )
            if rows[0]["kind"] != "none":  # Holm family: perturbed conditions only
                pvals[cond] = test["p_value"]
        table.append(summary)
    adjusted = holm(pvals)
    for s in table:
        if s["condition"] in adjusted:
            s["mcnemar_p_holm"] = adjusted[s["condition"]]
    kind_order = {"none": 0, "action_noise": 1, "action_delay": 2, "image_noise": 3, "occlusion": 4, "erase": 5}
    table.sort(key=lambda s: (kind_order.get(s["kind"], 9), s["level"], s["condition"] != baseline, s["condition"]))
    return table
