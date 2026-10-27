"""Check that batched evaluation is reproducible and measure the speed-up.

Runs the unperturbed policy on the same seeds three times:
  A, B: batch size N (default 10), deterministic settings -> must match bit for bit
  C:    batch size 1 on the first few seeds -> speed reference (results differ from A,
        because batched GPU kernels and diffusion noise draws differ)
and writes results/checks/determinism_batch<N>.json.

    python scripts/check_determinism.py --policy-dir ~/pusht_eval_outputs/migrated/diffusion_pusht
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pusht_eval.analysis import load_trajectories  # noqa: E402


def run(policy_dir: Path, out_dir: Path, n: int, batch: int, seed: int, device: str, extra: list[str]) -> dict:
    env = {**os.environ, "TQDM_DISABLE": "1", "PYTHONPATH": str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    cmd = [
        sys.executable, "-m", "pusht_eval.run", "--perturb=none", f"--policy.path={policy_dir}",
        f"--policy.device={device}", "--env.type=pusht", f"--eval.n_episodes={n}", f"--eval.batch_size={batch}",
        "--eval.use_async_envs=false", f"--seed={seed}", f"--output_dir={out_dir}", *extra,
    ]
    log = out_dir.parent / f"{out_dir.name}.log"
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as f:
        if subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT).returncode:
            sys.exit(f"run failed, see {log}")
    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((out_dir / "eval_info.json").read_text(encoding="utf-8"))["per_task"][0]["metrics"]
    return {"dir": out_dir, "manifest": manifest, "metrics": metrics}


def trajectories_identical(a: Path, b: Path) -> tuple[bool, list[int]]:
    ta, tb = load_trajectories(a), load_trajectories(b)
    differing = []
    for seed in sorted(set(ta) | set(tb)):
        x, y = ta.get(seed), tb.get(seed)
        if x is None or y is None or x.keys() != y.keys() or any(not np.array_equal(x[k], y[k]) for k in x):
            differing.append(seed)
    return not differing, differing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-root", type=Path, default=Path(os.environ.get("OUT_ROOT", Path.home() / "pusht_eval_outputs")))
    parser.add_argument("--skip-batch1", action="store_true", help="skip the slow batch-size-1 speed reference")
    parser.add_argument("--batch1-episodes", type=int, default=5, help="episodes for the batch-1 speed reference")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = args.out_root / "checks" / f"determinism-{stamp}"
    common = dict(policy_dir=args.policy_dir, n=args.n_episodes, seed=args.seed, device=args.device, extra=args.extra)
    print(f"run A (batch {args.batch_size}) ...", flush=True)
    a = run(out_dir=base / "batchN_a", batch=args.batch_size, **common)
    print(f"run B (batch {args.batch_size}) ...", flush=True)
    b = run(out_dir=base / "batchN_b", batch=args.batch_size, **common)
    c = None
    if not args.skip_batch1:
        print("run C (batch 1) ...", flush=True)
        c = run(out_dir=base / "batch1", batch=1, **{**common, "n": min(args.batch1_episodes, args.n_episodes)})

    same_metrics = a["metrics"]["max_rewards"] == b["metrics"]["max_rewards"] and a["metrics"]["successes"] == b["metrics"]["successes"]
    same_traj, differing = trajectories_identical(a["dir"], b["dir"])
    report = {
        "seeds": [args.seed, args.seed + args.n_episodes - 1],
        "batch_size": args.batch_size,
        "repeat_identical_metrics": same_metrics,
        "repeat_identical_trajectories": same_traj,
        "differing_seeds": differing,
        "rollout_seconds_per_episode": {
            f"batch_{args.batch_size}": a["manifest"]["rollout_seconds_per_episode"],
            **({"batch_1": c["manifest"]["rollout_seconds_per_episode"]} if c else {}),
        },
        "speedup_vs_batch1": round(
            c["manifest"]["rollout_seconds_per_episode"] / a["manifest"]["rollout_seconds_per_episode"], 2
        ) if c else None,
        "success_rate": {
            f"batch_{args.batch_size}": sum(a["metrics"]["successes"]) / args.n_episodes,
            **({"batch_1": sum(c["metrics"]["successes"]) / len(c["metrics"]["successes"])} if c else {}),
        },
        "device": a["manifest"]["device"],
        "versions": a["manifest"]["versions"],
        "note": "batch-1 and batch-N results are not expected to be identical; only repeats at the same batch size are.",
    }
    dest = REPO / "results" / "checks" / f"determinism_batch{args.batch_size}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not (same_metrics and same_traj):
        sys.exit("NOT reproducible at this batch size: do not run the sweep until this is resolved")


if __name__ == "__main__":
    main()
