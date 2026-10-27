"""Run every condition of a sweep config; resumable.

Each condition runs ``python -m pusht_eval.run`` in a fresh process with raw
outputs (LeRobot videos, per-episode trajectories) under ``$OUT_ROOT/runs`` and
copies eval_info.json, run_manifest.json and packed trajectories into
``results/sweep/<condition>/``. A condition whose results already exist with
the same seed/episode/batch settings is skipped, so an interrupted sweep can
simply be restarted.

    python scripts/run_sweep.py configs/sweep_b.json [--only clean,action_noise_10] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pusht_eval.analysis import collect_run  # noqa: E402
from pusht_eval.perturbations import PerturbationSpec  # noqa: E402


def expand(value: str) -> str:
    out_root = os.environ.get("OUT_ROOT", str(Path.home() / "pusht_eval_outputs"))
    return os.path.expanduser(value.replace("${OUT_ROOT}", out_root))


def already_done(dest: Path, cfg: dict) -> bool:
    manifest = dest / "run_manifest.json"
    if not (manifest.exists() and (dest / "eval_info.json").exists()):
        return False
    m = json.loads(manifest.read_text(encoding="utf-8"))
    return (m["start_seed"], m["n_episodes"], m["batch_size"]) == (cfg["start_seed"], cfg["n_episodes"], cfg["batch_size"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--results", type=Path, default=REPO / "results" / "sweep")
    parser.add_argument("--only", default="", help="comma-separated condition names to run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra lerobot args after --extra")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    policy_dir = Path(expand(cfg["policy_dir"]))
    out_root = Path(expand("${OUT_ROOT}"))
    if not args.dry_run and not (policy_dir / "config.json").exists():
        sys.exit(f"policy not found at {policy_dir}; run scripts/prepare_official_diffusion.sh first")
    specs = [PerturbationSpec.parse(c) for c in cfg["conditions"]]
    only = {s for s in args.only.split(",") if s}
    if only:
        unknown = only - {s.name for s in specs}
        if unknown:
            sys.exit(f"unknown condition(s) {sorted(unknown)}; known: {[s.name for s in specs]}")
        specs = [s for s in specs if s.name in only]

    log_dir = REPO / "results" / "logs" / "sweep"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TQDM_DISABLE": "1", "PYTHONPATH": str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    timings = []
    for i, spec in enumerate(specs, 1):
        dest = args.results / spec.name
        if already_done(dest, cfg):
            print(f"[{i}/{len(specs)}] {spec.name}: already done, skipping")
            continue
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = out_root / "runs" / f"{spec.name}-{stamp}"
        cmd = [
            sys.executable, "-m", "pusht_eval.run", f"--perturb={spec.kind}:{spec.level}" if not spec.is_clean else "--perturb=none",
            f"--policy.path={policy_dir}", f"--policy.device={args.device}", "--env.type=pusht",
            f"--eval.n_episodes={cfg['n_episodes']}", f"--eval.batch_size={cfg['batch_size']}",
            "--eval.use_async_envs=false", f"--seed={cfg['start_seed']}", f"--output_dir={out_dir}",
            *args.extra,
        ]
        print(f"[{i}/{len(specs)}] {spec.name}: " + " ".join(cmd[2:]))
        if args.dry_run:
            continue
        start = time.perf_counter()
        with open(log_dir / f"{spec.name}.log", "w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            sys.exit(f"{spec.name} failed (exit {proc.returncode}); see {log_dir / (spec.name + '.log')}")
        collect_run(out_dir, dest)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        remaining = sum(1 for s in specs[i:] if not already_done(args.results / s.name, cfg))
        eta = remaining * sum(timings) / len(timings)
        print(f"    done in {elapsed / 60:.1f} min; about {eta / 60:.0f} min left for {remaining} condition(s)")


if __name__ == "__main__":
    main()
