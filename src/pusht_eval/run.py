"""Run LeRobot 0.6.1 PushT evaluation with a perturbation and trajectory logging.

LeRobot keeps ownership of policy loading, processors, rollout, metrics and
videos. This entry point only

1. enables deterministic PyTorch settings (optional, on by default),
2. wraps every PushT sub-environment with ``TrajectoryRecorder`` and
   ``PerturbationWrapper`` after LeRobot builds the vector env, and
3. writes ``run_manifest.json`` next to LeRobot's ``eval_info.json``.

Usage (all arguments after ours are forwarded to ``lerobot-eval``)::

    python -m pusht_eval.run --perturb action_noise:10 \
        --policy.path=... --policy.device=cuda --env.type=pusht \
        --eval.n_episodes=100 --eval.batch_size=25 --eval.use_async_envs=false \
        --seed=2000 --output_dir=...
"""

from __future__ import annotations

import os

# cuBLAS needs this before CUDA is initialised to give deterministic matmuls.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import importlib.metadata as metadata  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from pusht_eval.perturbations import PerturbationSpec, PerturbationWrapper  # noqa: E402
from pusht_eval.trajectory import TrajectoryRecorder  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def _arg_value(args: list[str], name: str) -> str | None:
    values = [a.split("=", 1)[1] for a in args if a.startswith(f"--{name}=")]
    if len(values) > 1:
        raise SystemExit(f"--{name} given more than once")
    return values[0] if values else None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "src"], capture_output=True, text=True
        ).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def wrap_vector_env(vec: gym.vector.VectorEnv, spec: PerturbationSpec, traj_dir: Path | None) -> None:
    """Wrap each sub-env of a SyncVectorEnv in place: Perturbation(Recorder(PushT))."""
    if not isinstance(vec, gym.vector.SyncVectorEnv):
        raise SystemExit("pusht_eval needs --eval.use_async_envs=false (SyncVectorEnv)")
    wrapped = []
    for env in vec.envs:
        recorder = TrajectoryRecorder(env, traj_dir) if traj_dir is not None else None
        wrapped.append(PerturbationWrapper(recorder or env, spec, recorder=recorder))
    vec.envs = wrapped


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--perturb", default="none", help="kind:level, e.g. action_noise:10 (default: none)")
    parser.add_argument("--no-deterministic", action="store_true", help="keep LeRobot's cuDNN benchmark mode")
    parser.add_argument("--no-trajectories", action="store_true", help="skip per-step trajectory logging")
    parser.add_argument(
        "--policy-seed-offset", type=int, default=0,
        help="re-seed torch with seed+offset after LeRobot's set_seed: same initial states, "
             "different diffusion sampling noise (null control for run-to-run variation)",
    )
    args, lerobot_args = parser.parse_known_args(argv)

    spec = PerturbationSpec.parse(args.perturb)
    if _arg_value(lerobot_args, "eval.use_async_envs") != "false":
        parser.error("pass --eval.use_async_envs=false (perturbation wrappers need in-process envs)")
    for required in ("output_dir", "seed", "eval.n_episodes", "eval.batch_size", "policy.path"):
        if _arg_value(lerobot_args, required) is None:
            parser.error(f"pass --{required}=...")
    output_dir = Path(_arg_value(lerobot_args, "output_dir"))
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output_dir {output_dir} is not empty; LeRobot would refuse or mix runs")
    traj_dir = None if args.no_trajectories else output_dir / "trajectories"
    condition = spec.name + (f"_policyseed{args.policy_seed_offset}" if args.policy_seed_offset else "")
    deterministic = not args.no_deterministic

    from lerobot.scripts import lerobot_eval

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True

    original_make_env = lerobot_eval.make_env

    def make_env_wrapped(*env_args, **env_kwargs):
        if deterministic:
            # eval_main switches cudnn.benchmark on right before building envs.
            torch.backends.cudnn.benchmark = False
        envs = original_make_env(*env_args, **env_kwargs)
        if args.policy_seed_offset:
            # make_env runs after set_seed(cfg.seed) and before the policy is built and
            # rolled out, so this changes only the policy's random draws; episode
            # initial states come from the per-episode env seeds and stay the same.
            torch.manual_seed(int(_arg_value(lerobot_args, "seed")) + args.policy_seed_offset)
        for tasks in envs.values():
            for vec in tasks.values():
                wrap_vector_env(vec, spec, traj_dir)
        return envs

    lerobot_eval.make_env = make_env_wrapped
    sys.argv = [sys.argv[0], *lerobot_args]
    start = time.perf_counter()
    try:
        lerobot_eval.eval_main()
    finally:
        lerobot_eval.make_env = original_make_env
    duration = time.perf_counter() - start

    n_episodes = int(_arg_value(lerobot_args, "eval.n_episodes"))
    overall = json.loads((output_dir / "eval_info.json").read_text(encoding="utf-8")).get("overall", {})
    manifest = {
        "perturbation": spec.as_dict(),
        "condition": condition,
        "policy_seed_offset": args.policy_seed_offset,
        "policy_path": _arg_value(lerobot_args, "policy.path"),
        "start_seed": int(_arg_value(lerobot_args, "seed")),
        "n_episodes": n_episodes,
        "batch_size": int(_arg_value(lerobot_args, "eval.batch_size")),
        "deterministic_torch": deterministic,
        "trajectories": None if traj_dir is None else "trajectories",
        "duration_s": round(duration, 1),
        "seconds_per_episode": round(duration / n_episodes, 2),
        # LeRobot's own timer around the rollouts only (excludes model loading).
        "rollout_seconds_per_episode": round(float(overall.get("eval_ep_s", float("nan"))), 3),
        "lerobot_args": lerobot_args,
        "versions": {
            "lerobot": metadata.version("lerobot"),
            "gym_pusht": metadata.version("gym_pusht"),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "pusht_eval_commit": _git_commit(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[pusht_eval] {condition}: {n_episodes} episodes in {duration:.0f}s -> {output_dir}")


if __name__ == "__main__":
    main()
