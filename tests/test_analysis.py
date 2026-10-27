import json
import math

import numpy as np
import pytest

from pusht_eval.analysis import compare_to_baseline, load_run, pack_trajectories


def fake_run(tmp_path, name, kind, level, coverages, tamper=False):
    run = tmp_path / name
    traj_dir = run / "trajectories"
    traj_dir.mkdir(parents=True)
    max_rewards, successes = [], []
    for i, cov in enumerate(coverages):
        seed = 2000 + i
        n = len(cov)
        block = np.zeros((n, 3), np.float32)
        block[:, 0] = np.linspace(150, 250, n)
        np.savez_compressed(
            traj_dir / f"seed_{seed}.npz",
            seed=np.int64(seed), goal_pose=np.array([256, 256, math.pi / 4], np.float32),
            success=np.bool_(max(cov[1:]) > 0.95), complete=np.bool_(True),
            coverage=np.array(cov, np.float32), block_pose=block, agent_pos=np.zeros((n, 2), np.float32),
            action=np.zeros((n - 1, 2), np.float32), policy_action=np.zeros((n - 1, 2), np.float32),
            reward=np.clip(np.array(cov[1:], np.float32) / 0.95, 0, 1),
        )
        max_rewards.append(float(min(max(np.float32(c) for c in cov[1:]) / 0.95, 1.0)) + (0.1 if tamper and i == 0 else 0))
        successes.append(max(cov[1:]) > 0.95)
    (run / "eval_info.json").write_text(json.dumps({"per_task": [{"metrics": {
        "max_rewards": max_rewards, "successes": successes, "sum_rewards": [0.0] * len(coverages)}}]}))
    (run / "run_manifest.json").write_text(json.dumps({
        "condition": name, "perturbation": {"kind": kind, "level": level},
        "n_episodes": len(coverages), "start_seed": 2000}))
    pack_trajectories(traj_dir, run / "trajectories.npz")
    return run


COV = [[0.0, 0.5, 0.97], [0.0, 0.3, 0.9], [0.0, 0.2, 0.3], [0.0, 0.6, 0.96]]


def test_load_run_and_compare(tmp_path):
    clean = load_run(fake_run(tmp_path, "clean", "none", 0, COV))
    worse = load_run(fake_run(tmp_path, "action_delay_4", "action_delay", 4, [[0.0, 0.5, 0.6]] * 4))
    assert [r["category"] for r in clean] == ["success", "near_miss", "insufficient", "success"]
    table = compare_to_baseline({"clean": clean, "action_delay_4": worse})
    row = next(s for s in table if s["condition"] == "action_delay_4")
    assert (row["lost"], row["gained"]) == (2, 0) and row["delta_success"] == -0.5


def test_mismatch_with_eval_info_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="disagrees with eval_info"):
        load_run(fake_run(tmp_path, "clean", "none", 0, COV, tamper=True))
