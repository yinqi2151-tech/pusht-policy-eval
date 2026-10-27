"""Rule-based episode outcome categories computed from logged trajectories.

The thresholds below were fixed before any robustness-sweep episode was run
(see docs/failure_taxonomy.md). Categories are mutually exclusive and checked
in this order:

1. ``success``       coverage exceeded the environment's 0.95 threshold.
2. ``no_push``       the block never moved meaningfully: total path of the
                      block centre < NO_PUSH_PATH_PX and net rotation change
                      < NO_PUSH_ROT_DEG. The policy never engaged the object.
3. ``near_miss``     best coverage >= NEAR_MISS_COVERAGE but never above 0.95.
4. ``lost_progress`` best coverage >= LOST_MIN_PEAK, but final coverage fell
                      below LOST_FINAL_FRACTION of that peak: the block was
                      pushed out again after getting close.
5. ``insufficient``  everything else: the block was pushed but never got
                      close to the goal.

These are descriptions of the logged state, not causal explanations; the
human review sample (pusht_eval.review) measures how often they agree with a
person watching the replay.
"""

from __future__ import annotations

import math

import numpy as np

SUCCESS_COVERAGE = 0.95  # gym_pusht success_threshold
NO_PUSH_PATH_PX = 5.0
NO_PUSH_ROT_DEG = 3.0
NEAR_MISS_COVERAGE = 0.85
LOST_MIN_PEAK = 0.40
LOST_FINAL_FRACTION = 0.5

CATEGORIES = ("success", "no_push", "near_miss", "lost_progress", "insufficient")
CATEGORY_LABELS_ZH = {
    "success": "成功",
    "no_push": "未推动物块",
    "near_miss": "接近但未达阈值",
    "lost_progress": "接近后推离",
    "insufficient": "推动但进展不足",
}


def _angle_change(angles: np.ndarray) -> float:
    unwrapped = np.unwrap(angles.astype(np.float64))
    return float(np.abs(np.diff(unwrapped)).sum())


def episode_metrics(traj: dict[str, np.ndarray]) -> dict[str, float | int | bool]:
    coverage = traj["coverage"].astype(np.float64)
    block = traj["block_pose"].astype(np.float64)
    agent = traj["agent_pos"].astype(np.float64)
    action = traj["action"].astype(np.float64)
    goal = traj["goal_pose"].astype(np.float64)
    steps = len(action)
    after = coverage[1:] if len(coverage) > 1 else coverage
    success_steps = np.flatnonzero(after > SUCCESS_COVERAGE)
    first_contact = np.flatnonzero(np.linalg.norm(block[1:, :2] - block[0, :2], axis=1) > 1.0)
    jerk = np.linalg.norm(np.diff(action, axis=0), axis=1) if steps > 1 else np.zeros(1)
    angle_err = abs((block[0, 2] - goal[2] + math.pi) % (2 * math.pi) - math.pi)
    return {
        "seed": int(traj["seed"]),
        "steps": int(steps),
        "success": bool(traj["success"]),
        "max_coverage": float(after.max()),
        "final_coverage": float(coverage[-1]),
        "max_reward": float(min(after.max() / SUCCESS_COVERAGE, 1.0)),
        "time_to_success": int(success_steps[0] + 1) if len(success_steps) else -1,
        "first_push_step": int(first_contact[0] + 1) if len(first_contact) else -1,
        "block_path_px": float(np.linalg.norm(np.diff(block[:, :2], axis=0), axis=1).sum()),
        "block_rotation_deg": math.degrees(_angle_change(block[:, 2])),
        "agent_path_px": float(np.linalg.norm(np.diff(agent, axis=0), axis=1).sum()),
        "action_step_px": float(jerk.mean()),
        "init_block_goal_dist_px": float(np.linalg.norm(block[0, :2] - goal[:2])),
        "init_block_goal_angle_deg": math.degrees(angle_err),
        "init_agent_block_dist_px": float(np.linalg.norm(agent[0] - block[0, :2])),
        "complete": bool(traj.get("complete", True)),
    }


def classify(metrics: dict) -> str:
    if metrics["success"] or metrics["max_coverage"] > SUCCESS_COVERAGE:
        return "success"
    if metrics["block_path_px"] < NO_PUSH_PATH_PX and metrics["block_rotation_deg"] < NO_PUSH_ROT_DEG:
        return "no_push"
    if metrics["max_coverage"] >= NEAR_MISS_COVERAGE:
        return "near_miss"
    if metrics["max_coverage"] >= LOST_MIN_PEAK and metrics["final_coverage"] < LOST_FINAL_FRACTION * metrics["max_coverage"]:
        return "lost_progress"
    return "insufficient"


def thresholds() -> dict[str, float]:
    return {
        "success_coverage": SUCCESS_COVERAGE,
        "no_push_path_px": NO_PUSH_PATH_PX,
        "no_push_rot_deg": NO_PUSH_ROT_DEG,
        "near_miss_coverage": NEAR_MISS_COVERAGE,
        "lost_min_peak": LOST_MIN_PEAK,
        "lost_final_fraction": LOST_FINAL_FRACTION,
    }
