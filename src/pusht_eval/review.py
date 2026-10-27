"""Replay videos from logged trajectories and a blind human-review sample.

LeRobot only saves videos for the first 10 episodes of a run. Because every
episode's states are logged, any episode can be re-rendered afterwards by
resetting PushT to the logged state at each step. The replay shows the
environment as simulated (the policy's perturbed 96x96 view is not shown;
for occlusion and erase runs the square's footprint is drawn as a gray outline).
The replay sets each logged body pose directly; tests/test_replay.py checks
it against frames rendered live during a rollout.

The review sample is blind: videos are named by an anonymous id and carry no
category or coverage overlay; the mapping to condition/seed/auto category is
kept in a separate key file.
"""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

from pusht_eval.failures import CATEGORIES
from pusht_eval.perturbations import SQUARE_KINDS, PerturbationSpec, occlusion_box

VIS_SIZE = 384
OBS_SIZE = 96


def render_replay(traj: dict, perturbation: PerturbationSpec | None = None, fps_stride: int = 1) -> list[np.ndarray]:
    import gym_pusht  # noqa: F401  (registers the env)

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        visualization_width=VIS_SIZE,
        visualization_height=VIS_SIZE,
    )
    base = env.unwrapped
    agent, block = traj["agent_pos"], traj["block_pose"]
    executed, raw = traj["action"], traj.get("policy_action", traj["action"])
    box = None
    if perturbation is not None and perturbation.kind in SQUARE_KINDS and not perturbation.is_clean:
        row, col, h, w = occlusion_box(int(traj["seed"]), int(perturbation.level), OBS_SIZE)
        scale = VIS_SIZE / OBS_SIZE
        box = (int(col * scale), int(row * scale), int((col + w) * scale), int((row + h) * scale))
    frames = []
    try:
        env.reset(seed=int(traj["seed"]))
        for t in range(0, len(agent), fps_stride):
            # Set the logged pose directly: angle first, then position. gym_pusht's
            # reset_to_state/_set_state sets position first and then the angle, which
            # rotates the T about its centre of gravity and shifts it (fine for its own
            # sampled initial states, wrong for replaying a logged body pose).
            base.agent.position = (float(agent[t][0]), float(agent[t][1]))
            base.block.angle = float(block[t][2])
            base.block.position = (float(block[t][0]), float(block[t][1]))
            # Drawing reads cached shape geometry, normally refreshed by a physics step.
            base.space.reindex_shapes_for_body(base.agent)
            base.space.reindex_shapes_for_body(base.block)
            base._last_action = executed[t - 1] if t > 0 else None
            frame = np.ascontiguousarray(env.render())
            if t > 0 and not np.allclose(raw[t - 1], executed[t - 1]):
                px = (np.asarray(raw[t - 1]) / 512 * VIS_SIZE).astype(int)
                cv2.circle(frame, (int(px[0]), int(px[1])), 6, (0, 90, 255), 2)
            if box is not None:
                cv2.rectangle(frame, box[:2], box[2:], (110, 110, 110), 2)
            cv2.putText(frame, f"t={t:3d}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            frames.append(frame)
    finally:
        env.close()
    return frames


def write_replay(traj: dict, path: Path, perturbation: PerturbationSpec | None = None, fps: int = 10) -> Path:
    from lerobot.utils.io_utils import write_video

    path.parent.mkdir(parents=True, exist_ok=True)
    write_video(path, render_replay(traj, perturbation), fps)
    return path


def stratified_sample(rows: list[dict], per_category: int, seed: int = 0) -> list[dict]:
    """Pick up to ``per_category`` episodes per auto category, spread over conditions."""
    rng = random.Random(seed)
    by_cat: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_cat[row["category"]][row["condition"]].append(row)
    picked = []
    for cat in CATEGORIES:
        pools = {c: rng.sample(v, len(v)) for c, v in sorted(by_cat[cat].items())}
        chosen: list[dict] = []
        while len(chosen) < per_category and any(pools.values()):
            for cond in list(pools):  # round-robin across conditions
                if pools[cond] and len(chosen) < per_category:
                    chosen.append(pools[cond].pop())
        picked.extend(chosen)
    rng.shuffle(picked)
    return picked


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    n = len(labels_a)
    if n == 0:
        return float("nan")
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    ca, cb = Counter(labels_a), Counter(labels_b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return float("nan") if expected == 1 else (observed - expected) / (1 - expected)


def agreement(sheet_csv: Path, key_csv: Path) -> dict:
    with open(sheet_csv, newline="", encoding="utf-8-sig") as f:
        # tolerate formatting copied from the docs, e.g. `near_miss`
        human = {r["review_id"]: r["human_category"].strip().strip("`'\"").strip() for r in csv.DictReader(f)}
    with open(key_csv, newline="", encoding="utf-8-sig") as f:
        auto = {r["review_id"]: r["auto_category"] for r in csv.DictReader(f)}
    ids = [i for i in auto if human.get(i)]
    unknown = sorted({human[i] for i in ids} - set(CATEGORIES))
    if unknown:
        raise ValueError(f"human labels must be one of {CATEGORIES}; got {unknown}")
    a = [auto[i] for i in ids]
    h = [human[i] for i in ids]
    confusion = {ca: {ch: sum(1 for x, y in zip(a, h) if x == ca and y == ch) for ch in CATEGORIES} for ca in CATEGORIES}
    return {
        "n_labeled": len(ids),
        "n_sampled": len(auto),
        "accuracy": (sum(x == y for x, y in zip(a, h)) / len(ids)) if ids else float("nan"),
        "cohen_kappa": cohen_kappa(a, h),
        "confusion_auto_rows_human_cols": confusion,
    }
