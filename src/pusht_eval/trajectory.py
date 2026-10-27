"""Per-step state logging for PushT episodes run through LeRobot.

LeRobot's ``eval_info.json`` keeps only per-episode rewards and success flags.
Failure analysis needs the state trajectory, so this wrapper (placed directly
around the PushT env, inside any perturbation) records, for each seeded
episode, the executed action, the policy's raw action, pusher position, block
pose and coverage, and writes ``seed_<seed>.npz`` when the episode ends.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np

STATE_KEYS = ("agent_pos", "block_pose", "coverage")


class TrajectoryRecorder(gym.Wrapper):
    def __init__(self, env: gym.Env, out_dir: str | Path):
        super().__init__(env)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._seed: int | None = None
        self._buf: dict[str, list] | None = None
        self._pending_policy_action: np.ndarray | None = None

    def note_policy_action(self, action) -> None:
        self._pending_policy_action = np.asarray(action, dtype=np.float32).copy()

    def _snapshot(self, coverage: float) -> None:
        base = self.env.unwrapped
        self._buf["agent_pos"].append(np.asarray(base.agent.position, dtype=np.float32))
        self._buf["block_pose"].append(
            np.asarray([*base.block.position, base.block.angle], dtype=np.float32)
        )
        self._buf["coverage"].append(np.float32(coverage))

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is None:
            # Gymnasium autoreset inside a batch: not part of the evaluation.
            self._seed, self._buf = None, None
            return obs, info
        self._seed = int(seed)
        self._buf = {k: [] for k in (*STATE_KEYS, "action", "policy_action", "reward")}
        self._snapshot(self.env.unwrapped._get_coverage())
        self._goal = np.asarray(self.env.unwrapped.goal_pose, dtype=np.float32)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._buf is not None:
            executed = np.asarray(action, dtype=np.float32)
            policy_action = self._pending_policy_action if self._pending_policy_action is not None else executed
            self._buf["action"].append(executed.copy())
            self._buf["policy_action"].append(policy_action)
            self._buf["reward"].append(np.float32(reward))
            self._snapshot(info["coverage"])
            self._pending_policy_action = None
            if terminated or truncated:
                self._flush(success=bool(info.get("is_success", False)))
        return obs, reward, terminated, truncated, info

    def _flush(self, success: bool, complete: bool = True) -> None:
        arrays = {k: np.stack(v) for k, v in self._buf.items() if v}
        np.savez_compressed(
            self.out_dir / f"seed_{self._seed}.npz",
            seed=np.int64(self._seed),
            goal_pose=self._goal,
            success=np.bool_(success),
            complete=np.bool_(complete),
            **arrays,
        )
        self._seed, self._buf = None, None

    def close(self):
        # An episode cut off without terminated/truncated is kept but marked.
        if self._buf is not None and self._buf["action"]:
            self._flush(success=False, complete=False)
        return super().close()


def load_trajectory(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}
