"""Per-episode, seeded perturbations for PushT sub-environments.

Each wrapper sits between a single PushT environment and LeRobot's vector env.
Randomness comes from a generator seeded by ``(episode_seed, kind_id)``, so a
given episode sees the same perturbation regardless of batch size, condition
order or the policy's own random draws.

Units
-----
- ``action_noise``: Gaussian std in PushT action units (pixels of the 512x512
  workspace); the noisy target is clipped to the action space.
- ``action_delay``: number of control steps (10 Hz, so 1 step = 100 ms) between
  the policy emitting an action and the environment executing it. Until the
  first delayed action arrives the pusher holds its reset position.
- ``image_noise``: Gaussian std in uint8 intensity units, drawn independently
  for every pixel and step, added to the policy's 96x96 observation only.
- ``occlusion``: side length, in observation pixels, of a mid-gray square drawn
  on the policy's 96x96 observation. Its position is drawn once per episode.
  Mid-gray (128) is close to the T block's slate gray (143, 163, 184), so the
  square also acts as a T-coloured distractor.
- ``erase``: control for ``occlusion``: the same square at the same position,
  filled with the white background (255), so it hides whatever is under it
  without adding a gray object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

KINDS = {"none": 0, "action_noise": 1, "action_delay": 2, "image_noise": 3, "occlusion": 4, "erase": 5}
OCCLUSION_FILL = 128
ERASE_FILL = 255  # PushT background
SQUARE_KINDS = ("occlusion", "erase")


@dataclass(frozen=True)
class PerturbationSpec:
    kind: str = "none"
    level: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown perturbation kind {self.kind!r}; expected one of {sorted(KINDS)}")
        if not np.isfinite(self.level) or self.level < 0:
            raise ValueError("perturbation level must be finite and nonnegative")
        if self.kind == "action_delay" and float(self.level) != int(self.level):
            raise ValueError("action_delay level must be an integer number of steps")
        if self.kind == "none" and self.level != 0:
            raise ValueError("kind 'none' takes level 0")

    @classmethod
    def parse(cls, text: str) -> "PerturbationSpec":
        """Parse ``kind:level`` (e.g. ``action_noise:10``) or ``none``."""
        if text in ("none", "clean"):
            return cls()
        kind, sep, level = text.partition(":")
        if not sep:
            raise ValueError(f"expected kind:level, got {text!r}")
        return cls(kind, float(level))

    @property
    def is_clean(self) -> bool:
        return self.kind == "none" or self.level == 0

    @property
    def name(self) -> str:
        if self.is_clean:
            return "clean"
        level = int(self.level) if float(self.level).is_integer() else self.level
        return f"{self.kind}_{level}"

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind if not self.is_clean else "none", "level": self.level if not self.is_clean else 0}


def episode_rng(episode_seed: int, kind: str) -> np.random.Generator:
    return np.random.default_rng([int(episode_seed), KINDS[kind], 20261027])


def occlusion_box(episode_seed: int, side: int, image_size: int = 96) -> tuple[int, int, int, int]:
    """Return (row, col, height, width) of the occluder for this episode."""
    side = int(min(max(side, 0), image_size))
    rng = episode_rng(episode_seed, "occlusion")  # shared by "erase": identical positions
    row = int(rng.integers(0, image_size - side + 1))
    col = int(rng.integers(0, image_size - side + 1))
    return row, col, side, side


class PerturbationWrapper(gym.Wrapper):
    """Apply one perturbation to a single PushT env during seeded episodes.

    Gymnasium's argument-less autoreset (after an episode ends inside a batch)
    is passed through untouched: LeRobot discards those transitions.
    """

    def __init__(self, env: gym.Env, spec: PerturbationSpec, recorder=None):
        super().__init__(env)
        self.perturbation = spec
        self.recorder = recorder
        self._active = False
        self._rng: np.random.Generator | None = None
        self._queue: list[np.ndarray] = []
        self._box: tuple[int, int, int, int] | None = None
        self._low = np.asarray(env.action_space.low, dtype=np.float64)
        self._high = np.asarray(env.action_space.high, dtype=np.float64)

    # -- observation side -------------------------------------------------
    def _perturb_obs(self, obs: dict) -> dict:
        if not self._active or self.perturbation.is_clean or self.perturbation.kind not in ("image_noise", *SQUARE_KINDS):
            return obs
        pixels = np.asarray(obs["pixels"])
        if self.perturbation.kind == "image_noise":
            noisy = pixels.astype(np.float32) + self._rng.normal(0.0, self.perturbation.level, pixels.shape)
            pixels = np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
        else:
            row, col, h, w = self._box
            pixels = pixels.copy()
            fill = OCCLUSION_FILL if self.perturbation.kind == "occlusion" else ERASE_FILL
            pixels[row : row + h, col : col + w, :] = fill
        return {**obs, "pixels": pixels}

    # -- action side ------------------------------------------------------
    def _perturb_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if not self._active or self.perturbation.is_clean:
            return action
        if self.perturbation.kind == "action_noise":
            noisy = action + self._rng.normal(0.0, self.perturbation.level, action.shape)
            return np.clip(noisy, self._low, self._high)
        if self.perturbation.kind == "action_delay":
            self._queue.append(action.copy())
            return self._queue.pop(0)
        return action

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._active = seed is not None
        if self._active:
            self._rng = episode_rng(seed, self.perturbation.kind)
            if self.perturbation.kind == "action_delay":
                hold = np.asarray(obs["agent_pos"], dtype=np.float64)
                self._queue = [hold.copy() for _ in range(int(self.perturbation.level))]
            if self.perturbation.kind in SQUARE_KINDS:
                self._box = occlusion_box(seed, int(self.perturbation.level), np.asarray(obs["pixels"]).shape[0])
        return self._perturb_obs(obs), info

    def step(self, action):
        executed = self._perturb_action(action)
        if self.recorder is not None and self._active:
            self.recorder.note_policy_action(action)
        obs, reward, terminated, truncated, info = self.env.step(executed.astype(np.float32))
        obs = self._perturb_obs(obs)
        if terminated or truncated:
            self._active = False
        return obs, reward, terminated, truncated, info
