import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("gym_pusht")
pytest.importorskip("cv2")

from pusht_eval.review import VIS_SIZE, render_replay  # noqa: E402
from pusht_eval.trajectory import TrajectoryRecorder, load_trajectory  # noqa: E402


def live_rollout(tmp_path, seed=3, steps=60):
    """Push the T around while rendering live, and log the same episode."""
    env = gym.make(
        "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array",
        visualization_width=VIS_SIZE, visualization_height=VIS_SIZE, max_episode_steps=steps,
    )
    rec = TrajectoryRecorder(env, tmp_path)
    rec.reset(seed=seed)
    base = env.unwrapped
    frames = [env.render()]
    for t in range(steps):
        # aim through the block so it gets pushed and rotated
        target = np.asarray(base.block.position) + 40 * np.array([np.cos(0.2 * t), np.sin(0.2 * t)])
        _, _, term, trunc, _ = rec.step(np.clip(target, 0, 512).astype(np.float32))
        frames.append(env.render())
        if term or trunc:
            break
    env.close()
    return frames, load_trajectory(tmp_path / f"seed_{seed}.npz")


def test_replay_matches_live_rendering(tmp_path):
    live, traj = live_rollout(tmp_path)
    moved = np.linalg.norm(traj["block_pose"][-1, :2] - traj["block_pose"][0, :2])
    turned = abs(float(traj["block_pose"][-1, 2] - traj["block_pose"][0, 2]))
    assert moved > 10 or turned > 0.2, "rollout should move the block for this test to mean anything"
    replay = render_replay(traj)
    assert len(replay) == len(live)
    for t in range(1, len(live)):
        a, b = live[t].astype(int), replay[t].astype(int)
        b[:30, :80] = a[:30, :80]  # ignore the step-counter overlay
        # float32 logging can flip a few edge pixels; a misplaced T changes thousands
        assert (np.abs(a - b).max(axis=2) > 40).sum() < 150, f"frame {t} differs"
