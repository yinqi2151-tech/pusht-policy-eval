import numpy as np
import pytest

from pusht_eval.perturbations import OCCLUSION_FILL, PerturbationSpec, PerturbationWrapper, occlusion_box

gym_pusht = pytest.importorskip("gym_pusht")
gym = pytest.importorskip("gymnasium")


def make_env():
    return gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array", max_episode_steps=300)


class Spy(gym.Wrapper):
    """Records the action that actually reaches PushT."""

    def __init__(self, env):
        super().__init__(env)
        self.executed = []

    def step(self, action):
        self.executed.append(np.array(action, dtype=np.float64))
        return self.env.step(action)


def rollout(spec, seed=7, steps=12):
    spy = Spy(make_env())
    env = PerturbationWrapper(spy, spec)
    obs, _ = env.reset(seed=seed)
    observations, policy_actions = [obs], []
    for t in range(steps):
        a = np.array([250.0 + 80 * np.sin(0.3 * t), 250.0 + 80 * np.cos(0.2 * t)], dtype=np.float32)
        policy_actions.append(a.astype(np.float64))
        obs, *_ = env.step(a)
        observations.append(obs)
    env.close()
    return observations, policy_actions, spy.executed


def test_parse_and_names():
    assert PerturbationSpec.parse("none").name == "clean"
    assert PerturbationSpec.parse("action_noise:0").name == "clean"
    assert PerturbationSpec.parse("action_delay:2").name == "action_delay_2"
    assert PerturbationSpec.parse("image_noise:7.5").name == "image_noise_7.5"
    for bad in ("blur:3", "action_noise:-1", "action_delay:1.5", "action_noise"):
        with pytest.raises(ValueError):
            PerturbationSpec.parse(bad)


def test_clean_is_exact_passthrough():
    obs_w, pol, exe = rollout(PerturbationSpec())
    env = make_env()
    obs, _ = env.reset(seed=7)
    raw = [obs]
    for a in pol:
        obs, *_ = env.step(a.astype(np.float32))
        raw.append(obs)
    assert all(np.array_equal(a, b) for a, b in zip(pol, exe))
    assert all(np.array_equal(x["pixels"], y["pixels"]) for x, y in zip(obs_w, raw))


def test_action_noise_is_seeded_and_scaled():
    _, pol, exe = rollout(PerturbationSpec("action_noise", 20), steps=200)
    _, _, exe2 = rollout(PerturbationSpec("action_noise", 20), steps=200)
    noise = np.array(exe) - np.array(pol)
    assert all(np.array_equal(a, b) for a, b in zip(exe, exe2)), "same episode seed must give the same noise"
    assert 15 < noise.std() < 25
    _, _, other_seed = rollout(PerturbationSpec("action_noise", 20), seed=8, steps=200)
    assert not np.allclose(exe, other_seed)


def test_action_delay_shifts_and_holds_start():
    obs, pol, exe = rollout(PerturbationSpec("action_delay", 3))
    start = obs[0]["agent_pos"]
    for t in range(3):
        np.testing.assert_allclose(exe[t], start)
    for t in range(3, len(exe)):
        np.testing.assert_array_equal(exe[t], pol[t - 3])


def test_image_noise_only_touches_pixels():
    clean, _, exe_c = rollout(PerturbationSpec())
    noisy, _, exe_n = rollout(PerturbationSpec("image_noise", 16))
    assert all(np.array_equal(a, b) for a, b in zip(exe_c, exe_n))
    assert all(np.array_equal(a["agent_pos"], b["agent_pos"]) for a, b in zip(clean, noisy))
    diff = noisy[5]["pixels"].astype(int) - clean[5]["pixels"].astype(int)
    assert noisy[5]["pixels"].dtype == np.uint8 and 8 < diff.std() < 18


def test_occlusion_box_fixed_per_episode():
    clean, _, _ = rollout(PerturbationSpec())
    occl, _, _ = rollout(PerturbationSpec("occlusion", 24))
    row, col, h, w = occlusion_box(7, 24)
    for c, o in zip(clean, occl):
        inside = o["pixels"][row : row + h, col : col + w]
        assert (inside == OCCLUSION_FILL).all()
        mask = np.ones(c["pixels"].shape[:2], bool)
        mask[row : row + h, col : col + w] = False
        assert np.array_equal(c["pixels"][mask], o["pixels"][mask])
    assert occlusion_box(7, 24) == occlusion_box(7, 24) != occlusion_box(8, 24)


def test_autoreset_is_not_perturbed():
    spy = Spy(make_env())
    env = PerturbationWrapper(spy, PerturbationSpec("action_delay", 2))
    env.reset(seed=1)
    env.reset()  # Gymnasium autoreset: no seed
    a = np.array([200.0, 200.0], dtype=np.float32)
    env.step(a)
    np.testing.assert_array_equal(spy.executed[-1], a)


def test_erase_uses_same_square_filled_with_background():
    clean, _, _ = rollout(PerturbationSpec())
    erased, _, _ = rollout(PerturbationSpec("erase", 24))
    occl, _, _ = rollout(PerturbationSpec("occlusion", 24))
    row, col, h, w = occlusion_box(7, 24)
    for c, e, o in zip(clean, erased, occl):
        assert (e["pixels"][row : row + h, col : col + w] == 255).all()
        assert (o["pixels"][row : row + h, col : col + w] == OCCLUSION_FILL).all()
        mask = np.ones(c["pixels"].shape[:2], bool)
        mask[row : row + h, col : col + w] = False
        assert np.array_equal(c["pixels"][mask], e["pixels"][mask])
