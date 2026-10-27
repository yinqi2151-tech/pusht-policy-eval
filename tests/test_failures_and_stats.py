import math

import numpy as np
import pytest

from pusht_eval.failures import classify, episode_metrics
from pusht_eval.review import cohen_kappa, stratified_sample
from pusht_eval.stats import binom_two_sided_p, holm, mcnemar_exact, paired_bootstrap_ci, wilson_interval

GOAL = np.array([256, 256, math.pi / 4], dtype=np.float32)


def traj(coverage, block_xy=None, success=None, seed=1):
    n = len(coverage)
    block = np.zeros((n, 3), np.float32)
    block[:, :2] = 200
    if block_xy is not None:
        block[:, :2] = block_xy
    cov = np.asarray(coverage, np.float32)
    return {
        "seed": np.int64(seed),
        "coverage": cov,
        "block_pose": block,
        "agent_pos": np.full((n, 2), 100, np.float32),
        "action": np.full((n - 1, 2), 100, np.float32),
        "goal_pose": GOAL,
        "success": np.bool_(cov[1:].max() > 0.95 if success is None else success),
    }


def moving(n):
    return np.stack([np.linspace(150, 250, n), np.full(n, 200)], 1)


@pytest.mark.parametrize(
    "coverage,moves,expected",
    [
        ([0.1, 0.5, 0.97], True, "success"),
        ([0.0, 0.0, 0.0, 0.0], False, "no_push"),
        ([0.1, 0.6, 0.9, 0.88], True, "near_miss"),
        ([0.1, 0.6, 0.7, 0.2], True, "lost_progress"),
        ([0.1, 0.3, 0.35, 0.3], True, "insufficient"),
        ([0.1, 0.3, 0.5, 0.4], True, "insufficient"),  # fell back, but not below half the peak
    ],
)
def test_classify(coverage, moves, expected):
    t = traj(coverage, moving(len(coverage)) if moves else None)
    assert classify(episode_metrics(t)) == expected


def test_metrics_fields():
    m = episode_metrics(traj([0.1, 0.5, 0.97], moving(3)))
    assert m["time_to_success"] == 2 and m["first_push_step"] == 1
    assert m["max_reward"] == 1.0 and abs(m["block_path_px"] - 100) < 1e-3


def test_wilson_matches_known_values():
    lo, hi = wilson_interval(27, 50)
    assert round(lo, 3) == 0.404 and round(hi, 3) == 0.670
    assert wilson_interval(0, 20)[0] == 0.0


def test_binom_and_mcnemar():
    assert binom_two_sided_p(5, 10, 0.5) == pytest.approx(1.0)
    assert binom_two_sided_p(0, 10, 0.5) == pytest.approx(2 * 0.5**10)
    r = mcnemar_exact([True] * 10 + [False] * 10, [False] * 10 + [False] * 10)
    assert (r["lost"], r["gained"]) == (10, 0) and r["p_value"] == pytest.approx(2 * 0.5**10)
    assert mcnemar_exact([True, False], [True, False])["p_value"] == 1.0


def test_holm_monotone():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)


def test_bootstrap_ci_contains_mean():
    diffs = [0.1, -0.05, 0.2, 0.0, 0.15] * 10
    lo, hi = paired_bootstrap_ci(diffs)
    assert lo < np.mean(diffs) < hi


def test_kappa_and_sample():
    assert cohen_kappa(list("aabb"), list("aabb")) == 1.0
    rows = [{"category": c, "condition": f"k{i % 3}", "seed": i} for i, c in enumerate(["near_miss"] * 30 + ["success"] * 2)]
    s = stratified_sample(rows, 8, seed=0)
    assert sum(r["category"] == "near_miss" for r in s) == 8 and sum(r["category"] == "success" for r in s) == 2
    assert len({r["condition"] for r in s if r["category"] == "near_miss"}) == 3
    assert s == stratified_sample(rows, 8, seed=0)
