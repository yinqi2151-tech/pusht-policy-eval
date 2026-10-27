import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_sweep import parse_condition  # noqa: E402


def test_parse_condition_names():
    assert parse_condition("none")[2] == "clean"
    assert parse_condition("none@2")[1:] == (2, "clean_policyseed2")
    assert parse_condition("erase:16")[2] == "erase_16"
    assert parse_condition("action_delay:4@1")[2] == "action_delay_4_policyseed1"


def test_configs_share_seeds_and_names_are_unique():
    cfgs = [json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "configs").glob("*.json")]
    assert {(c["start_seed"], c["n_episodes"], c["batch_size"]) for c in cfgs} == {(2000, 100, 25)}
    for c in cfgs:
        names = [parse_condition(x)[2] for x in c["conditions"]]
        assert len(names) == len(set(names))
