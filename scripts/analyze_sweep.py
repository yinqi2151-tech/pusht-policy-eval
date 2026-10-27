"""Summarise a sweep: per-episode table, paired comparison to clean, figures.

    python scripts/analyze_sweep.py [--results results/sweep] [--out results]

Writes results/episodes.csv, results/conditions.csv, results/sweep_summary.md,
results/figures/robustness_curves.png and results/figures/outcome_breakdown.png.
Every episode is cross-checked against LeRobot's eval_info.json first.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pusht_eval.analysis import compare_to_baseline, load_sweep  # noqa: E402
from pusht_eval.failures import CATEGORIES, CATEGORY_LABELS_ZH, thresholds  # noqa: E402
from pusht_eval.plots import outcome_breakdown, robustness_curves  # noqa: E402


def fmt_p(p: float | None) -> str:
    if p is None:
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def markdown(table: list[dict], manifest: dict) -> str:
    lines = [
        "# 鲁棒性扫描结果",
        "",
        f"- 策略：`{manifest['policy_path']}`；设备：{manifest['device']}；LeRobot {manifest['versions']['lerobot']}",
        f"- 每个条件 {manifest['n_episodes']} 回合，种子 {manifest['start_seed']}–{manifest['start_seed'] + manifest['n_episodes'] - 1}，"
        f"批大小 {manifest['batch_size']}，确定性设置：{'是' if manifest['deterministic_torch'] else '否'}",
        "- 所有条件使用同一组初始状态，与 `clean` 逐回合配对。成功率差异用精确 McNemar 检验，"
        "并对全部扰动条件做 Holm 校正；最高奖励差为配对均值及 bootstrap 95% 区间。",
        "- `clean_policyseedN` 是对照组：不加扰动，只改变策略采样的随机种子。它们和 `clean` 之间的差异，"
        "代表同一组初始状态仅因策略随机性产生的波动，不计入 Holm 校正。`erase_N` 是 `occlusion_N` 的对照："
        "方块位置相同，但填充为白色背景。",
        "- 每回合的轨迹先与 LeRobot `eval_info.json` 的最高奖励和成功标志逐一核对，全部一致后才进入统计。",
        "",
        "| 条件 | 成功率 [95% Wilson] | Δ成功率 | 失去/新增成功 | McNemar p (Holm) | Δ平均最高奖励 [95% CI] |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for s in table:
        rate = f"{s['successes']}/{s['n']} = {100 * s['success_rate']:.0f}% [{100 * s['ci_low']:.0f}, {100 * s['ci_high']:.0f}]"
        if s["condition"] == "clean":
            lines.append(f"| **clean** | {rate} | — | — | — | 基线 {s['mean_max_reward']:.3f} |")
            continue
        lines.append(
            f"| {s['condition']} | {rate} | {100 * s['delta_success']:+.0f} pp | {s['lost']}/{s['gained']} | "
            f"{fmt_p(s['mcnemar_p'])} ({fmt_p(s['mcnemar_p_holm']) if 'mcnemar_p_holm' in s else '对照，不校正'}) | "
            f"{s['delta_max_reward']:+.3f} [{s['delta_max_reward_ci_low']:+.3f}, {s['delta_max_reward_ci_high']:+.3f}] |"
        )
    lines += ["", "## 自动结局分类（回合数）", "", "| 条件 | " + " | ".join(CATEGORY_LABELS_ZH[c] for c in CATEGORIES) + " |",
              "| --- |" + " ---: |" * len(CATEGORIES)]
    for s in table:
        lines.append(f"| {s['condition']} | " + " | ".join(str(s[f'n_{c}']) for c in CATEGORIES) + " |")
    lines += ["", "分类阈值（扫描前固定）：`" + json.dumps(thresholds()) + "`。定义见 `docs/failure_taxonomy.md`。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=REPO / "results" / "sweep")
    parser.add_argument("--out", type=Path, default=REPO / "results")
    args = parser.parse_args()

    runs = load_sweep(args.results)
    if "clean" not in runs:
        sys.exit("no clean run found; the clean condition is the paired baseline")
    table = compare_to_baseline(runs)
    episodes = [r for rows in runs.values() for r in rows]
    manifest = json.loads((args.results / "clean" / "run_manifest.json").read_text(encoding="utf-8"))

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "episodes.csv", episodes)
    write_csv(args.out / "conditions.csv", table)
    (args.out / "sweep_summary.md").write_text(markdown(table, manifest), encoding="utf-8")
    title = f"Diffusion Policy on PushT: success under perturbation ({manifest['n_episodes']} paired episodes per point)"
    robustness_curves(table, args.out / "figures" / "robustness_curves.png", title)
    outcome_breakdown(table, args.out / "figures" / "outcome_breakdown.png", "Episode outcomes by condition (rule-based categories)")
    print((args.out / "sweep_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
