# 旧脚本存档

这些脚本生成了 `results/archive/initial_checkpoints/` 里的结果，保留下来是为了能追溯这些结果的来源，之后不再维护。它们依赖旧仓库的目录结构（例如 `from mpc_go2.evaluation.stats import ...`），在本仓库里不能直接运行。

新流程中对应的功能：

| 旧脚本 | 新流程中的对应 |
| --- | --- |
| `eval_pusht_deterministic.py` | `python -m pusht_eval.run --perturb none`（默认开启确定性设置） |
| `eval_pusht_action_noise.py` | `python -m pusht_eval.run --perturb action_noise:<σ>`（支持批量运行，每回合独立播种） |
| `summarize_policy_eval.py`、`compare_policy_evals.py` | `scripts/analyze_sweep.py`（精确 McNemar 检验 + Holm 校正） |
| `derive_pusht_scenes.py`、`analyze_pusht_scenes.py` | 初始几何量直接从轨迹中计算（`failures.episode_metrics`） |
| `make_pusht_contact_sheets.py`、`triage_policy_failures.py` | `scripts/make_review_sample.py`（按状态重新渲染回放视频 + 盲审） |
| `migrate_legacy_policy.py`、`run_official_diffusion_check.sh` | `pusht_eval.migrate_legacy_policy`、`scripts/prepare_official_diffusion.sh` |
