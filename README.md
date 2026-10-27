# PushT 策略鲁棒性评测

对 LeRobot 官方 Diffusion Policy 在 PushT 仿真推物任务上做**可复现、配对设计**的鲁棒性评测。评测分四步：先确认评测流程可信，再施加 4 类扰动、每类 4 档强度，然后用逐步轨迹对每个回合的结局自动分类，最后抽样做人工盲审，检验自动分类是否可靠。

> 本仓库只评测已有的预训练策略，不训练模型，也不修改 LeRobot 源码。PushT 是平面推物仿真，结论不外推到机械臂抓取或真实机器人。

## 主要结果

### 1. 评测流程校验

| | 本流程（50 回合） | 官方模型卡（500 回合） |
| --- | --- | ---: |
| 成功率 | 54.0%，95% 区间 40.4%–67.0% | 65.4% |
| 平均回合最高奖励 | 0.952 | 0.955 |

项目早期评测两个第三方权重时，成功率只有 0–5%。为找出原因，依次做了以下排查（详见 [docs/validation.md](docs/validation.md)）：

- 核对训练配置，发现其中的 Diffusion 权重只训练了 1000 步（官方为 20 万步）。
- 修复 LeRobot 0.6.1 迁移旧格式权重时的配置序列化 bug。
- 用官方权重复测，结果与公开基准一致。
- 发现同一种子重复运行时结果不同（0.614 对 0.392），通过确定性设置修复。

### 2. 鲁棒性扫描

> 待运行：`bash scripts/run_all.sh`。跑完后，这里会放鲁棒性曲线（`results/figures/robustness_curves.png`）、结局构成（`results/figures/outcome_breakdown.png`）和汇总表（`results/sweep_summary.md`）。

### 3. 自动分类与人工复核的一致率

> 待完成：标注 `results/review/review_sheet.csv` 后，运行 `scripts/review_agreement.py`。

## 方法要点

- **配对设计**：17 个条件（1 个无扰动 + 16 个扰动）使用同一组 100 个初始状态（种子 2000–2099）。扰动的随机数只由“回合种子 + 扰动类型”决定，与批大小和条件顺序无关。
- **统计方法**：成功率给出 Wilson 区间；与无扰动条件的比较使用精确 McNemar 检验，并对 16 个扰动条件做 Holm 校正；平均最高奖励差给出配对 bootstrap 区间。
- **不改 LeRobot**：策略加载、rollout、成功判定和视频录制都由 LeRobot 0.6.1 完成。本仓库只在 LeRobot 建好环境之后，为每个子环境套上记录层和扰动层。不加扰动时，结果与原生 LeRobot 逐回合完全相同。
- **数据交叉核对**：每回合轨迹里的最高奖励和成功标志，都要与 LeRobot 的 `eval_info.json` 逐一核对，一致后才进入统计。
- **可重复**：使用 PyTorch 确定性设置；批量运行的可重复性单独检查（`scripts/check_determinism.py`）；每次运行都写入 `run_manifest.json`，记录版本号、参数和耗时。
- **阈值预先固定**：扰动档位和结局分类的阈值，都在运行任何扫描回合之前就已确定。

详细协议见 [docs/protocol.md](docs/protocol.md)，结局分类规则见 [docs/failure_taxonomy.md](docs/failure_taxonomy.md)。

## 运行

需要 WSL2 或 Linux 系统、一块 NVIDIA GPU，以及 LeRobot 0.6.1 环境。

```bash
conda activate lerobot
pip install matplotlib pytest      # LeRobot 环境之外只需这两个
pytest                             # 单元测试，约 3 秒
bash scripts/run_all.sh            # 下载并迁移权重 → 可重复性检查 → 扫描 → 分析 → 抽取复核样本
```

`run_all.sh` 的每一步都可以断点续跑：中途中断后重新执行同一条命令，已完成的条件会自动跳过。原始输出（LeRobot 视频、逐回合轨迹）写在 `$OUT_ROOT`（默认 `~/pusht_eval_outputs`），汇总后的结果写进本仓库的 `results/` 目录。

单独运行某一个条件：

```bash
PYTHONPATH=src python -m pusht_eval.run --perturb action_delay:2 \
  --policy.path=$HOME/pusht_eval_outputs/migrated/diffusion_pusht --policy.device=cuda \
  --env.type=pusht --eval.n_episodes=100 --eval.batch_size=25 --eval.use_async_envs=false \
  --seed=2000 --output_dir=/tmp/pusht_run
```

## 仓库结构

```
src/pusht_eval/
  run.py                  入口：确定性设置 + 包装 LeRobot 环境 + 写运行清单
  perturbations.py        4 类扰动（每回合独立播种）
  trajectory.py           逐步记录推杆位置、物块位姿、覆盖率和动作
  failures.py             回合指标和 5 类结局规则
  analysis.py             读取结果、与 eval_info 交叉核对、与无扰动条件配对比较
  stats.py                Wilson 区间、精确 McNemar 检验、Holm 校正、bootstrap
  review.py               按状态重新渲染回放视频、盲审抽样、Cohen's κ
  plots.py                README 用的图
  migrate_legacy_policy.py  修复 LeRobot 0.6.1 迁移旧格式权重的 bug
scripts/                  run_all / run_sweep / check_determinism / analyze_sweep / 复核脚本
configs/sweep_b.json      扫描配置（17 个条件）
results/validation/       官方权重的 50 回合校验结果
results/archive/          早期第三方权重的实验结果（原样保存）
docs/                     协议、校验、结局分类、项目过程记录
```

## 第三方资源

- 策略权重：[`lerobot/diffusion_pusht`](https://huggingface.co/lerobot/diffusion_pusht)（Apache-2.0）。
- 依赖：[LeRobot](https://github.com/huggingface/lerobot)、[gym-pusht](https://github.com/huggingface/gym-pusht)。
- 项目代码在 AI 编程工具的协助下完成。实验设计、运行和结论核对由作者负责。
