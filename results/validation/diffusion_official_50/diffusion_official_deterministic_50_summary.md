# LeRobot 策略评测摘要

- 策略：`lerobot/diffusion_pusht (migrated, 84a7c231)`
- 每次执行的动作步数：`8`
- 任务：`pusht` / `0`
- 原始数据：`diffusion_official_deterministic_50.json`
- 起始随机种子：`2000`（由运行命令提供；LeRobot 0.6.1 的该汇总文件不保存种子）
- 评测方式：单环境同步运行；逐回合种子从起始值递增

## 结果

- 成功率：27/50 = 54.0%
- 成功率 95% Wilson 区间：[40.4%, 67.0%]
- 回合最高奖励均值：0.952
- 回合最高奖励范围：0.424–1.000
- 平均累计奖励：116.083

## 解读限制

- 奖励与成功标志来自仿真环境；原因分类需要人工查看视频。
- 视频只覆盖已保存的回合；空白的 `failure_reason` 不代表没有失败。
- 此结果只描述该策略在这组仿真配置与随机种子下的表现。

逐回合数据见 `diffusion_official_deterministic_50_episodes.csv`。
