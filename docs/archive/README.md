# 早期报告存档

这里是项目还在 `go2-mpc-mujoco` 仓库时期写的报告，原样保存，内容没有改动。

- 文中提到的 `scripts/...`、`results/...` 指的是旧仓库中的路径。对应的代码在本仓库的 `archive/legacy_scripts/` 下，原始结果在 `results/archive/initial_checkpoints/` 下。
- 报告中的结论有一部分后来被推翻或修正，最新的结论以 [../history.md](../history.md) 和 [../validation.md](../validation.md) 为准。主要有两处：
  1. 首批 ACT 消融得到的 +0.164 提升，在独立种子和确定性设置下都没有复现。
  2. `DanqingZ/diffusion_pusht` 的 0/20，原因是这份权重只训练了 1000 步，不能代表 Diffusion Policy 这个算法的水平。
