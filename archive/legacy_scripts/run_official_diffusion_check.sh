#!/usr/bin/env bash
# 用官方 lerobot/diffusion_pusht 权重校验 PushT 评测流程（LeRobot 0.6.1）。
# 在 WSL 的 (lerobot) 环境中运行：
#   conda activate lerobot
#   bash /mnt/c/Users/qyq71/Desktop/mpc_go2/scripts/run_official_diffusion_check.sh
# 可选环境变量：N_EPISODES（默认 50）、SEED（默认 2000）、OUT_ROOT（默认 ~/mpc_go2_outputs）
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N_EPISODES="${N_EPISODES:-50}"
SEED="${SEED:-2000}"
OUT_ROOT="${OUT_ROOT:-$HOME/mpc_go2_outputs}"
SRC_REPO="lerobot/diffusion_pusht"
RAW_DIR="$OUT_ROOT/hub/diffusion_pusht_raw"
MIGRATED_DIR="$OUT_ROOT/migrated/diffusion_pusht"
EVAL_DIR="$OUT_ROOT/eval/diffusion_official_deterministic_${N_EPISODES}"
TAG="diffusion_official_deterministic_${N_EPISODES}"
LOG_DIR="$REPO_DIR/results/logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

echo "== 0. 环境检查"
python - <<'PY'
import importlib.metadata as m, torch
print("lerobot", m.version("lerobot"), "| gym_pusht", m.version("gym_pusht"), "| torch", torch.__version__)
assert torch.cuda.is_available(), "CUDA 不可用：请确认在 WSL 的 (lerobot) 环境里运行"
print("GPU:", torch.cuda.get_device_name(0))
PY

echo "== 1. 下载官方权重到本地（固定版本，记录 commit；已下载则跳过）"
if [[ -f "$RAW_DIR/model.safetensors" && -f "$RAW_DIR/config.json" && -f "$RAW_DIR/source_revision.json" ]]; then
  echo "已存在 $RAW_DIR，跳过下载：$(cat "$RAW_DIR/source_revision.json" | tr -d '\n ')"
else
python - "$SRC_REPO" "$RAW_DIR" <<'PY2'
import sys, json
from huggingface_hub import snapshot_download, HfApi
repo, out = sys.argv[1], sys.argv[2]
sha = HfApi().model_info(repo).sha
snapshot_download(repo, revision=sha, local_dir=out,
                  allow_patterns=["config.json", "model.safetensors", "train_config.json", "eval_info.json", "README.md"])
json.dump({"repo": repo, "revision": sha}, open(f"{out}/source_revision.json", "w"), indent=2)
print("downloaded", repo, "@", sha)
PY2
fi

echo "== 2. 迁移到新版处理器格式（LeRobot 自带脚本 + 列表转元组修复，见 scripts/migrate_legacy_policy.py）"
rm -rf "$MIGRATED_DIR"
python "$REPO_DIR/scripts/migrate_legacy_policy.py" \
  --pretrained-path "$RAW_DIR" --output-dir "$MIGRATED_DIR" 2>&1 | tee "$LOG_DIR/${TAG}_migrate.log"
cp "$RAW_DIR/source_revision.json" "$MIGRATED_DIR/"

echo "== 3. 迁移结果检查"
python - "$MIGRATED_DIR" "$LOG_DIR/${TAG}_migrate.log" <<'PY'
import sys, json, glob, os
d, log = sys.argv[1], open(sys.argv[2], encoding="utf-8").read()
need = ["config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"]
missing = [f for f in need if not os.path.exists(os.path.join(d, f))]
assert not missing, f"迁移输出缺文件: {missing}"
assert glob.glob(os.path.join(d, "policy_preprocessor_step_*normalizer*.safetensors")), "缺归一化统计文件"
assert "unexpected missing keys" not in log, "迁移日志出现缺失权重，见 migrate.log"
assert "unexpected keys" not in log, "迁移日志出现多余权重，见 migrate.log"
from safetensors.torch import load_file
for f in sorted(glob.glob(os.path.join(d, "policy_*processor_step_*.safetensors"))):
    st = load_file(f)
    print(os.path.basename(f), sorted(st.keys()))
    feats = {k.rsplit(".", 1)[0] for k in st}
    for k in ("observation.state", "action"):
        assert k in feats, f"{os.path.basename(f)} 缺 {k} 的统计量"
cfg = json.load(open(os.path.join(d, "config.json")))
print("config:", {k: cfg.get(k) for k in ["type", "n_obs_steps", "horizon", "n_action_steps", "crop_shape", "normalization_mapping"]})
print("迁移检查通过")
PY

echo "== 4. 确定性协议评测：$N_EPISODES 回合，种子 $SEED 起（约 15–30 分钟）"
rm -rf "$EVAL_DIR"
python "$REPO_DIR/scripts/eval_pusht_deterministic.py" \
  --policy.path="$MIGRATED_DIR" \
  --policy.device=cuda \
  --env.type=pusht \
  --eval.n_episodes="$N_EPISODES" \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --seed="$SEED" \
  --output_dir="$EVAL_DIR" 2>&1 | tee "$LOG_DIR/${TAG}_eval.log"

echo "== 5. 汇总并写回项目"
cp "$EVAL_DIR/eval_info.json" "$REPO_DIR/results/${TAG}.json"
cp "$MIGRATED_DIR/source_revision.json" "$REPO_DIR/results/${TAG}_source_revision.json"
cd "$REPO_DIR"
python scripts/summarize_policy_eval.py "results/${TAG}.json" \
  --policy "$SRC_REPO (migrated, $(python -c "import json;print(json.load(open('results/${TAG}_source_revision.json'))['revision'][:8])"))" \
  --seed "$SEED" --n-action-steps 8
python - "results/${TAG}.json" <<'PY'
import sys, json
g = json.load(open(sys.argv[1]))["per_group"]["pusht"]
print(f"\n成功率 {g['pc_success']:.1f}%  | 平均回合最高奖励 {g['avg_max_reward']:.3f}  | 回合数 {g['n_episodes']}")
print("参考：官方模型卡 500 回合 65.4% / 0.955。若远低于此，说明评测流程或迁移仍有问题。")
PY
echo "完成。结果: results/${TAG}.json 及同名 _episodes.csv / _summary.md；日志在 results/logs/"
