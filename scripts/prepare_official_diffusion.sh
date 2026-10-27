#!/usr/bin/env bash
# Download lerobot/diffusion_pusht at a pinned commit and migrate it to the
# LeRobot 0.6.1 processor format. Idempotent: existing outputs are reused.
#   OUT_ROOT (default ~/pusht_eval_outputs)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$HOME/pusht_eval_outputs}"
SRC_REPO="lerobot/diffusion_pusht"
REVISION="${REVISION:-84a7c23178445c6bbf7e1a884ff497017910f653}"
RAW_DIR="$OUT_ROOT/hub/diffusion_pusht_raw"
MIGRATED_DIR="$OUT_ROOT/migrated/diffusion_pusht"
LOG_DIR="$REPO_DIR/results/logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

if [[ -f "$MIGRATED_DIR/config.json" && -f "$MIGRATED_DIR/policy_preprocessor.json" ]]; then
  echo "migrated policy already at $MIGRATED_DIR"; exit 0
fi

# Reuse a download from the earlier mpc_go2 workflow if present (it took ~45 min).
OLD_RAW="$HOME/mpc_go2_outputs/hub/diffusion_pusht_raw"
if [[ ! -f "$RAW_DIR/model.safetensors" && -f "$OLD_RAW/model.safetensors" ]]; then
  mkdir -p "$(dirname "$RAW_DIR")"; cp -r "$OLD_RAW" "$RAW_DIR"; echo "reused $OLD_RAW"
fi

if [[ ! -f "$RAW_DIR/model.safetensors" ]]; then
  python - "$SRC_REPO" "$REVISION" "$RAW_DIR" <<'PY'
import sys, json
from huggingface_hub import snapshot_download
repo, rev, out = sys.argv[1:]
snapshot_download(repo, revision=rev, local_dir=out,
                  allow_patterns=["config.json", "model.safetensors", "train_config.json", "README.md"])
json.dump({"repo": repo, "revision": rev}, open(f"{out}/source_revision.json", "w"), indent=2)
print("downloaded", repo, "@", rev)
PY
fi

rm -rf "$MIGRATED_DIR"
PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" python -m pusht_eval.migrate_legacy_policy \
  --pretrained-path "$RAW_DIR" --output-dir "$MIGRATED_DIR" 2>&1 | tee "$LOG_DIR/migrate_diffusion_pusht.log"
cp "$RAW_DIR/source_revision.json" "$MIGRATED_DIR/" 2>/dev/null || true

python - "$MIGRATED_DIR" "$LOG_DIR/migrate_diffusion_pusht.log" <<'PY'
import sys, json, glob, os
from safetensors.torch import load_file
d, log = sys.argv[1], open(sys.argv[2], encoding="utf-8").read()
for f in ["config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"]:
    assert os.path.exists(os.path.join(d, f)), f"missing {f}"
assert "unexpected missing keys" not in log and "unexpected keys" not in log, "weight keys did not match; see log"
assert "all keys matched" in log, "migration did not report a full weight match"
for f in sorted(glob.glob(os.path.join(d, "policy_*processor_step_*.safetensors"))):
    feats = {k.rsplit(".", 1)[0] for k in load_file(f)}
    assert {"observation.state", "action"} <= feats, f"{f} lacks stats"
cfg = json.load(open(os.path.join(d, "config.json")))
assert tuple(cfg["crop_shape"]) == (84, 84) and cfg["n_action_steps"] == 8, cfg
print("migration check passed:", d)
PY
