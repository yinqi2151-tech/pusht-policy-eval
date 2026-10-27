#!/usr/bin/env bash
# Full pipeline in the WSL (lerobot) environment. Every step is resumable.
#   bash scripts/run_all.sh                 # everything
#   SKIP_CHECK=1 bash scripts/run_all.sh    # skip the determinism/speed check
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OUT_ROOT="${OUT_ROOT:-$HOME/pusht_eval_outputs}"
CONFIG="${CONFIG:-$REPO_DIR/configs/sweep_b.json}"
BATCH="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['batch_size'])" "$CONFIG")"
cd "$REPO_DIR"

python -c "import lerobot, gym_pusht, av, matplotlib" 2>/dev/null || {
  echo "missing packages: run inside the (lerobot) env and 'pip install matplotlib'"; exit 1; }
[[ "$(python -c 'import importlib.metadata as m; print(m.version("lerobot"))')" == "0.6.1" ]] || {
  echo "expected lerobot 0.6.1 (the protocol and the migration fix are pinned to it)"; exit 1; }

echo "== 1/5 prepare official Diffusion Policy weights"
bash scripts/prepare_official_diffusion.sh

if [[ -z "${SKIP_CHECK:-}" && ! -f "results/checks/determinism_batch${BATCH}.json" ]]; then
  echo "== 2/5 batch determinism + speed check (2 x $BATCH episodes at batch $BATCH, 5 at batch 1)"
  python scripts/check_determinism.py --policy-dir "$OUT_ROOT/migrated/diffusion_pusht" \
    --batch-size "$BATCH" --n-episodes "$BATCH"
else
  echo "== 2/5 determinism check: skipped (already done or SKIP_CHECK set)"
fi

echo "== 3/5 robustness sweep ($CONFIG)"
python scripts/run_sweep.py "$CONFIG"

echo "== 4/5 analysis"
python scripts/analyze_sweep.py

echo "== 5/5 blind review sample"
if [[ -f results/review/review_sheet.csv ]]; then
  echo "review sample already drawn"
else
  python scripts/make_review_sample.py
fi
echo "done. Next: label results/review/review_sheet.csv, then python scripts/review_agreement.py"
