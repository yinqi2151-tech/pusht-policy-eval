"""Draw a blind, stratified sample of episodes and render replay videos.

    python scripts/make_review_sample.py [--per-category 8] [--seed 0]

Writes results/review/videos/R##.mp4, results/review/review_sheet.csv (what the
reviewer fills in: human_category + notes) and results/review/review_key.csv
(condition, seed and automatic category; do not open before labelling).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pusht_eval.analysis import load_sweep, load_trajectories  # noqa: E402
from pusht_eval.failures import CATEGORIES  # noqa: E402
from pusht_eval.perturbations import PerturbationSpec  # noqa: E402
from pusht_eval.review import stratified_sample, write_replay  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=REPO / "results" / "sweep")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "review")
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sheet_path, key_path = args.out / "review_sheet.csv", args.out / "review_key.csv"
    if sheet_path.exists():
        sys.exit(f"{sheet_path} exists; delete it deliberately before drawing a new sample")
    runs = load_sweep(args.results)
    rows = [r for rs in runs.values() for r in rs if "category" in r]
    sample = stratified_sample(rows, args.per_category, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    trajs = {}
    sheet, key = [], []
    for i, row in enumerate(sample, 1):
        rid = f"R{i:02d}"
        cond = row["condition"]
        if cond not in trajs:
            trajs[cond] = load_trajectories(args.results / cond)
        spec = PerturbationSpec(row["kind"], row["level"])
        video = write_replay(trajs[cond][row["seed"]], args.out / "videos" / f"{rid}.mp4", spec)
        sheet.append({"review_id": rid, "video": video.relative_to(args.out).as_posix(), "human_category": "", "notes": ""})
        key.append({"review_id": rid, "condition": cond, "seed": row["seed"], "auto_category": row["category"]})
        print(f"{rid}: rendered", flush=True)
    for path, data in ((sheet_path, sheet), (key_path, key)):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(f"\n{len(sheet)} videos. Label each with one of {', '.join(CATEGORIES)} in {sheet_path}")
    print("(rules: docs/failure_taxonomy.md). Then run scripts/review_agreement.py.")


if __name__ == "__main__":
    main()
