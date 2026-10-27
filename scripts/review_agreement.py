"""Agreement between human labels and the automatic outcome categories.

    python scripts/review_agreement.py

Reads results/review/review_sheet.csv (human) and review_key.csv (auto), writes
results/review/agreement.json and prints accuracy, Cohen's kappa and the
confusion matrix (rows: automatic category, columns: human label).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pusht_eval.failures import CATEGORIES  # noqa: E402
from pusht_eval.review import agreement  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--review", type=Path, default=REPO / "results" / "review")
    args = parser.parse_args()
    result = agreement(args.review / "review_sheet.csv", args.review / "review_key.csv")
    (args.review / "agreement.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"labelled {result['n_labeled']}/{result['n_sampled']}; accuracy {result['accuracy']:.2f}; "
          f"Cohen's kappa {result['cohen_kappa']:.2f}")
    width = max(len(c) for c in CATEGORIES) + 2
    print("auto \\ human".ljust(width) + "".join(c[:10].rjust(12) for c in CATEGORIES))
    for c in CATEGORIES:
        print(c.ljust(width) + "".join(str(result["confusion_auto_rows_human_cols"][c][h]).rjust(12) for h in CATEGORIES))


if __name__ == "__main__":
    main()
