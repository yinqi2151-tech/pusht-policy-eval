"""Static figures for the README (light surface, reference palette)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pusht_eval.failures import CATEGORIES  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
ERASE_COLOR = "#eb6834"  # categorical slot 2
CATEGORY_COLORS = dict(zip(CATEGORIES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
CATEGORY_NAMES = {
    "success": "Success",
    "no_push": "Never pushed block",
    "near_miss": "Near miss (≥0.85)",
    "lost_progress": "Got close, pushed away",
    "insufficient": "Pushed, little progress",
}
PANELS = [
    ("action_noise", "Action noise", "σ (workspace px)"),
    ("action_delay", "Action delay", "steps (×100 ms)"),
    ("image_noise", "Image noise", "σ (uint8 levels)"),
    ("occlusion", "Occlusion", "square side (px of 96)"),
]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _series(ax, rows, xs, color, label=None, label_pos="right"):
    ys = [100 * r["success_rate"] for r in rows]
    # Clamp: float rounding can put a Wilson bound a hair past the estimate at 0% or 100%.
    lo = [max(0.0, 100 * (r["success_rate"] - r["ci_low"])) for r in rows]
    hi = [max(0.0, 100 * (r["ci_high"] - r["success_rate"])) for r in rows]
    ax.errorbar(xs, ys, yerr=[lo, hi], color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, capsize=0, elinewidth=1.5)
    text = f"{ys[-1]:.0f}%" if label is None else f"{label} {ys[-1]:.0f}%"
    offset = (9, 0) if label_pos == "right" else (9, 11)
    ax.annotate(text, (xs[-1], ys[-1]), textcoords="offset points", xytext=offset,
                ha="left", va="center", fontsize=9, color=TEXT)


def robustness_curves(table: list[dict], out: Path, title: str) -> Path:
    clean = next(s for s in table if s["condition"] == "clean")
    # Unperturbed repeats with different policy sampling seeds (null control), if run.
    repeats = [s for s in table if s["kind"] == "none"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True, facecolor=SURFACE)
    for ax, (kind, name, xlabel) in zip(axes, PANELS):
        _style(ax)
        if len(repeats) > 1:
            lo_r = 100 * min(r["success_rate"] for r in repeats)
            hi_r = 100 * max(r["success_rate"] for r in repeats)
            ax.axhspan(lo_r, hi_r, color=GRID, alpha=0.6, linewidth=0)
        ax.axhline(100 * clean["success_rate"], color=TEXT_2, linewidth=1, linestyle=(0, (3, 3)))
        rows = [clean] + sorted((s for s in table if s["kind"] == kind), key=lambda s: s["level"])
        xs = [0 if r is clean else r["level"] for r in rows]
        if kind == "occlusion" and any(s["kind"] == "erase" for s in table):
            erase = [clean] + sorted((s for s in table if s["kind"] == "erase"), key=lambda s: s["level"])
            _series(ax, erase, [0 if r is clean else r["level"] for r in erase], ERASE_COLOR, "white", "above")
            _series(ax, rows, xs, SERIES, "gray")
            xs = sorted({*xs, *(0 if r is clean else r["level"] for r in erase)})
        else:
            _series(ax, rows, xs, SERIES)
        ax.margins(x=0.14)
        ax.set_title(name, fontsize=11, color=TEXT, loc="left")
        ax.set_xlabel(xlabel, fontsize=9, color=TEXT_2)
        ax.set_xticks(xs)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Success rate (%)", fontsize=9, color=TEXT_2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=TEXT)
    note = "Points: success rate with 95% Wilson interval. Dashed line: unperturbed."
    if len(repeats) > 1:
        note += f" Gray band: range over {len(repeats)} unperturbed runs with different policy sampling seeds."
    if any(s["kind"] == "erase" for s in table):
        note += " Occlusion: gray square (blue) vs. same square filled with the white background (orange)."
    fig.text(0.01, -0.02, note, fontsize=8, color=TEXT_2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def outcome_breakdown(table: list[dict], out: Path, title: str) -> Path:
    rows = list(reversed(table))
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(rows) + 1.4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    labels = [r["condition"] for r in rows]
    left = [0.0] * len(rows)
    for cat in CATEGORIES:
        widths = [100 * r[f"n_{cat}"] / r["n"] for r in rows]
        ax.barh(labels, widths, left=left, color=CATEGORY_COLORS[cat], label=CATEGORY_NAMES[cat],
                height=0.7, edgecolor=SURFACE, linewidth=2)
        left = [a + b for a, b in zip(left, widths)]
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of episodes (%)", fontsize=9, color=TEXT_2)
    ax.legend(ncol=5, fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), labelcolor=TEXT,
              handlelength=1.2, columnspacing=1.2)
    ax.set_title(title, loc="left", fontsize=12, color=TEXT, pad=26)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out
