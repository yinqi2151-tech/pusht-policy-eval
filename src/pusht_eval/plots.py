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


def robustness_curves(table: list[dict], out: Path, title: str) -> Path:
    clean = next(s for s in table if s["condition"] == "clean")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True, facecolor=SURFACE)
    for ax, (kind, name, xlabel) in zip(axes, PANELS):
        _style(ax)
        rows = [clean] + sorted((s for s in table if s["kind"] == kind), key=lambda s: s["level"])
        xs = [0 if r is clean else r["level"] for r in rows]
        ys = [100 * r["success_rate"] for r in rows]
        # Clamp: float rounding can put a Wilson bound a hair past the estimate at 0% or 100%.
        lo = [max(0.0, 100 * (r["success_rate"] - r["ci_low"])) for r in rows]
        hi = [max(0.0, 100 * (r["ci_high"] - r["success_rate"])) for r in rows]
        ax.axhline(100 * clean["success_rate"], color=TEXT_2, linewidth=1, linestyle=(0, (3, 3)))
        ax.errorbar(xs, ys, yerr=[lo, hi], color=SERIES, linewidth=2, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=2, capsize=0, elinewidth=1.5)
        if len(rows) > 1:
            ax.annotate(f"{ys[-1]:.0f}%", (xs[-1], ys[-1]), textcoords="offset points", xytext=(9, 0),
                        ha="left", va="center", fontsize=9, color=TEXT)
        ax.margins(x=0.14)
        ax.set_title(name, fontsize=11, color=TEXT, loc="left")
        ax.set_xlabel(xlabel, fontsize=9, color=TEXT_2)
        ax.set_xticks(xs)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Success rate (%)", fontsize=9, color=TEXT_2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.text(0.01, -0.02, "Points: success rate with 95% Wilson interval. Dashed line: unperturbed.",
             fontsize=8, color=TEXT_2)
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
