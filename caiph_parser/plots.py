"""Simple, non-expert-readable charts. Single hue, thin marks, no legends
needed (one series per panel; the title names it)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .classify import DPICS

BLUE = "#2a78d6"
LIGHT = "#9ec5f4"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"

PRETTY = {k: k.replace("_", " ") for k in DPICS}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def dpics_bars(summary: pd.DataFrame, path: Path) -> None:
    """Mean % of caregiver utterances per DPICS class, with per-transcript dots."""
    cols = [f"pct_{k}" for k in DPICS]
    means = summary[cols].mean().values
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = range(len(DPICS))
    ax.barh(y, means, color=BLUE, height=0.55)
    for i, c in enumerate(cols):
        ax.scatter(summary[c], [i] * len(summary), s=14, color=INK2, alpha=0.5, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels([PRETTY[k] for k in DPICS])
    ax.invert_yaxis()
    ax.set_xlabel("% of caregiver utterances (bar = mean, dots = each transcript)", color=INK2)
    ax.set_title(f"DPICS distribution of caregiver speech  (n={len(summary)} transcripts)",
                 loc="left", color=INK, fontsize=11)
    for i, (v, c) in enumerate(zip(means, cols)):
        ax.text(summary[c].max() + 1.2, i, f"mean {v:.0f}%", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, summary[cols].max().max() + 12)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def do_dont(summary: pd.DataFrame, path: Path) -> None:
    """Per-transcript Do vs Don't skill share."""
    d = summary.sort_values("pct_do_skills", ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.28 * len(d) + 1.2)))
    y = range(len(d))
    ax.barh(y, d["pct_do_skills"], color=BLUE, height=0.55, label="Do (praise, reflect, describe)")
    ax.barh(y, d["pct_dont_skills"], left=d["pct_do_skills"] + 0.8, color=LIGHT,
            height=0.55, label="Don't (question, command, negative)")
    ax.set_yticks(list(y))
    ax.set_yticklabels(d["transcript"])
    ax.invert_yaxis()
    ax.set_xlabel("% of caregiver utterances", color=INK2)
    ax.set_title("PCIT Do vs Don't skills per transcript", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def turn_taking(summary: pd.DataFrame, path: Path) -> None:
    d = summary.sort_values("pct_child_initiations_answered", ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.28 * len(d) + 1.2)))
    y = range(len(d))
    ax.barh(y, d["pct_child_initiations_answered"], color=BLUE, height=0.55)
    ax.set_yticks(list(y))
    ax.set_yticklabels(d["transcript"])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of child-initiated turns answered by caregiver within 5 s", color=INK2)
    ax.set_title("Caregiver responsiveness to child initiations", loc="left", color=INK, fontsize=11)
    for i, (v, n) in enumerate(zip(d["pct_child_initiations_answered"], d["child_initiations"])):
        ax.text(v + 1, i, f"{v:.0f}%  (n={n})", va="center", fontsize=8, color=INK2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
