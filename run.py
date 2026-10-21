#!/usr/bin/env python3
"""Transcript -> DPICS labels -> features -> charts.

  python run.py data/raw --backend rules
  python run.py data/raw --backend ollama --model qwen3:8b --limit 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from caiph_parser.classify import DPICS, classify
from caiph_parser.features import compute_all
from caiph_parser.parse_chat import parse
from caiph_parser import plots


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="directory of .cha files (or one file)")
    ap.add_argument("--backend", default="rules", choices=["rules", "ollama"])
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--out", default="out")
    ap.add_argument("--limit", type=int, default=0, help="only first N transcripts")
    a = ap.parse_args()

    src = Path(a.src)
    files = sorted(src.glob("*.cha")) if src.is_dir() else [src]
    if a.limit:
        files = files[: a.limit]
    out = Path(a.out)
    (out / "labeled").mkdir(parents=True, exist_ok=True)

    rows = []
    for f in files:
        t = parse(f)
        kw = {"model": a.model} if a.backend == "ollama" else {}
        labels = classify(t.utterances, backend=a.backend, **kw)

        pd.DataFrame([{
            "idx": u.idx, "speaker": u.speaker, "segment": u.segment,
            "start_ms": u.start_ms, "text": u.text,
            "dpics": labels.get(u.idx, ""),
        } for u in t.utterances]).to_csv(out / "labeled" / f"{t.tid}.csv", index=False)

        row = compute_all(t, labels)
        row["backend"] = a.backend
        rows.append(row)
        print(f"{t.tid}: {row['n_parent_utts']} parent utts | "
              f"do/don't {row['pct_do_skills']:.0f}/{row['pct_dont_skills']:.0f}% | "
              f"child initiations answered {row['pct_child_initiations_answered']:.0f}%")

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    plots.dpics_bars(summary, out / "dpics_distribution.png")
    plots.do_dont(summary, out / "do_dont.png")
    plots.turn_taking(summary, out / "turn_taking.png")
    print(f"\nwrote {out}/summary.csv and charts")


if __name__ == "__main__":
    main()
