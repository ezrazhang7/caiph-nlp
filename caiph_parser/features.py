"""Transcript-level features. Each function returns a flat dict so the
summary table is one row per transcript. Add a feature = add a function
and register it in FEATURES.
"""
from __future__ import annotations

from collections import Counter

from .classify import DO, DONT, DPICS
from .parse_chat import Transcript, Utterance

RESPONSE_WINDOW_MS = 5000   # parent reply counts if it starts within 5 s
INITIATION_GAP_MS = 3000    # child utterance after >3 s silence = new initiation


# --- 1. DPICS distribution (grant: "% of caregiver audio in each class") ----
def dpics_distribution(t: Transcript, labels: dict[int, str]) -> dict:
    parent = [u for u in t.utterances if u.is_parent]
    n = len(parent) or 1
    c = Counter(labels.get(u.idx, "") for u in parent)
    return {f"pct_{k}": 100 * c[k] / n for k in DPICS}


# --- 2. PCIT Do/Don't summary (maps to Positive Regard / Intrusiveness) ----
def do_dont(t: Transcript, labels: dict[int, str]) -> dict:
    parent = [u for u in t.utterances if u.is_parent]
    n = len(parent) or 1
    do = sum(labels.get(u.idx) in DO for u in parent)
    dont = sum(labels.get(u.idx) in DONT for u in parent)
    return {
        "pct_do_skills": 100 * do / n,
        "pct_dont_skills": 100 * dont / n,
        "do_dont_ratio": do / dont if dont else float("inf"),
    }


# --- 3. Talk share ----------------------------------------------------------
def talk_share(t: Transcript, labels: dict[int, str]) -> dict:
    parent = [u for u in t.utterances if u.is_parent and u.text]
    child = [u for u in t.utterances if u.is_child and u.text]
    total = len(parent) + len(child) or 1
    mlu = lambda us: (sum(u.n_words for u in us) / len(us)) if us else 0.0
    return {
        "n_parent_utts": len(parent),
        "n_child_utts": len(child),
        "pct_parent_talk": 100 * len(parent) / total,
        "parent_words_per_utt": mlu(parent),
        "child_words_per_utt": mlu(child),
    }


# --- 4. Child-initiated turns that get a parent response --------------------
def turn_taking(t: Transcript, labels: dict[int, str]) -> dict:
    utts = [u for u in t.utterances if u.is_parent or u.is_child]
    initiations = responded = 0
    for i, u in enumerate(utts):
        if not u.is_child or not u.text:
            continue
        prev = utts[i - 1] if i else None
        gap = (u.start_ms - prev.end_ms) if (prev and u.start_ms and prev.end_ms) else None
        is_init = prev is None or not prev.is_child or (gap is not None and gap > INITIATION_GAP_MS)
        if not is_init:
            continue
        initiations += 1
        nxt = utts[i + 1] if i + 1 < len(utts) else None
        if nxt and nxt.is_parent:
            if nxt.start_ms is None or u.end_ms is None or nxt.start_ms - u.end_ms <= RESPONSE_WINDOW_MS:
                responded += 1
    return {
        "child_initiations": initiations,
        "pct_child_initiations_answered": 100 * responded / initiations if initiations else 0.0,
    }


FEATURES = [dpics_distribution, do_dont, talk_share, turn_taking]


def compute_all(t: Transcript, labels: dict[int, str]) -> dict:
    row = {"transcript": t.tid, "child_age": t.child_age}
    for fn in FEATURES:
        row.update(fn(t, labels))
    return row
