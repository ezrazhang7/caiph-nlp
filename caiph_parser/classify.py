"""DPICS utterance classification for caregiver speech.

Two backends:
  rules  - regex/%mor heuristics. No model. Baseline + smoke test.
  ollama - local open-weight LLM via Ollama (default: qwen3:8b), zero-shot.
           Nothing leaves the machine. Swap `model=` for any Ollama tag.

Future backend: fine-tuned RoBERTa (Nikbakhtbideh et al. 2023 recipe) once
DPICS-labeled utterances exist.
"""
from __future__ import annotations

import json
import re
from typing import Iterable

from .parse_chat import Utterance

DPICS = [
    "acknowledgment",          # "okay", "yeah", "mhm", "I see"
    "information_description", # neutral statement about things/world
    "behavior_description",    # describes what child is doing, no judgment
    "reflective_statement",    # repeats/paraphrases child's speech
    "labeled_praise",          # praise naming the behavior ("nice stacking")
    "unlabeled_praise",        # "good job", "wow", "nice"
    "question",                # any question
    "direct_command",          # imperative
    "indirect_command",        # "can you...", "let's...", "why don't you..."
    "negative_talk",           # criticism, "no", "stop", "don't", sarcasm
]

DO = {"labeled_praise", "unlabeled_praise", "reflective_statement", "behavior_description"}
DONT = {"question", "direct_command", "indirect_command", "negative_talk"}

DEFINITIONS = """\
DPICS categories for a CAREGIVER utterance spoken to a young child during play:
- acknowledgment: brief neutral response ("okay", "yeah", "mhm", "uh huh", "I see").
- information_description: neutral statement about objects, events, or the world; not about the child's current behavior.
- behavior_description: non-evaluative statement about what the child is doing right now ("you're putting the red block on top").
- reflective_statement: repeats or paraphrases what the child just said.
- labeled_praise: positive evaluation that names the specific behavior ("great job sharing", "I like how you waited").
- unlabeled_praise: positive evaluation without naming the behavior ("good job", "nice", "wow", "yay").
- question: any question, including rhetorical and tag questions ("what's this?", "right?").
- direct_command: imperative telling the child to do something ("put it here", "look", "come sit").
- indirect_command: suggestion/request phrased softly ("can you put it here?", "let's build", "why don't you try").
- negative_talk: criticism, disapproval, or correction ("no", "stop", "don't do that", "that's wrong").
If an utterance has no words (laughter, name only, unintelligible), use acknowledgment."""


# ---------------------------------------------------------------- rules ----
_ACK = re.compile(r"^(okay|ok|yeah|yes|yep|mhm|mm|uh huh|uhhuh|hm|oh|ah|alright|right|i see|sure)[.!?]?$", re.I)
_PRAISE = re.compile(r"\b(good (job|girl|boy|work)|nice|great|awesome|perfect|wow|yay|well done|excellent|beautiful|i like (how|that)|thank you|good)\b", re.I)
_NEG = re.compile(r"^(no|nope|stop|don't|do not|not like that|uh uh|that's not|wrong)\b|\b(stop it|don't|quit|no no)\b", re.I)
_INDIRECT = re.compile(r"^(can you|could you|will you|would you|let's|lets|why don't you|how about|do you want to|wanna|you want to|maybe you)\b", re.I)
_BEHAV = re.compile(r"^(you're|you are|you've|you've got|you got|you)\s+\w+ing\b", re.I)


def _rules_one(u: Utterance, prev_child_text: str | None) -> str:
    t = u.text.strip()
    if not re.search(r"[A-Za-z]", t):
        return "acknowledgment"
    if _ACK.match(t):
        return "acknowledgment"
    if _NEG.search(t):
        return "negative_talk"
    if t.endswith("?"):
        return "indirect_command" if _INDIRECT.match(t) else "question"
    if _INDIRECT.match(t):
        return "indirect_command"
    if _PRAISE.search(t):
        return "labeled_praise" if u.n_words >= 4 else "unlabeled_praise"
    if u.is_imperative or re.match(r"^(look|come|put|go|give|take|sit|try|hold|wait|open|press|push|pull|turn|say|show|make)\b", t, re.I):
        return "direct_command"
    if prev_child_text and _overlap(prev_child_text, t) >= 0.6:
        return "reflective_statement"
    if _BEHAV.match(t):
        return "behavior_description"
    return "information_description"


def _overlap(a: str, b: str) -> float:
    wa = set(re.findall(r"[a-z']+", a.lower()))
    wb = set(re.findall(r"[a-z']+", b.lower()))
    return len(wa & wb) / len(wb) if wa and wb else 0.0


def classify_rules(utts: list[Utterance]) -> dict[int, str]:
    out: dict[int, str] = {}
    last_child = None
    for u in utts:
        if u.is_child and u.text:
            last_child = u.text
        elif u.is_parent:
            out[u.idx] = _rules_one(u, last_child)
    return out


# --------------------------------------------------------------- ollama ----
def classify_ollama(utts: list[Utterance], model: str = "qwen3:8b",
                    batch: int = 25, context: int = 2) -> dict[int, str]:
    import ollama  # pip install ollama; `ollama pull qwen3:8b`

    parent = [u for u in utts if u.is_parent]
    out: dict[int, str] = {}
    for i in range(0, len(parent), batch):
        chunk = parent[i:i + batch]
        items = []
        for u in chunk:
            ctx = [x for x in utts[max(0, u.idx - context):u.idx] if x.text]
            items.append({
                "id": u.idx,
                "prior_context": [f"{x.speaker}: {x.text}" for x in ctx],
                "utterance": u.text or "(no words)",
            })
        prompt = (
            f"{DEFINITIONS}\n\nLabel each utterance below with exactly one category. "
            f"Use prior_context only to disambiguate (e.g. reflective_statement). "
            f'Return ONLY a JSON object mapping id -> category, e.g. {{"12": "question"}}.\n\n'
            f"{json.dumps(items, ensure_ascii=False)}"
        )
        resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}],
                           format="json", options={"temperature": 0})
        try:
            labels = json.loads(resp["message"]["content"])
        except json.JSONDecodeError:
            labels = {}
        for u in chunk:
            lab = str(labels.get(str(u.idx), "")).strip().lower()
            out[u.idx] = lab if lab in DPICS else _rules_one(u, None)  # fallback
    return out


def classify(utts: list[Utterance], backend: str = "rules", **kw) -> dict[int, str]:
    if backend == "rules":
        return classify_rules(utts)
    if backend == "ollama":
        return classify_ollama(utts, **kw)
    raise ValueError(f"unknown backend {backend}")
