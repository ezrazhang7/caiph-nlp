"""Minimal CHAT (.cha) parser for TalkBank transcripts.

Pulls main-tier utterances (*MOT:, *CHI:, ...) with timestamps, the %mor tier,
and the @G: activity segment each utterance falls in. No external deps.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

PARENT_CODES = {"MOT", "FAT", "GRA", "GRM", "GRF", "PAR", "ADU"}
CHILD_CODES = {"CHI"}

_TS = re.compile(r"\s*(\d+)_(\d+)\s*$")          # trailing "22301_22451"
_TS_ANY = re.compile(r"\b\d+_\d+\b")             # same, anywhere (e.g. before "?")
_UNINTEL = re.compile(r"\b(xxx|yyy|www)\b")
_EVENT = re.compile(r"&=\S+|&-\S+|&\S+")          # &=laughs, &-um, &vocalizes
_BRACKET = re.compile(r"\[[^\]]*\]")               # [?], [//], [: word]
_ANGLE = re.compile(r"[<>]")
_AT = re.compile(r"@\w+")                          # vroom@o
_PUNCT_SPACE = re.compile(r"\s+([?!.,])")


@dataclass
class Utterance:
    idx: int
    speaker: str
    raw: str
    text: str                 # cleaned
    start_ms: int | None
    end_ms: int | None
    mor: str = ""
    segment: str = ""
    comments: list[str] = field(default_factory=list)

    @property
    def is_parent(self) -> bool:
        return self.speaker in PARENT_CODES

    @property
    def is_child(self) -> bool:
        return self.speaker in CHILD_CODES

    @property
    def is_imperative(self) -> bool:
        return "-Imp" in self.mor

    @property
    def n_words(self) -> int:
        return len([w for w in re.findall(r"[A-Za-z']+", self.text)])


@dataclass
class Transcript:
    path: Path
    participants: dict[str, str]
    child_age: str | None
    utterances: list[Utterance]

    @property
    def tid(self) -> str:
        return self.path.stem


def clean(raw: str) -> str:
    t = re.sub(r"[\x00-\x1f\x7f]", " ", raw)        # CHAT wraps timestamps in \x15
    t = _TS_ANY.sub(" ", t)
    t = _EVENT.sub(" ", t)
    t = _BRACKET.sub(" ", t)
    t = _ANGLE.sub("", t)
    t = _AT.sub("", t)
    t = _UNINTEL.sub(" ", t)
    t = re.sub(r"\(\.+\)", " ", t)                  # pause markers (.) (..)
    t = re.sub(r"\+\.\.\.|\+/+\.?|\+\"|\+\^|\+,", " ", t)  # trailing-off / interruption codes
    t = re.sub(r"\s+", " ", t).strip()
    t = _PUNCT_SPACE.sub(r"\1", t)
    return t.strip()


def parse(path: str | Path) -> Transcript:
    path = Path(path)
    participants: dict[str, str] = {}
    child_age = None
    utts: list[Utterance] = []
    segment = ""
    cur: Utterance | None = None

    with path.open(encoding="utf-8") as f:
        lines = f.read().splitlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        # continuation lines start with a tab
        while i + 1 < len(lines) and lines[i + 1].startswith("\t"):
            i += 1
            line += " " + lines[i].strip()
        i += 1

        if line.startswith("@Participants:"):
            for part in line.split(":", 1)[1].split(","):
                bits = part.strip().split()
                if bits:
                    participants[bits[0]] = bits[-1]
        elif line.startswith("@ID:") and "|CHI|" in line:
            child_age = line.split("|")[3] or None
        elif line.startswith("@G:"):
            segment = line.split(":", 1)[1].strip()
        elif line.startswith("*"):
            code, _, body = line[1:].partition(":")
            body = body.strip()
            m = _TS.search(body)
            s, e = (int(m.group(1)), int(m.group(2))) if m else (None, None)
            cur = Utterance(
                idx=len(utts), speaker=code.strip(), raw=body,
                text=clean(body), start_ms=s, end_ms=e, segment=segment,
            )
            utts.append(cur)
        elif line.startswith("%mor:") and cur:
            cur.mor = line.split(":", 1)[1].strip()
        elif line.startswith("%com:") and cur:
            cur.comments.append(line.split(":", 1)[1].strip())

    return Transcript(path=path, participants=participants,
                      child_age=child_age, utterances=utts)
