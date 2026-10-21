# caiph-nlp — linguistic features from parent–child transcripts

Proof of concept for the linguistic-content arm of Aim 2 (C5a) of the CAIPH / REACH
NIH proposal. Takes speaker-tagged transcripts, labels every caregiver utterance with a
DPICS category, and reduces each transcript to a short row of interpretable numbers.

Developed on the public **NYU-Emerson** corpus (ASDBank, 30 parent–child play
sessions with autistic children, English, recorded at home). Same pipeline runs
unchanged on REACH transcripts once Ani exports them.

## Quick start

```bash
pip install -r requirements.txt
python run.py data/raw --backend rules            # no model, runs anywhere, ~5 s
python run.py data/raw --backend ollama --limit 3 # local LLM, see "Backends"
```

Outputs in `out/`:
- `labeled/<id>.csv` — every utterance with speaker, timestamp, segment, DPICS label
- `summary.csv` — one row per transcript, all features below
- `dpics_distribution.png`, `do_dont.png`, `turn_taking.png`

## Features

Each is a function in `caiph_parser/features.py`. Add one = write a function returning a
dict and append it to `FEATURES`.

| # | Feature | Source | Status | Maps to (grant Table 2) |
|---|---|---|---|---|
| 1 | **DPICS distribution** — % of caregiver utterances in each of 10 classes: acknowledgment, information description, behavior description, reflective statement, labeled praise, unlabeled praise, question, direct command, indirect command, negative talk | Grant C5a, "Linguistic feature extraction" (mimics Nikbakhtbideh et al. 2023) | done | Positive Regard, Sensitivity, Intrusiveness |
| 2 | **PCIT Do / Don't share** — Do = praise + reflection + behavior description; Don't = questions + commands + negative talk; ratio | Standard DPICS summary; added | done | Positive Regard (Do), Intrusiveness (Don't) |
| 3 | **Talk share** — utterance counts, % caregiver talk, words per utterance per speaker | Added | done | Context for 1–2; linguistic "elaboration" |
| 4 | **Child initiations answered** — % of child-initiated turns the caregiver responds to within 5 s | Grant lists under HARMONI ("child-initiated conversations that become turns"); cheap from transcripts | done | Sensitivity / responsiveness |
| 5 | Per-segment breakdown (free play vs. pretend play vs. puzzles) — `@G:` tags are already parsed | — | not started | Lets Iheoma compare tasks |
| 6 | Contingent praise (praise within N s of a child action/utterance) | — | idea | Positive Regard |
| 7 | Caregiver vocabulary diversity (type-token ratio) | — | idea | Cognitive stimulation |

Not included on purpose: generic positive/neutral/negative sentiment. It maps to
nothing in Table 2 and the grant text specifies DPICS.

## Backends (`caiph_parser/classify.py`)

| Backend | What it is | Use for |
|---|---|---|
| `rules` | Regex + CHAT `%mor` tags (imperative verbs → commands, `?` → question, praise/negation word lists, lexical overlap with the child's last utterance → reflection) | Smoke tests, baseline, anything without a GPU |
| `ollama` | Zero-shot local LLM. Prompt = DPICS definitions + batch of 25 utterances with 2 lines of prior context, JSON out, temperature 0. Default model `qwen3:8b`; any Ollama tag works. Falls back to rules on a bad label. | The actual demo. Nothing leaves the machine. |
| `roberta` | **Planned.** Fine-tune `roberta-base` on DPICS-labeled utterances (the recipe in Nikbakhtbideh et al. 2023 — 72–80% accuracy on their data). | Once we have ~500+ labeled utterances. Faster and more reproducible than an LLM. |

Rules-backend numbers on NYU-Emerson (n=30, caregiver utterances): questions 25%,
information 33%, direct commands 14%, acknowledgment 14%, praise 3%, negative 3%.
Do/Don't ≈ 5% / 48%. 75% of child initiations answered. Treat as a sanity check, not a
result — behavior description and reflection are badly under-detected by regex, which
is exactly what the LLM backend is for.

## Tech stack

- **Python 3.11+**, `pandas`, `matplotlib`. Parser has no dependencies.
- **Input**: TalkBank CHAT (`.cha`). `caiph_parser/parse_chat.py` reads `*SPK:` tiers,
  `\x15start_end\x15` ms timestamps, `%mor` grammar, `%com` comments, `@G:` segments.
  Caregiver codes: MOT/FAT/GRA/GRM/PAR; child: CHI. Investigator (INV) lines are kept
  but excluded from all features. REACH transcripts from Ani just need to land in the
  same `*MOT:` / `*CHI:` shape (or a CSV with speaker, text, start_ms — trivial to add a
  reader).
- **Model runtime**: [Ollama](https://ollama.com). `ollama pull qwen3:8b` (≈5 GB;
  `qwen3:4b` if RAM is tight). Qwen3 weights are Apache-2.0, which is the cleanest
  license line for a methods section. Llama 3.x / Gemma 3 are drop-in alternatives.
- **Compute**: laptop for public data. **Longleaf** (UNC, SLURM, GPU partitions) for
  REACH data so transcripts never leave the closed environment — install Ollama in user
  space or use `transformers` with weights in scratch. Longleaf, not Sycamore: these
  are many independent single-node jobs, not one multi-node MPI job.
- **Privacy**: no API calls. The LLM runs locally; `pip` is the only network access.

## Layout

```
run.py                 CLI: parse → classify → features → charts
caiph_parser/
  parse_chat.py        CHAT parser → Transcript / Utterance
  classify.py          DPICS taxonomy, rules + ollama backends
  features.py          one function per feature, FEATURES registry
  plots.py             three charts, single hue, non-expert readable
data/raw/              .cha files (NYU-Emerson; not committed)
out/                   generated
```

## Citation

Steen, K., Buonocore, T., Luyster, R., Sancimino, C., & Arunachalam, S. (2026).
Introducing the NYU-Emerson Corpus. Meeting on Language in Autism. (required by TalkBank)

Nikbakhtbideh, B., Duffett-Leger, L., Moshirpour, M. (2023). Assessment of
Parent–Child Interaction Quality from Dyadic Dialogue. *Applied Sciences*.
