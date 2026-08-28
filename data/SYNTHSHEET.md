# Synthsheet — sudani_lm

> Paths in this file are relative to the repo root.

`DATASHEET.md` (alongside this file) is the single self-contained record of **every dataset** in this repo —
acquired, synthetic, used, and planned. This file does not duplicate it. This file documents
**how the synthetic data is produced**: the method, the models, and the kinds of documents
generated. When a synthetic corpus enters a training mixture, its row goes in DATASHEET; the
recipe that made it stays here.

Code: `src/synthesis/` (module map in `src/synthesis/README.md`). Research write-up and full
experimental record: `src/synthesis/report.md`. Plan of record: `plan.md` Part IV.

---

## 1. Why synthesise at all

Everything in `data/raw/` that is genuinely Sudanese is small. Before acquisition wave 3 the
Sudanese slice was ~11.1M tokens against 33.9B tokens of MSA web text — 0.03% of the corpus.
Sudanese Arabic is a spoken dialect that mostly exists as private chat, so there is no large
public corpus to buy or download; the acquisition track (see `SCRAPESHEET.md`) raises the
ceiling but cannot produce **conversational** dialect in the owner's register at scale.

Synthesis targets exactly that gap: multi-party WhatsApp-style Sudanese chat and first-person
Sudanese discourse, anchored on real corpus material so the generators imitate a measured
distribution rather than their own idea of "Arabic".

## 2. What is synthesised

| Output | File (`data/interim/synthetic/`) | What it is | Loss treatment downstream |
|---|---|---|---|
| **Chat — owner-labelled** | `chats_me.jsonl` | Multi-party Sudanese chat, owner's turns labelled `ME` | Stage C full-token; Stage D masked to owner turns |
| **Chat — pseudonym arm** | `chats_pseudo.jsonl` | The *same* conversations with the owner under his name | The other arm of the labelling ablation |
| **Monologue** | `monologue.jsonl` | First-person Sudanese discourse over a genre × audience grid | Stage C full-token |
| **Monologue (local-only pool)** | `monologue_gemma.jsonl` | Gemma-generated subset kept separable for the §4.4 ablation | as above |
| **MSA→Sudanese transform** | `transformed.jsonl` | **Retired.** Pilot killed 89/120 on structural QC (lazy MSA passthrough); only 39% of survivors cleared the dialect judge | Not generated in production |

Every row carries its generator and full seed provenance (participants, topic source,
situation id, writer persona, anchor source, prompt version).

## 3. Seed design — what the generators are anchored on

Unanchored generation produces mode-collapsed pseudo-dialect. Every request is therefore built
from measured real material (`seed_sampler.py`, `situations.py`, `persona_cards.py`):

- **Persona cards** — 42+1 cards distilled from the private PKB, one per real interlocutor.
- **Situation bank** — 2,005 two-sentence situations, built offline with Verbalized Sampling.
- **Party count** — drawn from the *empirical* distribution of real conversations (capped at 5).
- **Participants** — from the real mention graph and real group rosters.
- **Topic** — 70% from the pair's measured topic distribution, 30% from the situation bank.
- **Style anchor** — a rotated real excerpt, chosen **independently of the topic**, so it
  supplies register without bleeding content.
- **Imbalance** — ≥3-party chats carry a participation-imbalance instruction sampled from the
  real group's turn shares.
- **Monologues** — writer-persona (50%, including the owner) or plain-anchor voice; anchor drawn
  from the high-dialect podcast/forum pool; genre × audience grid.

Exclusions are enforced at sampling time: validation chats and the `blocklist.py` off-record set
can never become seeds.

Prompts are versioned (`prompts.py`, `PROMPT_VERSION`); the version is recorded per generation.

## 4. Models and backends

`MODEL_REGISTRY` in `synth_data.py`. Adding a model is one registry entry.

| Key | Task | Backend | How it runs | Concurrency |
|---|---|---|---|---|
| `sonnet` | chat | Claude | owner's Claude subscription via headless `claude -p` (not the paid API) | 8 |
| `opus` | chat | Claude | as above | 8 |
| `haiku` | chat + **judge** | Claude | as above; also the dialect judge in `qc.py` | 8 |
| `gpt-5.6-terra` | chat | Codex CLI | owner's ChatGPT account, `codex exec`, read-only sandbox | 4 |
| `gpt-5.5` | chat | Codex CLI | as above | 4 |
| `gemma3` | monologue | Ollama | `gemma3:27b`, local GPUs | 2 |
| `gemma4` | monologue | Ollama | `gemma4:12b`, local GPUs | 2 |
| `jais2-8b` | chat | Ollama | local GPUs | 2 |

Sampling for local models: `temperature 0.8, top_p 0.95, num_ctx 8192, num_predict 2048`.
Claude usage-limit hits pause all Claude workers 15 min and resume automatically. Local
generation and GPU training cannot run at once — same VRAM.

**Why this line-up** (measured, `report.md` §4.3, seven local models × 198 identical prompts):

| model | structural pass | judge ≥4 (survivors) | end-to-end | chats | monologues |
|---|---|---|---|---|---|
| gemma3:27b | 71% | 96% | **69%** | 41% | **96%** |
| gemma4:12b | 50% | 97% | **48%** | 0% | **96%** |
| qwen3.8 | 57% | 60% | 34% | 12% | 55% |
| jais2-8b | 48% | 56% | 27% | 0% | 54% |
| llama3.3:70b | 13% | 56% | 7% | 14% | 0% |
| fanar2-27b | 13% | 23% | 3% | 3% | 3% |
| allam-7b | 14% | 19% | 2.5% | 5% | 0% |
| *Claude (ceiling)* | *85%* | *97.5%* | *~83%* | *~82%* | *~85%* |

The split follows directly: **local models write monologues, frontier models write chats.** No
local model can hold a multi-party Sudanese conversation together (best chat rate 41%), while
Gemma monologues clear the judge at 96%. Codex audition (§4.5) put `gpt-5.6-terra` at 82% and
`gpt-5.5` at 81% end-to-end against Sonnet's 82% reference — so both were added as chat
generators on a second free subscription.

## 5. The pipeline

```
seeds (real corpus) → versioned prompt → backend (claude -p / codex exec / ollama)
   → raw/<id>.json  (full seed config + generator recorded)
   → QC filter  → Haiku dialect judge → render → training jsonl
   → diversity report
```

One command; each `MODEL N` pair samples N fresh random seeds for that model's task:

```bash
uv run python -m src.synthesis.synth_data gemma3 1000 gemma4 1000 sonnet 1000
uv run python -m src.synthesis.qc all            # filter → judge → render (on demand)
uv run python -m src.synthesis.diversity report
```

Sampling is **stateless by design** — every invocation draws a fresh entropy seed, seed configs
practically never repeat, so there is no done-tracking; the run stamp in each id keeps
invocations distinct. QC is **fully incremental** and keyed by id, so it is safe to run
mid-generation and pays only for unseen documents.

## 6. QC harness

`qc.py`, cheapest-first, every filter logs its kill count (a filter that never fires is as
informative as one that always fires):

1. **Format validity** — chat parses into turns, expected speakers (lenient, canonicalizing
   name matching), 8–120 turns.
2. **Degeneration** — repeated n-grams, line repeats, compression ratio.
3. **Real-name scan** — no-op under the real-names policy; active if pseudonymisation returns.
4. **Leakage** — 8-gram overlap vs WhatsApp val chats and Flores DEVTEST → hard reject;
   8-gram overlap vs the request's *own seed* → regurgitation reject.
5. **Near-duplicate dedup** across the kept pool (shingle Jaccard).
6. **Dialect judge** — `claude -p` (Haiku) against a 1–5 Sudanese-authenticity rubric where
   5 = "as a Sudanese would write it on WhatsApp" and 2 = "MSA in disguise"; **keep ≥ 4**.

Rejects land in `kills.jsonl` with the reason; scores in `judged.jsonl`.

**Diversity** (`diversity.py`) reports per-generator compression ratio, self-similarity and
embedding nearest-neighbour distance against real-corpus reference rows.

## 7. Privacy and policy

- Generation runs on the owner's own subscriptions; prompts are written to
  `requests/<id>.md` **before** they are sent, so every payload is reviewable.
- `blocklist.py` (off-record exclusion) is **always on**.
- `pseudonyms.py` is the retired pseudonymisation layer, kept behind a `POLICY` flag
  (owner decision 2026-08-23: real person details throughout).
- The real WhatsApp corpus is private, is never in the repo, and is never redistributed;
  synthetic output is checked against the held-out slice for leakage.
- Everything under `data/` except DATASHEET is gitignored.

## 8. Production state — measured 2026-08-28

| Quantity | Value |
|---|---|
| Raw generations on disk (`raw/*.json`) | **5,271** |
| By generator | gemma4 2,468 · gemma3 1,429 · mono_v2 360 · group_v2 112 · trans_v2 90 · gpt-5.5 64 · sonnet 62 · mono 57 · trans 30 · gpt-5.6-terra 17 |
| Passed structural filter (`filtered.jsonl`) | **2,119** (27.6 MB) |
| Killed (`kills.jsonl`) | **353** |
| Judged (`judged.jsonl`) | **2,119** |
| Rendered chats (each arm) | **684** conversations (`chats_me` 1.75 MB / `chats_pseudo` 1.79 MB) |
| Rendered monologues | **1,229** (10.1 MB; `monologue_gemma.jsonl` 7.0 MB of it) |
| Situation bank | **2,005** situations |
| Persona cards | 42+1 |

Pilot result for reference (`report.md` §4.2): ~1M kept tokens, 85% structural pass, 97.5% of
survivors judged ≥4. Gemma-only training ablation (§4.4) passed its gate: 813 QC-passed Gemma
monologues (1.12M tokens, 4.6% of a 25.4M-token mixture) gave a ~1% Flores-Sudanese improvement
with no MMLU regression — **safe, mild positive**.

## 9. Where the output goes

Synthetic files enter training only through a mixture manifest (`configs/mixtures/*.yaml`),
which records per-source token counts in the pack's `meta.json` — same rule as every acquired
source. Chats feed Stage C (all of it) and Stage D (owner-heavy, **masked** to the owner's
replies); monologues feed Stage C. The `chats_me` / `chats_pseudo` pair exists to run the
owner-labelling ablation, not to be used together.

## 10. Prerequisites

Persona cards compiled, situation bank built, dialect classifier trained
(`data/interim/dialect_clf.joblib`), interim corpora present (`whatsapp`, `oddadmix`,
`sudaneseonline`), Ollama serving the local models, `claude` CLI logged in.

---

Keep this file current when a generator, task, prompt version, or QC gate changes. Dataset
rows, token counts and mixture roles belong in `DATASHEET.md`.
