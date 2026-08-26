# src/synthesis — the synthetic data generator

Generates Sudanese-Arabic training data (WhatsApp-style chats + discourse monologues),
anchored on the real corpus and the 42 persona cards.

## The one command

```bash
uv run python -m src.synthesis.synth_data gemma3 1000 gemma4 1000 sonnet 1000
```

Each `MODEL N` pair samples N fresh random seeds for that model's task and generates.
The model→task mapping lives in `MODEL_REGISTRY` (`synth_data.py`):

| model | task | backend |
|---|---|---|
| `sonnet` / `opus` / `haiku` | chat | Claude subscription (`claude -p`) |
| `gemma3` / `gemma4` | monologue | Ollama (local GPUs) |
| `jais2-8b` | chat | Ollama (local GPUs) |

Adding a model = one registry entry (task, backend, tag, concurrency).

When generation finishes, the QC chain runs automatically (filter → Haiku judge → render)
plus the per-generator diversity report. `--skip-qc` generates only.

**Outputs**
- `data/interim/synthetic/raw/sd_<runstamp>_<model>_<i>.json` — one per generation, with
  the generator and the full seed config (participants, topic source, situation id, writer,
  anchor source, prompt version)
- `data/interim/synthetic/{chats_me,chats_pseudo,monologue}.jsonl` — QC-passed training
  files (the two chat arms differ only in the owner's speaker label), each row carrying
  generator + seed provenance
- `data/interim/synthetic/diversity_report.json` — per-generator diversity metrics vs
  real-corpus reference rows

**Notes**
- Sampling is stateless by design: no done-tracking; every invocation is a fresh draw.
- Local generation and GPU training can't run simultaneously (same VRAM) — stop one first.
- Usage-limit hits pause all Claude workers 15 min and resume automatically.

## Module map

| module | what it does |
|---|---|
| `synth_data.py` | the CLI: registry, seed samplers, backends, auto-QC |
| `situations.py` | builds the situation bank (2,005 two-sentence situations, offline, Verbalized Sampling) — rebuild with `... situations build` |
| `prompts.py` | all prompt templates, versioned (`PROMPT_VERSION`) |
| `seed_sampler.py` | real-corpus access: excerpts, mention-graph pairs, group rosters, measured topics; enforces val-chat + blocklist exclusions |
| `persona_cards.py` | compiles the 42+1 persona cards from the PKB (one-time; cards live in `data/interim/synthetic/cards/`) |
| `qc.py` | filter → judge → render chain (subcommands `filter` / `judge` / `render`) |
| `diversity.py` | per-generator diversity metrics (`... diversity report`) |
| `bakeoff.py` | the local-model evaluation harness (kept for future model auditions) |
| `blocklist.py` / `pseudonyms.py` | the off-record exclusion (always on) and the retired pseudonymization layer (off; `POLICY` flag) |
| `generate.py` | pilot-era driver, superseded for generation but still hosts the persona-card build (`plan-cards` / `run` / `collect`) |

## Prerequisites

Persona cards compiled, situation bank built, dialect classifier trained
(`data/interim/dialect_clf.joblib`), interim corpora present (`whatsapp`, `oddadmix`,
`sudaneseonline`), Ollama serving the local models, `claude` CLI logged in.
