# Synthetic Sudanese-Arabic Training Data: Research, Method, and Results

*sudani_lm project — synthesis track. Covers work from 2026-08-22 to 2026-08-27.*

---

## Abstract

The sudani_lm project trains a small Sudanese-Arabic chat language model from scratch. Its
binding constraint is data: general (Modern Standard) Arabic is abundant, but the entire
public supply of Sudanese-dialect *text* is on the order of a few hundred thousand tokens,
and multi-turn conversational Sudanese — the register the model is actually for — does not
exist publicly at all. This report documents the program that closed that gap: a synthetic
data pipeline that generates Sudanese WhatsApp-style conversations and discourse-length
monologues, anchored on the owner's real chat corpus and a set of 42 richly documented
personas of his actual contacts.

The headline results: (1) a 1M-token pilot passed a two-axis quality harness with 97.5% of
outputs judged authentically Sudanese, and adding acquired-plus-synthetic data to training
improved every Sudanese metric of the 110M-parameter model with zero MSA regression;
(2) a controlled seven-model bake-off of locally-runnable open models found that no open
model can hold a multi-turn Sudanese conversation, but two Gemma models generate Sudanese
*monologues* at 96% usable — above the Claude reference itself — enabling a hybrid design
where roughly half of all generation runs free on local GPUs; (3) a research-grounded seed
redesign replaced flat topics with a bank of 2,000 machine-proposed "situations", empirical
party-count sampling, and per-generator diversity instrumentation. Resources: one Claude Max
(20×) subscription and a dual-RTX-5090 workstation; no paid API usage.

---

## 1. Background and literature review

### 1.1 Why synthesis is necessary at all

A survey of the public data landscape (2026-08) found:

- **Public Sudanese text is minuscule.** The best clean source, Lisan-Sudanese, is 47K
  tokens (CC-BY). The arbml tweet corpora add ~250K noisy tokens; Sudanese_Flores provides
  2,009 MSA↔Sudanese parallel sentences (kept as evaluation, never trained on). The largest
  claimed corpus (SudSenti, "~6.5M tokens") turned out to be a 572KB release — the claim did
  not survive measurement.
- **The two large real sources are non-conversational.** ~200 hours of transcribed Sudanese
  podcasts (~2.5M tokens after cleaning) and the sudaneseonline.com forum archive (~243M
  tokens crawled, of which ~20M are dialect-dense) supply spoken and written *discourse*,
  not chat.
- **Multi-turn conversational Sudanese does not exist publicly.** No dataset, no benchmark:
  Sudan is absent from AraDiCE, DialectalArabicMMLU, and the AMIYA dialect-generation shared
  task (VarDial 2026). No Sudanese generative text model or LoRA exists on HuggingFace.

### 1.2 What is known about LLMs and Arabic dialects

- **AL-QASIDA** (arXiv 2412.04193): frontier LLMs score below 50% dialect fidelity when
  asked to *author* Arabic dialects cold. The diagnosis is *reluctance*, not inability —
  and few-shot anchoring with real dialect text measurably helps. This is the single
  finding the whole pipeline is built on: **never generate cold; always anchor on real
  Sudanese text.**
- **ArabCulture-Dialogue** (arXiv 2605.00119, the only benchmark including Sudan): open
  7–9B models score 0.02–0.36 on dialect steering (ALLaM-7B best at 0.36; Fanar-1-9B 0.038;
  SILMA 0.078), vs ~0.5 for frontier models. No open model larger than 9B had ever been
  tested — a gap this project's bake-off filled.
- **Dialect-LM recipes** (Atlas-Chat for Moroccan, NileChat for Egyptian): the converged
  approach is consolidate native resources → seeded synthesis → aggressive filtering.
  NileChat's negative finding — naive MSA→dialect translation produces culturally
  misaligned text — anticipated this project's own result (transform-style generation was
  the only kind that failed QC).

### 1.3 What is known about synthetic-data seeding and diversity

A second survey covered the seeding literature; its load-bearing findings:

- **Persona Hub** (2406.20094): persona conditioning scales diversity, but a 2025 follow-up
  (2505.17390) showed fine-grained personas add little *lexical* diversity and long prompts
  suppress persona effects. Personas earn their place as *voice grounding*, not as a
  diversity engine.
- **Cosmopedia v1→v2** (HuggingFace): the field moved from ~145 flat topics to a 34,000-node
  curated taxonomy *plus retrieved real seed text* — never away from real seeds. A flat
  ~200-topic list is an anti-pattern.
- **SODA** (2212.10465): ground dialogues in a *two-sentence situation with a causal hook*,
  not a topic noun. "Football" underdetermines a conversation; "his team lost and his
  brother won't stop texting stickers" determines one.
- **TinyStories** (2305.07759): random cross-products of orthogonal attributes are what
  measurably fight repetition — not longer topic lists.
- **Verbalized Sampling** (2510.01171): letting a model free-choose topics is the canonical
  mode-collapse regime (direct prompting retains ~24% of base diversity, traced to
  typicality bias in preference training). Asking for *k* candidates *with probabilities*
  in one call restores ~67% and improved downstream training in their experiments. Hence:
  the model proposes topics **offline** into a bank; generation is always conditioned.
- **Multi-party dialogue** (AAAI-26, 2502.13592): synthetic group chats come out
  structurally too polite; participation imbalance must be imposed explicitly. Turn-by-turn
  generation beats one-pass on repetition, at 2–4× the call cost (noted, not adopted).
- **Seed-anchor risk** (Canary's Echo, ICML 2025): in-context real excerpts create
  membership-inference exposure on the synthetic output — a privacy consideration, not a
  quality one. For this strictly-personal project the owner explicitly opted for real
  names and details throughout; the residual mitigation kept is rotating excerpts and an
  n-gram regurgitation filter.
- **Diversity measurement** (2403.00553): the non-redundant metric set is compression
  ratio, long-n-gram self-repetition/Self-BLEU, and embedding distance. Multi-generator
  sourcing (2511.01490) measurably improves how well models trained on synthetic data
  model *human* text — a direct argument for the multi-model design.

---

## 2. Data

### 2.1 The persona asset (unique to this project)

The owner's WhatsApp export (1.0M messages, 587 chats, 1,169 senders, ~10.5M tokens,
2016–2026) had previously been distilled into a personal knowledge base: **42 usable
persona profiles** (~1.35M words) with a fixed schema including a per-relationship "How we
talk" section — effectively a hand-written style guide per person — plus a 78K-word
self-model of the owner's own registers (script-mixing rules, vowel-lengthening signature,
emoji semantics, per-interlocutor mirroring). Machine-readable scaffolding accompanies it:
per-chat vocative frequencies, mention graphs, topic-count distributions, and full
speaker-tagged transcripts. One person is excluded from every artifact by a standing
decision, enforced by hard asserts in code.

Each profile was distilled (one Claude call per person) into a ~600–900-token **persona
card**: relationship, register, script mix, vocatives, topics, running jokes, verbatim
voice samples. An owner card sits in every chat request.

### 2.2 Real corpora used as anchors and seeds

| corpus | size | role in synthesis |
|---|---|---|
| WhatsApp conversations | ~10.5M tokens | chat excerpts (style anchors); empirical party-count distribution; participation-imbalance statistics |
| PKB corpora + group segments | 43 people, 15,306 eligible 1:1 segments, 4,554 group segments | excerpt pool; group rosters |
| Podcast transcripts (oddadmix) | ~2.5M tokens, 255 episodes | monologue anchors (spoken register) |
| sudaneseonline.com forum | ~243M tokens crawled; ~5.8M at dialect ≥0.8 | monologue anchors (written discourse register) |
| Sudanese_Flores DEV | 1,012 parallel pairs | few-shot exemplars (transform kind, later dropped); in-loop eval. DEVTEST never touched |
| Lisan-Sudanese | 47K tokens | training + a 281-sentence holdout eval independent of both the owner's contacts and translationese |

The forum corpus deserves a note: 72,761 threads were crawled politely (single-threaded,
0.5s delay, ~13.7h, 2 failures) covering 99.4% of the site's unique threads; a per-line
repair reversed the site's legacy UTF-8-as-cp1256 double-encoding, and every thread was
scored by a Sudanese-vs-MSA classifier so that only the dialect-dense band feeds synthesis
and training.

### 2.3 The situation bank

2,005 two-sentence Sudanese situations with causal hooks, generated offline via Verbalized
Sampling (8 candidates with probabilities per call) over roots taken from the personas'
*measured* topic distributions crossed with an orthogonal attribute grid (time-of-day ×
emotional valence × narrative arc × media type), then 4-gram-Jaccard deduplicated. This is
the only topic source the generators ever see; free topic choice never happens at
generation time.

### 2.4 Privacy stance

The project is strictly personal; the model and data never leave the owner's machines
except as generation requests to Claude under his own subscription. An initial
pseudonymization layer (stable fake names, phone/email masking) was built and then, by
explicit owner decision, **retired in favor of real person details** — the model is meant
to know his actual people, and real sender-name speaker labels align synthetic chats with
the real corpus. Phones and emails remain masked in outbound material. The one off-record
person remains excluded everywhere.

---

## 3. Methodology

### 3.1 Generation architecture

One CLI drives everything: the caller names models and sample counts; an internal registry
maps each model to its task and backend (Claude models → chats via headless subscription
calls; local Ollama models → monologues; entries are one-line additions). All requested
models generate concurrently. Sampling is stateless — every invocation draws fresh entropy,
and seed configurations are rich enough that repeats are practically impossible, so no
done-tracking exists. Every output records its generator and complete seed configuration,
and that provenance flows into the final training files.

**Chat seeds.** Party count is drawn from the real corpus's empirical distribution (2–5).
Two-party chats split between owner+contact and contact-pairs from the mention graph
(weighted by how often each mentions the other); three-plus-party chats use real group
rosters and carry an explicit participation-imbalance instruction sampled from that group's
actual turn shares (countering the "too polite" synthetic-group failure mode). The topic is
70% the participants' measured topic distribution, 30% situation bank. Every request
carries a rotated real excerpt chosen *independently of the topic* — the excerpt supplies
register, the situation supplies content, and decoupling prevents style-anchor content
bleed. Length: 30–50 turns with explicit anti-closure texture rules (topic drift, dead air,
no wrap-up endings).

**Monologue seeds.** Half are voiced by a persona card ("writer X", including the owner);
all carry a real discourse anchor (podcast or high-dialect forum text, topic-decoupled), a
situation from the bank, and a genre × audience assignment (6 genres × 4 audiences),
targeting 700–1,200 words with anti-padding instructions.

**Prompt evolution.** v1: 15–25-turn chats, 300–600-word monologues, flat topics. v2
(owner-directed): doubled lengths, anti-closure texture, non-owner pair/group chats. v3:
the situation-grounded design above. Every output records its prompt version; a
model-specific strict-format variant exists for Jais-2-8B (see experiments).

### 3.2 The evaluation harness

Two independent layers.

**Data-level QC** (every generated document): cheapest-first filters with logged kill
counts — format validity (chats must parse into turns by declared speakers only);
degeneration (repeated-n-gram ratio, word-run caps, compression ratio, Arabic-share floor);
leakage (hard reject on 8-gram overlap with the held-out evaluation chats or Flores
DEVTEST); seed regurgitation (8-gram overlap with the request's own anchor); near-duplicate
removal across the kept pool (4-gram Jaccard). Survivors face two *deliberately separate*
dialect axes:

- a **char-n-gram classifier** (Sudanese-vs-MSA, 98.1% holdout accuracy, trained on the
  project's own corpora) — free and instant, used for corpus-scale ranking, but blind to
  the difference between Sudanese and other dialects;
- an **LLM judge** (Claude Haiku, 1–5 rubric where 3 explicitly denotes
  "colloquial-but-not-Sudanese") — the only automated check that catches Egyptianization,
  the single most likely failure mode. Keep threshold: ≥4.

The axes are never blended: a high classifier score with a low judge score reads precisely
as "dialectal but the wrong dialect," which is how two bake-off models were caught.

**Model-level evaluation** (does the data help?): retraining ablations on the 110M model,
scored on held-out chat perplexity, owner-reply perplexity, Flores-Sudanese (dialect signal
independent of the owner's contacts), the Lisan holdout (native dialect, no shared people),
ArabicMMLU as an MSA-forgetting guard, and per-source validation perplexities.

**Diversity instrumentation** (per generator, continuous): compression ratio,
self-similarity (pairwise 4-gram Jaccard), and embedding nearest-neighbour distance,
reported against reference rows computed on the real corpora — the mode-collapse alarm for
the large run.

### 3.3 Resources and throughput

- **Claude Max subscription (20×)** — all Claude generation and judging; no paid API. A
  usage-limit hit pauses all workers on a shared 15-minute deadline and resumes
  automatically. Measured API-equivalent value: a near-empty headless call ≈ $0.02 (the
  ~30K-token session overhead rides on cache reads); a chat generation ≈ $0.04–0.05
  (Sonnet-class); a judge call ≈ $0.01. The 1M-token pilot consumed roughly $200–300
  API-equivalent; judging ~1,000 documents ≈ $10–15.
- **Dual RTX 5090 (32 GB each)** — local generation and all training. Local single-stream
  throughput: gemma4:12b ~144 tok/s (~340 monologues/h), gemma3:27b ~77 tok/s (~250/h),
  jais2-8b ~155 tok/s, llama3.3-70B ~35 tok/s across both GPUs. Training and local
  generation cannot overlap (same VRAM).
- **Claude concurrency**: 8 parallel headless calls ≈ 870–900 short-chat requests/hour or
  ~415/h for v2-length outputs; a continuous worker pool (not lock-step batches) was
  necessary to keep slots full.

---

## 4. Experiments

### 4.1 Acquisition ablation (does real acquired data help?)

Adding the acquired corpora (podcast transcripts, Lisan, lyrics, organic samples — ~9M new
tokens) to the stage-C training mixture, holding the recipe fixed:

| metric | baseline | + acquisitions |
|---|---|---|
| chat-holdout ppl | 63.3 | **60.0** |
| Flores-Sudanese ppl | 255.9 | **218.7** |
| owner-reply ppl | 206.4 | **186.3** |
| Flores-MSA ppl | 75.6 | 74.8 (no regression) |
| ArabicMMLU | 0.310 | 0.318 |

Every Sudanese metric improved at zero MSA cost, before any synthesis — establishing that
the pipeline's evaluation harness responds to real gains.

### 4.2 The pilot (~1M kept tokens)

2,000→1,231 requests (resized mid-flight by owner decision from a 5M-token plan, for hand
inspection; and re-mixed to 50% chat / 40% monologue / 10% transform with v2 prompts).
Results across 1,188 raw outputs:

- **Structural QC pass: 85%.** Kill profile: 130 seed-regurgitation (dominantly transforms
  lazily copying their MSA source), 39 format, 1 true leakage (removed).
- **Judge: 97.5% of survivors ≥4** (868 fives, 111 fours).
- **By kind: chats 99%, monologues 100%, transforms 39%.** The transform kind (MSA→dialect
  rewriting) failed twice over — exactly as NileChat's negative result predicted — and was
  **dropped entirely**; the production mix became 50% chat / 50% monologue.
- **By model arm: Sonnet 98% ≈ Opus 97%** — statistically indistinguishable, so production
  chat generation is Sonnet-only (~5× cheaper per token than Opus).
- **v2 (longer) prompts scored above v1** (98% vs 95%): length did not cost quality.
- Kept: 551 chats + 416 monologues + 12 transforms ≈ **0.70M tokens**, rendered in two arms
  (owner turns labeled `ME` vs labeled with his name) for a later ablation on identity
  contamination.

### 4.3 The local-model bake-off (seven models, 198 identical prompts each)

Question: can any locally-runnable open model generate usable Sudanese, offsetting
subscription usage? Design: every model answered the *same* 198 seed-anchored prompts
(99 chat / 99 monologue), scored on the same two axes, with the Claude pilot as ceiling.
Candidates chosen to test distinct hypotheses: Gemma-3-27B (best general Arabic-per-
parameter), Fanar-2-27B (best Arabic CPT, built *on* Gemma-3), Jais-2 (largest
Arabic-native family), ALLaM-7B (only open model with a measured dialect-steering edge),
Llama-3.3-70B (dialect mirroring reputation), Qwen3.8 and Gemma-4-12B (owner's local
models).

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

Findings:

1. **The Gemma family writes Sudanese monologues at 96% usable — above the Claude
   reference itself** — at zero marginal cost. Gemma-4-12B matches Gemma-3-27B's quality at
   twice the speed.
2. **No open model can hold a multi-turn Sudanese conversation.** The best (gemma3, 41%)
   is half of Claude's rate; several are at or near zero.
3. **Arabic-specialist models were catastrophic** (Fanar-2 at 3% *underperforming its own
   Gemma-3 base at 69%*), consistent with the published dialect-steering numbers. The
   classifier-vs-judge gap exposed why: their outputs are "dialectal" but generic/Egyptian,
   not Sudanese.
4. **Jais-2-70B was untestable — and a follow-up investigation closed the question.**
   Its official GGUFs degenerate under Ollama-CUDA *and* stock llama.cpp-Vulkan, in both
   K-quant (Q4_K_M) and plain quant (Q4_0). The publisher's documented micro-batch
   workaround (`-b 8`) makes prompt processing coherent but generation still collapses into
   repetition — i.e. the weights are probably sound but llama.cpp's GPU support for the
   custom architecture (squared-ReLU, μP) is numerically broken. Parked until the publisher
   or llama.cpp fixes it; not a model verdict. (Two other packaging defects were found and fixed during the bake-off: a
   community GGUF with an unparseable chat template, and a wrong auto-derived template;
   both are why the report distinguishes *model* failures from *packaging* failures.)
5. **A model-tuned strict-format prompt matters**: Jais-2-8B's chats went from 0% → 29%
   end-to-end (structural 3% → 37%, survivor judge 78%) with a hard output contract and
   shorter chats — real, but still third place.

Preamble-stripping was applied identically to all models at scoring time (measured effect:
only Jais benefited — an acknowledgment-preamble habit specific to it).

### 4.4 Gemma-only training ablation (does local synthetic data help the model?)

Before scaling local generation, an owner-requested ablation isolated the effect of the
Gemma-generated data alone. Control: the acquisitions-only stage-C model (§4.1's winner).
Arm: identical mixture, recipe, and ~3-epoch schedule, plus exactly one new ingredient —
the 813 QC-passed Gemma monologues then in the pool (1.12M tokens, 4.6% of the 25.4M-token
mixture; 528 from gemma4, 285 from gemma3 — a ~71% end-to-end QC pass rate in production
conditions, consistent with the bake-off's 96%-of-generated prediction after the judge's
share is applied to a larger, unsupervised batch).

| metric | control | + Gemma monologues |
|---|---|---|
| Flores-Sudanese ppl | 218.7 | **216.5** |
| chat-holdout ppl | 60.0 | 60.4 |
| owner-reply ppl | 186.3 | 186.0 |
| Lisan holdout ppl | 407.3 | 407.3 |
| Flores-MSA ppl (guard) | 74.8 | 75.3 |
| ArabicMMLU (guard) | 0.318 | **0.322** |
| per-source val (whatsapp / podcast / lyrics) | 47.8 / 115.7 / 158.9 | 47.6 / 115.3 / 158.7 |

Verdict: **safe, mild positive — gate passed.** A ~1% Flores-Sudanese improvement and
marginal gains on every per-source validation, with both forgetting guards intact and no
metric meaningfully regressed. The signal profile matches the register: monologue data moves
the discourse-sensitive metrics, not chat-holdout. At a 4.6% dose a small effect is the
expected shape of a genuine one; the result green-lights scaling local Gemma generation,
with dose-response left to the full-mixture experiment.

### 4.5 Codex-model audition (OpenAI free tier, chats)

The owner's ChatGPT account was added as a third backend (OpenAI Codex CLI, headless,
read-only sandbox). Probing showed a ChatGPT account permits exactly two models:
gpt-5.6-terra and gpt-5.5. The full free tier was drained generating chats on identical v3
prompts alongside a fresh 60-chat Sonnet reference:

| model | raw | end-to-end usable | survivor judge ≥4 | dialect prob |
|---|---|---|---|---|
| Claude Sonnet (reference) | 60 | **82%** | 49/50 | 0.962 |
| gpt-5.6-terra | 17 | **82%** | 14/15 | 0.951 |
| gpt-5.5 | 64 | **81%** | 52/52 | 0.940 |

The three are statistically indistinguishable at these sample sizes — Codex models generate
Sudanese chat at Sonnet's level on this harness. Two operational notes: gpt-5.6-terra
consumed the free-tier quota ~5× faster per output (reasoning-heavy default), and the
audition exposed two QC-parser artifacts that had been deflating *all* chat numbers
(case/shortening-strict speaker matching, and a 70-line ceiling that bursty 30–50-turn
conversations legitimately exceed). Fixing both and re-evaluating the cached kills recovered
~60 good conversations corpus-wide and raised every generator's measured rate — the earlier
Sonnet "68%" on v3 prompts was entirely artifact.

Verdict: worth keeping as chat generators — **adopted (owner decision 2026-08-28): both
GPT models join the production chat fleet alongside Sonnet.** On the free tier their
contribution is a small diversity garnish (~70 usable chats per drain), with gpt-5.5 as the
cost-effective workhorse; on a paid ChatGPT tier they become a real second chat engine at
Sonnet-equal quality.

### 4.6 Resulting production design

- **Chats → Claude Sonnet + gpt-5.6-terra + gpt-5.5** (all three measured at ~81–82%
  usable on identical prompts and gates; Sonnet carries the volume, the GPT models add
  generator diversity within their subscription quota — gpt-5.5 preferred of the two for
  quota efficiency).
- **Monologues → gemma3 + gemma4 locally** (96% usable, free) — roughly half of all
  generation moved off the subscription at measured-equal quality.
- **jais2-8b** remains an optional registry entry for a small chat share (generator
  diversity at zero cost) pending owner decision.
- Per-generator labels persist through to training files so a later ablation can detect
  generator-specific artifacts; per-generator diversity metrics run continuously.

---

## 5. Conclusion

Three claims are established by measurement rather than argument. First, **seed-anchored
synthesis works for Sudanese**: with real excerpts, persona cards, and situation grounding,
a frontier model produces conversations that pass an adversarial two-layer harness at rates
comparable to its performance on far better-resourced registers — and the resulting data
improved every dialect metric of a from-scratch model without MSA cost. Second, **the open-
model landscape is sharply bimodal for this task**: nothing local can sustain multi-turn
Sudanese chat, yet two Gemma models exceed the frontier reference on single-voice Sudanese
discourse, which makes a hybrid pipeline strictly better than either extreme — about half
the corpus generates free on two consumer GPUs. Third, **design choices that the synthetic-
data literature flags actually bind in practice**: MSA→dialect transformation failed
exactly as predicted and was dropped; free topic choice was never allowed (situations are
proposed offline under Verbalized Sampling); group chats need imposed participation
imbalance; and diversity is monitored per generator rather than assumed.

The pipeline's distinguishing asset is not any single technique but the grounding chain:
real corpus → measured distributions (party counts, topics, turn shares, vocatives) →
persona cards with verbatim voice samples → situation bank rooted in measured topics →
generation that is conditioned at every step and audited at two independent levels. All of
it runs on one subscription and two GPUs, with the expensive decisions (what to scale, on
which model, at what mix) each settled by a small measured experiment before any large
spend.

Open items at time of writing: the Sonnet-chat training ablation (the Gemma-monologue arm
passed, §4.4), the owner's blind calibration of the LLM judge, and the scale-up itself.
