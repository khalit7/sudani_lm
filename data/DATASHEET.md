# Datasheet — sudani_lm

## Overview

This file is the single reference for every dataset in this repository: what it is, where it came
from, where it lives on disk, how big it is, and which training stage consumes it.

The project trains a Sudanese-Arabic chat model from scratch. **The pipeline is four stages, and
every model and config off that path has been deleted:**

| stage | config | data | what it does |
|---|---|---|---|
| **A** smoke test | `configs/stage_a.yaml` | `data/packed/pretrain/` | 42M params, 1.5B tokens — validates the pipeline before the long run |
| **B** pretraining | `configs/stage_b.yaml` | `data/packed/pretrain/` | 110M params, 5B tokens |
| **C** continued pretraining | `configs/stage_c.yaml` | `data/packed/stage_c/` | **all** chat + Sudanese ×3 + 40% Arabic replay, loss on every token |
| **D** style SFT | `configs/stage_d.yaml` | `data/packed/stage_d/` | owner-heavy chats, **loss on the owner's replies only** → **final model** |

How C and D divide the chat corpus was settled by ablation (`plan.md`, *Ablation: how should C and
D divide the chat corpus?*): Stage C must see the **whole** chat corpus — every variant that
reserved chat for D was worse at both ends — and Stage D earns its place only because its loss is
**masked**, which makes it a different gradient rather than more epochs.

The `Role` column below records what each dataset is *actually* used for — which, for SmolKalam and
InstAr-500k, is nothing. Both are kept on disk (4.4 GB combined) rather than deleted; the reason
each went unused is in its own section, and both would matter again if the project acquires new
supervision.

Every figure here was measured on the local copy, not copied from a dataset card. Row counts are
exact. **Token counts were measured with the 8k tokenizer in `tokenizers/init_tokenizer/`, which
has since been replaced by `tokenizers/v2_32k/`** — the v2 tokenizer compresses ~15–25% better, so
treat these as relative. Measured against v2 the ArabicWeb24 half is 13.49B rather than the 17.0B
implied here. ArabicWeb24's token
count is derived from an exact sum of the corpus's own `metadata.token_count` scaled by a measured
1.190 calibration factor to our tokenizer; all others are exact counts over the full text.

`data/` holds only data — all code lives in `src/` (`src/preprocessing/` for offline preparation,
`src/tokenizer/` for tokenizer building, `src/dataset/` for the runtime loaders). Sources are
immutable and live under `data/raw/<dataset>/`, one folder per dataset. Derived artifacts —
`data/interim/` for intermediate splits and `data/packed/<stage>/` for training-ready token streams —
are regenerable and not documented here. Everything under `data/` except the sheets (`DATASHEET.md`, `SYNTHSHEET.md`,
`SCRAPESHEET.md`) is gitignored, so no raw sample of any dataset is kept in the repository; the per-dataset examples below are the
canonical record of what each one looks like.

Keep this file current when a dataset is added, removed, or re-scoped.

---

## Summary — the single source of truth for ALL data

One row per dataset, every dataset the project holds. **Added**: w1 = original stack
(2026-08-16), w2 = data-expansion v2 (2026-08-22), w3 = acquisition wave 3 + final sweep
(2026-08-28 → 08-31). Tokens are exact (project tokenizer, v2_32k); gold/seasoned/silver =
dialect-score bands ≥0.8 / 0.5–0.8 / <0.5; **TBC** cells are filled when their count or
crawl finishes. Dialect quartiles are per-doc [25% / median / 75%]; "—" = not scored
(MSA corpora, evals, non-text). Scraped rows are authors' copyright, **private use only,
never redistributed**; details per dataset in the sections below.

| Dataset | Added | Status | Role | Local path | Raw size (MB) | Tokens | Dialect (quartiles) | License |
|---|---|---|---|---|---|---|---|---|
| [ArabicWeb24](https://huggingface.co/datasets/lightonai/ArabicWeb24) | w1 | ready | **Stages A & B** pretraining (5B of 13.49B) **+ 40% replay in Stage C** | `data/raw/arabicweb24/` | 189,378 | **33.9B** | — | ODC-BY |
| [SmolKalam](https://huggingface.co/datasets/AdaMLLab/SmolKalam-Arabic-Conversational-SFT) | w1 | ⚠️ not used | only fed the SFT stage the ablation dropped | `data/raw/smolkalam/` | 3,366 | **1.62B** | — | Apache-2.0 |
| [InstAr-500k](https://huggingface.co/datasets/ClusterlabAi/InstAr-500k) | w1 | ⚠️ not used | never entered any packed stage | `data/raw/instar500k/` | 1,043 | **147.2M** | — | Apache-2.0 |
| WhatsApp export (personal) | w1 | ready | **Stage C** (all of it) **and Stage D** (owner-heavy chats, masked); held-out chats are the main metric | `data/raw/whatsapp/` | 138 | **10.47M** | — (all-dialect by construction) | private, not redistributable |
| [ArabicMMLU](https://huggingface.co/datasets/MBZUAI/ArabicMMLU) | w1 | eval only | MSA forgetting guard in Stages C/D | `data/raw/arabicmmlu/` | 8 | **556.4K** | — | CC-BY-NC-4.0 |
| [SudSenti](https://github.com/mustafa20999/Sudanese-Arabic-Sentiment-Datasets) | w1 | ready | **Stage C**, upsampled ×3 | `data/raw/sudsenti/` | 2 | **381.6K** | — | unstated (academic release) |
| [Sudanese_Dialect_Tweet_Tele](https://huggingface.co/datasets/arbml/Sudanese_Dialect_Tweet_Tele) | w1 | ready | **Stage C**, upsampled ×3 | `data/raw/sudanese_tweets_tele/` | 1 | **204.0K** | — | unstated |
| [Sudanese_Flores](https://huggingface.co/datasets/McGill-NLP/Sudanese_Flores) | w1 | eval only | independent Sudanese signal (DEV tracked; **DEVTEST held out of everything**, enforced by the 8-gram screen) | `data/raw/sudanese_flores/` | 1 | **140.9K** | — | unstated (FLORES derivative) |
| [Sudanese_Dialect_Tweet](https://huggingface.co/datasets/arbml/Sudanese_Dialect_Tweet) | w1 | ready | **Stage C**, upsampled ×3 | `data/raw/sudanese_tweets/` | 1 | **50.0K** | — | unstated |
| [Tarab — Sudanese slice](https://huggingface.co/datasets/drelhaj/Tarab) | w2 | ready | **Stage C v2**, capped ~2% of mixture | `data/raw/tarab/` | 16 | **428.8K** (post-dedup) | — | CC-BY-4.0 |
| [Lisan-Sudanese](https://huggingface.co/datasets/AymanMansour/Lisan-Sudanese-TTS-Dataset) (via TTS mirrors) | w2 | ready | **Stage C v2** + **Lisan holdout eval** (15% held out) | `data/raw/lisan/` | 1 | **47.0K** | — | CC-BY-4.0 |
| [Organic Sudanese sample](https://huggingface.co/datasets/ebubekr53/organic-sudanese-arabic-dialect-dataset) | w2 | ready | **Stage C v2** (via `sudani.py`) | `data/raw/organic_sudanese/` | 1 | **3.6K** | — | CC-BY-4.0 |
| [Alexandria — SD subset](https://huggingface.co/datasets/UBC-NLP/alexandria) | w2 | ⚠️ reserved | eval/seed material only, not training | `data/raw/alexandria_sd/` | 1 | 345 conversations | — | CC-BY-NC-4.0 |
| oddadmix Sudanese transcripts ([podcast](https://huggingface.co/datasets/oddadmix/arabic-audio-collection-sudanese-sudan-podcast), nuuar, gobara) | w2 | ready | **Stage C v2** | `data/raw/oddadmix/` | 26 | **2.49M** (255 episodes, 6,545 docs) | — | unstated (scraped YouTube) — private use only |
| [sudaneseonline.com forum crawl](https://sudaneseonline.com/board/index.htm) (69,633 threads, 99.4% of site) | w2 | ready | **Stage C v2**, dialect-ranked: mixture takes ≥0.5 ≈ **19.6M** | `data/raw/sudaneseonline/` → `data/interim/sudaneseonline/` | 1,547 | **~243M** (gold ≈5.8M, ≥0.5 ≈19.6M) | banded 2026-08-23 | private use only |
| Telegram channels ×16 (nasra8ya, rwayatSudan, novelsforus2, …) | w3 | ready | **Stage C v2** — largest gold source in the project | `data/raw/telegram/` → `data/interim/telegram/` | 632 | **58.1M** (gold **47.1M**, seasoned 4.6M, silver 6.4M) | [0.88 / 0.973 / 0.991] | private use only |
| anasudani.net forum (phpBB, 47,827 topic docs) | w3 | ready | **Stage C v2**, dialect-ranked | `data/raw/anasudani/` → `data/interim/anasudani/` | 772 | **64.6M** (gold 9.7M, seasoned 7.2M, silver 47.7M) | [0.051 / 0.241 / 0.689] | private use only |
| alnilin.com reader comments (158,896 article-grouped docs) | w3 | ready | **Stage C v2** — comment gold | `data/raw/alnilin/` → `data/interim/alnilin_comments/` | 2,202 (shared with posts) | **42.6M** (gold 9.0M, seasoned 11.5M, silver 22.1M) | [0.381 / 0.702 / 0.912] | private use only |
| alnilin.com articles (536,340 docs) | w3 | ready | silver MSA reservoir, bottom-ranked | `data/raw/alnilin/` → `data/interim/alnilin_posts/` | ″ | **251.2M** (gold 3.4M, seasoned 4.8M, silver 243.0M) | [0.01 / 0.029 / 0.095] | private use only |
| sudanile.com articles (141,508 docs) | w3 | ready | silver MSA reservoir | `data/raw/sudanile/` → `data/interim/sudanile_posts/` | 1,368 | **188.8M** (gold 1.6M, seasoned 4.0M, silver 183.2M) | [0.019 / 0.048 / 0.132] | private use only |
| koorasudan.net comments (16,084 docs) | w3 | ready | **Stage C v2** — football-fan dialect | `data/raw/koorasudan/` → `data/interim/koorasudan_comments/` | 518 (shared with posts) | **3.2M** (gold 0.5M, seasoned 0.8M, silver 2.0M) | [0.337 / 0.624 / 0.863] | private use only |
| koorasudan.net articles (127,832 docs) | w3 | ready | silver sports MSA | `data/raw/koorasudan/` → `data/interim/koorasudan_posts/` | ″ | **49.8M** (gold 1.2M, seasoned 1.8M, silver 46.8M) | [0.024 / 0.068 / 0.196] | private use only |
| cover-sd.com articles (1,653 docs) | w3 | ready | silver | `data/raw/cover_sd/` → `data/interim/cover_sd_posts/` | 4 | **0.4M** (silver 0.4M) | [0.028 / 0.13 / 0.357] | private use only |
| Blogger cluster ×12 (1,419 posts) | w3 | ready | **Stage C v2** — dialect lives in sudanese_novels/katabsudsnese/hageebatalfun; delta blogs are MSA | `data/raw/blogger/` → `data/interim/blogger/` | 30 | **1.3M** (gold 0.2M, seasoned 0.2M, silver 1.0M) | [0.016 / 0.063 / 0.227] | private use only |
| aghaniwamthal.com (564 docs) | w3 | ready | **Stage C v2** — proverbs + lyrics | `data/raw/small_sites/` → `data/interim/small_sites/aghaniwamthal/` | 54 | **0.24M** (gold 0.03M, seasoned 0.03M, silver 0.18M) | [0.076 / 0.276 / 0.669] | private use only |
| Dead-forum archives — Wayback ×7 domains + Common Crawl ×4 (cross-source deduped) | w3 | **crawling** (~3–4 days) | **Stage C v2** once miners finish | `data/raw/wayback/`, `data/raw/commoncrawl/` → `data/interim/vbarchive/<domain>/` | 913+ (growing) | **TBC** (sudanesesongs partial: 2,081 docs / 4.9M chars) | TBC | private use only |
| Synthetic pilot pool (QC-passed: 2,119 chats + 2,042 monologues + 12 transforms; see SYNTHSHEET) | w3 | ready (pilot); production run pending | **Stage C v2** synthetic side, ≤50% cap | `data/interim/synthetic/` | 254 | **≈5.1M** kept | judge-gated ≥4/5 | self-generated, private |
| [muzammilsoft/Sudanese_dialect_dataset](https://huggingface.co/datasets/muzammilsoft/Sudanese_dialect_dataset) (LLM-generated) | w3 | **pending QC** | candidate for the synthetic pool only; never counts as organic | `data/raw/hf_muzammilsoft/` | 13 | **TBC — unevaluated** | TBC | CC-BY-4.0 |

**Total on disk: ≈197 GB** (`data/raw/`, measured 2026-08-31). ArabicWeb24 is ~96% of that
footprint. The Sudanese-register side has grown from **11.1M tokens** (w1, 0.03% of the total)
to **≈73M gold + ≈35M seasoned** across the wave-3 corpora alone (all counts now exact except
the still-crawling forum archives), plus sudaneseonline's ≥0.5 band (19.6M) and the w2
acquisitions — **≈130M dialect-bearing tokens in total** — with silver MSA news reservoirs of
**≈476M** behind them.

Pipeline rule (applies to every row): each source lands in `data/raw/<name>/`, gets a
preprocessing module producing `data/interim/<name>/{train,val}.jsonl` split by container
(episode / thread / channel-section / article — never by post), is dialect-scored per doc,
passes a spot audit and the Flores-DEVTEST 8-gram leakage screen
(`src/preprocessing/leakage_screen.py`, zero hits across all files 2026-08-31), and enters
training only through a mixture manifest (`configs/mixtures/*.yaml`) that records per-source
token counts in the pack's `meta.json`.

---

## ArabicWeb24

- **Role:** **Stages A and B** pretraining — 5B tokens drawn from the 13.49B kept half — **plus
  the 40% Arabic replay in Stage C**, sampled directly from `data/packed/pretrain/train.bin` so replay
  comes from exactly the distribution the base model saw · **Path:** `data/raw/arabicweb24/` · **Tokens:** 33.9B · **Size:** 189,378 MB
- **License:** ODC-BY · **Source:** https://huggingface.co/datasets/lightonai/ArabicWeb24

Arabic-only web crawl, filtered and deduplicated by LightOn, released as 38,159,291 documents across
396 Arrow shards. Crawled in a single window, **2024-01-25 to 2024-02-21**, and heavily weighted
toward news and media — sampled domains include `baladi-news.com`, `filgoal.com` (sports),
`aljornal.com`, `amad.ps`, `rosaelyoussef.com`. Almost entirely Modern Standard Arabic; it contains
no meaningful Sudanese dialect. Documents carry `metadata.token_count` and
`metadata.labels.language_score` (99.9% labelled `ar`, mean score 0.994), both usable as quality
filters. Mean length is 748 tokens by the corpus's own count, with a long tail — 23.8% of documents
exceed 1024 tokens and the maximum is 20,505.

**Examples** (truncated to 300 characters; each is one document's `text`):

1. `بعث الملك محمد السادس برقية تهنئة إلى أعضاء نادي الرجاء الرياضي لكرة القدم، وذلك بمناسبة تتويج النادي بكأس الكونفدرالية الإفريقية 2018. ومما جاء في برقية الملك «فبمناسبة فوز نادي الرجاء الرياضي لكرة القدم بكأس الكونفدرالية الإفريقية 2018، يطيب لنا أن نتوجه إليكم، ومن خلالكم لكافة مكونات الفريق وجم`
2. `مباشر: ذكرت شبكة "سي إن بي سي" الأمريكية أن الحظر الأمريكي على استيراد بعض نماذج ساعات "آبل" الذكية دخل حيز التنفيذ اليوم الثلاثاء، بعدما عمدت إدارة الرئيس جو بايدن، إلى عدم استخدام حق النقض على حكم بشأن انتهاكات براءات الاختراع.`
3. `التقديم في وظائف هيئة الأمر بالمعروف والنهي عن المنكر ” رابط التقدم لشغل وظائف هيئة الأمر بالمعروف والنهي عن المنكر “هيئة الأمر بالمعروف والنهي عن المنكر هي إحدى الجهات التابعة مباشرة لرئاسة مجلس الوزراء، وقد صدر مرسوم إنشائها في عام 1437/4/4 هـ بموجب المرسوم رقم 289`
4. `قيادة الجيش الألماني في الغرب (Oberbefehlshaber West) (بالألمانية: بالاحرف الأولى من OB West) هي القيادة العامة للجبهة الغربية، والقوات المسلحة الألمانية على الجبهة الغربية خلال الحرب العالمية الثانية. كانت تابعة مباشرة للقيادة العليا للقوات المسلحة الألمانية.`

---

## SmolKalam

- **Role:** ⚠️ **not used in the final pipeline** · **Path:** `data/raw/smolkalam/` · **Tokens:** 1.62B · **Size:** 3,366 MB
- **License:** Apache-2.0 · **Source:** https://huggingface.co/datasets/AdaMLLab/SmolKalam-Arabic-Conversational-SFT

Multi-turn Arabic conversations produced by ensemble machine translation of
`HuggingFaceTB/smoltalk2` (Seed-X 7B and Gemma 3 27B, with per-sample selection by an Arabic reward
model). One row per conversation, format `messages: [{role, content}, …]`, plus `source`,
`rank_score`, and `SCR`/`LR` script-purity and language-ratio scores usable as quality filters.

The upstream release has 24 configs and 1,790,478 rows. **Only the 8 conversational configs are kept
here — 1,117,451 rows:** `smol_magpie_ultra` (406,843), `OpenHermes_2.5` (384,900),
`multilingual8` (244,736), `systemchats_30k` (33,997), `systemchats_Qwen3` (27,436), `aya_dataset`
(15,222), `everyday_conversations` (2,260), `everyday_convs_reasoning` (2,057). The reasoning,
tool-calling, table and long-context configs were dropped: they teach capabilities a 110M model will
not have, and the `LongAlign_*` sets carry 64k–131k-token contexts against a 1024-token model.

It is machine-translated MSA, so the register is stiff and translationese. It was intended to supply
turn-taking *structure*, not style.

**Why it ended up unused.** It was accepted and trimmed specifically to feed a chat-SFT stage. The
ablation in `plan.md` then found that stage makes the model worse — and, counter-intuitively, that
*removing* SmolKalam made it worse still (Flores Sudanese 338 vs 311), because the off-distribution
replay was regularising against chat overfitting. So it was helping, inside a stage that should not
exist. Kept on disk: it is the obvious replay source if a future stage earns its place. Note that some configs
retain `<think>` reasoning traces in the assistant turn, which must be stripped before use.

**Examples** (one turn pair per config):

1. `everyday_conversations` — `[user] مرحباً` → `[assistant] مرحباً! كيف يمكنني مساعدتك اليوم؟`
2. `everyday_conversations` — `[user] أبحث عن طرق لتحسين مهاراتي في التعلم. ما هي خرائط العقل؟` → `[assistant] تعتبر رسم الخرائط الذهنية تقنية بصرية لتنظيم وربط الأفكار، باستخدام مخططات تتضمن كلماتًا وصورًا وألوانًا. وهي تساعد في هيكلة المعلومات...`
3. `aya_dataset` — `[user] أجب عن السؤال: "ما هو طول نهر الغانج؟"` → `[assistant] <think> حسنًا، يسأل المستخدم عن طول نهر الغانج. دعني أبدأ باسترجاع ما أعرفه...`
4. `OpenHermes_2.5` — `[assistant] كشف النص أن شركة BMW، وهي شركة تصنيع السيارات التي تتخذ من ميونيخ مقراً لها، أعلنت عن زيادة كبيرة تتجاوز ٢٠٪ في الأرباح الأساسية لتصل إلى ٢٫٥٢ مليار يورو`

---

## InstAr-500k

- **Role:** ⚠️ **not used in the final pipeline** — never entered any packed stage. Loaders
  survive at `src/dataset/arabic_ift.py` · **Path:** `data/raw/instar500k/` · **Tokens:** 147.2M · **Size:** 1,043 MB
- **License:** Apache-2.0 · **Source:** https://huggingface.co/datasets/ClusterlabAi/InstAr-500k

Natively-Arabic instruction-tuning set, 481,281 rows split 433,152 train / 48,129 test. Fields:
`uuid, source, task, type, topic, system, instruction, output`. 61% generated, 39% human-crafted.

Its value relative to SmolKalam is that it is *natively Arabic* rather than translated, so the
phrasing is idiomatic — but it is **single-turn only**, and heavily concentrated: 42% of rows come
from one source (`arabic_categorization_sanad`, news categorisation) and 45% of rows are
Classification. The remaining sources are `aya_collection` (15%), `arabica_qa` (13%),
`classical_arabic_poetry` (9%), `cidar`, `aqad`, `101_billion_arabic_dataset`, `abu_el_khair`.
Every row has a populated `system` field.

**Why it ended up unused.** Planned as the instruction-tuning corpus, then superseded by SmolKalam
for multi-turn structure, then made moot entirely when the ablation dropped the SFT stage. It fed
one pre-rebuild run (8k tokenizer, old trainer) and never the current pipeline. Kept on disk: being
*natively* Arabic rather than translated, it is the better source if a future stage needs idiomatic
instruction data.

**Examples** (instruction → output, truncated):

1. *(Mixed / Economy&Finance / cidar)* `أجب على السؤال التالي. يجب أن يكون الجواب باللغة العربية ولا تخرج عن سياق الموضوع. أنشئ مثالًا لمراجعة جيدة من العملاء.` → `اشتريت مؤخرًا منتجًا من هذه الشركة ولم أستطع أن أكون أكثر رضاً. كانت جودة المنتج ممتازة ووصل في حالة مثالية.`
2. *(Rewrite / Sports)* `إعادة صياغة النص التالي: مباراة الأزمة بيرلو يعلن تشكيل يوفنتوس ونابولي لم يحضر...` → `أعلن بيرلو تشكيلة يوفنتوس لمواجهة نابولي التي قد لا تقام بسبب كورونا، حيث لم يسافر نابولي إلى تورينو`
3. *(Mixed / Entertainment)* `ما هي الفئة التي ينتمي إليها هذا السؤال؟ ما هو آخر فيلم لمارفل مع ثانوس؟` → `الفئة التي ينتمي إليها السؤال هي أفلام`
4. *(Classification / Economy&Finance)* `تم أخذ هذا الخبر من صحيفة إلكترونية عربية... ما هي الفئة التي ينتمي إليها هذا الخبر؟ قالت شركة أرامكو السعودية إنها أوقفت التعامل...` → `الفئة التي ينتمي إليها هذا الخبر هي اقتصاد`

---

## WhatsApp export (personal)

- **Role:** **Stage C** — all 32,286 conversations, loss on every token — **and Stage D**, the
  24,427 conversations from the 339 chats where the owner's token share is ≥20%, with loss masked
  to the owner's own replies (1.88M of 5.88M tokens scored). Held-out chats are the headline
  metric · **Path:** `data/raw/whatsapp/`
- **Tokens:** 10.47M · **Size:** 138 MB · **License:** private, not redistributable
- **Source:** personal iPhone export via iMazing (see `notes.md`, STEP1)

The target-domain corpus and the reason this project exists. 1,014,518 rows — 588,870 `Incoming`,
412,830 `Outgoing`, 12,818 `Notification` — of which **942,245 carry text**, spanning **587 chat
sessions** and **1,169 senders**. Fields include `Chat Session, Message Date, Sent Date, Type,
Sender ID, Sender Name, Text, Replying to, Attachment`; `Replying to` gives explicit threading and
the timestamps allow session segmentation by idle gap.

At 10.47M tokens it is **16.5× larger than all three public Sudanese corpora combined** (0.64M), so
it is treated as the Sudanese corpus rather than as a fine-tuning extra. **15.6% of its characters
are Latin** — Arabizi and English code-switching — which the Arabic-only 8k tokenizer fragments
badly and which motivates the 32k rebuild.

This contains private messages from 1,169 people who did not consent to being modelled, and a model
trained on it will memorise names and numbers. Keep the model local.

**Examples.** This datasheet is published to a public repository, so unlike every other dataset here
the examples are restricted to the two messages that were **already public** in the sample file
committed at `5f0abe8` (that file has since been removed from the working tree):

1. `[Incoming] Mukh: شباب…دايرين نخش ال ballot`
2. `[Incoming] Mukh: رسلو ال Membership number و last name عشان اعمل ليكم add🙌`

Both happen to be incoming group-chat messages, and they illustrate the code-switching well
(`دايرين نخش ال ballot`) but not the corpus as a whole. Three registers are **not** represented above
and are described rather than quoted: outgoing turns (the 412,830 messages the model learns to
generate, which carry the target persona); one-to-one conversation, which is where the longest
threads live and reads far more informally than group chat; and fully Latin-script Arabizi such as
`kaif al7al ya zol`, which appears throughout and which the current tokenizer splits into 11 tokens.

---

## ArabicMMLU

- **Role:** Eval only — MSA forgetting guard during Stages C and D · **Path:** `data/raw/arabicmmlu/` · **Tokens:** 556.4K · **Size:** 8 MB
- **License:** CC-BY-NC-4.0 · **Source:** https://huggingface.co/datasets/MBZUAI/ArabicMMLU

Multiple-choice questions drawn from school and professional exams, 14,455 test rows plus a 120-row
`dev` few-shot pool. Used as an MSA-knowledge regression guard only — **it contains no Sudan
content** (Jordan 5,990, unlabelled 2,997, Egypt 2,487, Palestine 2,032, Morocco 314, Lebanon 239,
UAE 184, Kuwait 111, KSA 101), so it cannot measure progress toward this project's goal.

Two properties matter when writing the evaluator: **option counts vary** — 4 options for 10,120
questions, 3 for 2,121, 2 for 1,874, and 5 for only 340 — so code that assumes five is wrong for
97.6% of the set; and `Answer Key` skews toward A, so option order must be shuffled deterministically.

**Examples:**

1. *(Islamic Studies / Primary / Palestine, key=C)* `الكوثر هو` — options: `واد في جهنم` · `شجرة مباركة` · `نهر في الجنة`
2. *(Math / Primary / Jordan, key=A)* `جد ناتج 65 ÷ 523` — options: `8 والباقي 3` · `3 والباقي 3` · `8 والباقي 31` · `8 والباقي 8`
3. *(Natural Science / Primary / Jordan, key=D)* `أي الحيوانات التالية لاتشبه والديها في مرحلة أو أكثر من دورة حياتها` — options: `الماعز` · `الدجاجة` · `الحصان` · `الضفدع`

---

## SudSenti

- **Role:** **Stage C**, upsampled ×3 · **Path:** `data/raw/sudsenti/` · **Tokens:** 381.6K · **Size:** 2 MB
- **License:** unstated (academic release) · **Source:** https://github.com/mustafa20999/Sudanese-Arabic-Sentiment-Datasets

Two Sudanese sentiment corpora, SudSenti2 (2-class) and SudSenti3 (3-class), released alongside
[arXiv:2201.12664](https://arxiv.org/abs/2201.12664). Eight files: `SudSenti2-Tweets.txt` (4,652
lines) and `SudSenti3-Tweets.txt` (7,542 lines) carry the text, tab-separated as `text<TAB>label`
with labels `pos` / `neg` / `neural` *(sic — the source misspells "neutral")*; the remaining six are
small train/test/validation splits.

The largest of the three public Sudanese sets, but the least dialectal: a substantial share is
Sudan-*related* MSA news copy and even Quranic quotation rather than Sudanese dialect. It needs
filtering before it counts as dialect data.

**Examples** (verbatim, including the trailing label):

1. `السودان يعلن فشل مفاوضات سد النهضة وإعادة الملف إلى الاتحاد الأفريقي | المصري اليوم 	neg`
2. `عمليات المسح الحشري للأطوار المائية والطائرة لمواقع التوالد داخل وخارج المنازل بمحلية طوكر	pos`
3. `القاء بحث بعنوان نظم وخدمات المعلومات المتخصصه في قطاع المنتجات الزراعيه / مؤتمر الاتحاد العربي للمكتبات والمعلومات الثاني والعشرين (اعلم) جامعه الخرطوم / السودان / ٢٠١١	neural`
4. `سوق الناس بالخلاء يافريق الخلاء ، نهايتك اقتربت في الموعد القادم لجلسة كوشيب ٧.١٢.٢٠٢٠ المفاجئة داااااااوية ، دا مانبيل أديب 	neg`

---

## Sudanese_Dialect_Tweet_Tele

- **Role:** **Stage C**, upsampled ×3 · **Path:** `data/raw/sudanese_tweets_tele/` · **Tokens:** 204.0K · **Size:** 1 MB
- **License:** unstated · **Source:** https://huggingface.co/datasets/arbml/Sudanese_Dialect_Tweet_Tele

5,346 Telegram posts in Sudanese dialect with a 3-way sentiment `label` (0: 3,754, 1: 851, 2: 741).
Fields: `Tweet_ID, Tweet_Text, Date, label`. Being Telegram-sourced it sits closer to chat register
than the Twitter set, which makes it the more useful of the two for this project despite still being
a classification corpus. Text retains raw newlines and truncation markers from collection.

**Examples:**

1. `زىن لو شغل الاتصالات ده غالبكم اشتغلوا تجار بصل فى السوق المركزى ماممكن شبكة النت تقع اكتر من اربعة ساعات ولا تعب…`
2. `ام تى ان ىعنى م ممكن الزول مشترك ف الحزمة البلاتىنىة و ىرسل التوىتة كم مرة حتى تمش وفىدىوهات الوانس م ىقدر ىنزلها حرامىه`
3. `مقاطعة.زىن اضعف نت جربته`
4. `شركة.زىن زفتتت`

Note the systematic `ي` → `ى` substitution throughout — a collection or normalisation artefact, not
dialect spelling. It should be normalised before training or it will teach the model a spelling that
does not occur naturally.

---

## Sudanese_Flores

- **Role:** Eval only — the independent Sudanese signal. **DEV** is tracked during training;
  **DEVTEST is untouched**, reserved for a final number · **Path:** `data/raw/sudanese_flores/` · **Tokens:** 140.9K · **Size:** 1 MB
- **License:** unstated (FLORES derivative; upstream is CC-BY-SA) · **Source:** https://huggingface.co/datasets/McGill-NLP/Sudanese_Flores

FLORES sentences translated into Sudanese Arabic and paired with MSA: `DEV.jsonl` (1,012 rows) and
`DEVTEST.jsonl` (997), each row a single `translation{Arb, Sud}` field. No train split — it is a
benchmark. Track DEV during training and hold DEVTEST back for final numbers.

This is the only independent Sudanese signal available. Held-out WhatsApp perplexity is
self-referential — the same 1,169 people, topics and idiolect — so a model can score well on it
while having learned nothing generalisable; Flores is unrelated text and also supports an
MSA→Sudanese translation eval. The Sudanese side is genuinely dialectal rather than lightly-edited
MSA: note `هسة`, `دي`, `زي`, `خالص`, `لي` below.

**Examples** (MSA → Sudanese):

1. `يعتمد مطبخ مايوركا، مثل بقية المناطق المماثلة في البحر الأبيض المتوسط، على الخبز والخضروات واللحوم` → `بعتمد مطبخ مايوركا، زي باقي المناطق اللي زيها في البحر الابيض المتوسط، على العيش و الخضروات و اللحوم`
2. `حصل ديل بوترو على ميزة مبكرة في المجموعة الثانية، لكن هذا تطلب أيضاً كسر التعادل بعد الوصول إلى 6-6.` → `ديل بوترو حصل على ميزة مبكرة في المجموعة التانية، لكن دا اتطلب برضو كسر التعادل بعد الوصول لي 6-6.`
3. `يكتب الناس الآن رسائل على شاشات الكمبيوتر، دون الاحتياج إلى استخدام المباراة.` → `الناس هسة بتكتب الرسايل على شاشات الكمبيوتر، من غير حاجة لاستعمال المباراة`
4. `الذرات صغيرة للغاية حتى أن تريليونات منها يمكنها أن تشغل حيز النقطة الموجودة في نهاية هذه الجملة.` → `الذرات صغيرة خالص حتى انو تريلونات منها ممكن تملا مساحة النقطة الموجودة في نهاية الجملة دي.`

---

## Sudanese_Dialect_Tweet

- **Role:** **Stage C**, upsampled ×3 · **Path:** `data/raw/sudanese_tweets/` · **Tokens:** 50.0K · **Size:** 1 MB
- **License:** unstated · **Source:** https://huggingface.co/datasets/arbml/Sudanese_Dialect_Tweet

The smallest dataset here: 2,119 Sudanese tweets with three independent sentiment annotations plus a
majority `Mode` (846 negative, 764 positive, 509 neutral). Fields: `Tweet, Annotator 1..3, Mode,
Date`. Topically narrow — a large share concerns complaints about ride-hailing and telecom services —
so it contributes dialect vocabulary and register rather than breadth. Its main practical use is as
upsampled material when training the 32k tokenizer, where dialect merges matter more than volume.

**Examples:**

1. `ما شاء الله الخدمة متازة وتعامل الكباتن جدا راقي فقط مسألة تقيم الخدمة يمكن ان تكون عبر التطبيق وبعد المشوار مباشرة بدلاً عن الاتصال أحياناً يكون الاتصال مزعج` *(Mode=2)*
2. `وامدرمان طيب` *(Mode=1)*
3. `ترحال اسعارهم غالية و ما بقبلو كل المشاوير و ماف عربات باقي التطبيقات دايما ما عندهم عربات كفاية` *(Mode=0)*
4. `مقاطعة ترحال ادونا شروط التسجيل عندكم` *(Mode=0)*

---

## Tarab — Sudanese lyric slice (v2)

- **Role:** **Stage C v2**, capped ~2% of the mixture, `repeat: 1` · **Path:** `data/raw/tarab/tarab_Sudanese.csv` · **Tokens:** 428.8K post-dedup · **Size:** 16 MB raw
- **License:** CC-BY-4.0 · **Source:** https://huggingface.co/datasets/drelhaj/Tarab (ABJAD-NLP 2026; includes the older Habibi lyric corpus — `corpus_version` column)

89,226 verses across 2,663 songs; `src/preprocessing/lyrics.py` reassembles songs from
`art_id`/`verse_order`, and because repeated choruses ship as duplicate rows with the same
`verse_order`, sorting makes them adjacent and the consecutive-duplicate collapse reduces the
slice to near unique-verses-per-song (2.59M → 1.28M chars). That is intentional: sung repetition
is a training hazard for a model with a repetition failure mode. Split by song, 2% val.

**Example** (Haqiba-register):
`انت ما قتلى لي... كلمتنى عيونى / عن مشاعر صادقة حية ... نابعة من اعماق شجونك`

## Lisan-Sudanese (v2)

- **Role:** **Stage C v2** via `sudani.py`; **15% (281 sentences) held out** as `data/interim/sudani/lisan_holdout.jsonl`, the `lisan_holdout` eval · **Path:** `data/raw/lisan/` · **Tokens:** 47.0K (1,878 unique sentences)
- **License:** CC-BY-4.0 · **Source:** mirrored inside https://huggingface.co/datasets/AymanMansour/Lisan-Sudanese-TTS-Dataset (+ `New-...` twin, identical text) — the full Lisan corpus is at sina.birzeit.edu/currasat behind a request form

The cleanest verified-dialect Sudanese that publicly exists: Facebook/YouTube comments,
native-speaker verified, morphologically annotated upstream. The holdout matters more than the
training tokens: it shares no people with the WhatsApp data and is native dialect rather than
Flores translationese, so it is the one metric that catches contact-memorisation and
translator-Sudanese at the same time.

**Example:** `مشيت لقيت المظاهرات في نص الحلة خشيت أكورك مع الناس و هم يجدعوا في البمبان`

## Organic Sudanese sample (v2)

- **Role:** **Stage C v2** via `sudani.py` (154 of 300 rows survive cleaning) · **Path:** `data/raw/organic_sudanese/` · **Tokens:** 3.6K
- **License:** CC-BY-4.0 · **Source:** https://huggingface.co/datasets/ebubekr53/organic-sudanese-arabic-dialect-dataset

Spontaneous messages from a live dialect-translation app. Genuinely conversational
(`متين انا وانت نخرج مع بعضنا`, `اها قلت شنو`) but with foreign probe rows ("paard") and some
non-Sudanese dialect mixed in; majority-Arabic + length filters apply. Tiny; kept for register.

## Alexandria — SD subset (v2)

- **Role:** ⚠️ **reserved: eval/seed material only, never training** · **Path:** `data/raw/alexandria_sd/` · **Size:** 345 conversations (test split only — SD ships no train/dev)
- **License:** CC-BY-NC-4.0 · **Source:** https://huggingface.co/datasets/UBC-NLP/alexandria

English↔Sudanese parallel *conversations* — the only multi-turn parallel Sudanese resource
anywhere. Too small to train on and too valuable as an independent check on synthetic chat
quality; reserved accordingly.

## oddadmix Sudanese transcripts (v2)

- **Role:** **Stage C v2**, `repeat: 2` · **Path:** `data/raw/oddadmix/{sudan_podcast,nuuar,ahmed_gobara}.jsonl` · **Tokens:** **2.49M** (255 episodes → 6,545 docs of ~1.5K chars)
- **License:** unstated; audio scraped from YouTube, transcripts AI-generated + QC'd — **private use only, never redistributed** · **Source:** https://huggingface.co/datasets/oddadmix/arabic-audio-collection-sudanese-sudan-podcast (+ `-nuuar`, `-ahmed-gobara`)

~200 hours of transcribed spoken Sudanese — the closest public register to WhatsApp chat and
~10× all other public Sudanese text combined. Fetched via column projection over `hf://` so the
~14 GB of audio never touched disk. `src/preprocessing/oddadmix.py` strips the transcriber's
full diacritization (nothing in the target distribution is vowelled), production tags, and
ASR stutter runs; splits by episode. **Audit (2026-08-22): 25/30 random docs clearly Sudanese
conversational** — the remainder is formal-MSA guest speech and one Gulf-dialect interviewee;
gate (≥80%) passed.

**Example:** `طبعا نحن متين شعرنا بالكلام ده؟ لمن طلعنا من السودان. شوفنا المعاملة كيف.`

## sudaneseonline.com forum crawl (v2)

- **Role:** **Stage C v2**, dialect-ranked: the mixture takes threads with dialect score ≥0.5 ≈ **19.6M tokens** (~5.8M at ≥0.8) of the **~243M** extracted · **Path:** `data/raw/sudaneseonline/` (`html/` 72,761 gzipped thread pages = 99.4% of the ~73,180 unique threads in the site's sitemaps; the 110,410 sitemap URLs list most threads twice) · **Interim:** `data/interim/sudaneseonline/{train,val}.jsonl`, 69,633 threads, each row carrying a `dialect` probability
- **License:** forum posts, authors' copyright — **private use only, never redistributed**. robots.txt disallows only `/admin/`; the crawl was single-threaded, rate-limited, self-identifying, with error backoff (finished 2026-08-23, 2 hard failures).
- **Procured:** [sudaneseonline.com](https://sudaneseonline.com) static thread archives, crawled by
  `scripts/scrape_sudaneseonline.py`: thread URLs enumerated from the site's sitemap index, ids
  parsed with `[-/](\d+)\.html?` (threads appear under two URL shapes), one gzipped page per thread.

25+ years of Sudanese forum discussion — the largest native-Sudanese text reservoir on the open
web. Site quirks handled by `src/preprocessing/sudaneseonline.py`: per-line repair of the
2015-era UTF-8-as-cp1256 double-encoding (irreparable lines dropped), site-boilerplate line
stripping, quote-pyramid/signature dedup by seen-line, split by thread. Register runs from MSA
news pastes to pure dialect storytelling, so every thread is scored by
`src/preprocessing/dialect_score.py` (char 2–5-gram logistic head, Sudanese-vs-MSA, 98.1%
holdout accuracy; trained on the project's own trusted corpora). Band audit 2026-08-23: ≥0.8 is
genuine dialect discourse (ونسة, banter, colloquial poetry) with zero mojibake; 0.5–0.8 is
dialect-seasoned opinion; <0.3 is MSA news.

**Example (score 0.92):**
`جيت جاري جري علي البيت.. قلت متين اصل.. والعنقريب جاك زول.. فتران فتر شديد وتعبان تعب شديد ..و....نعسااااان`

## Telegram channels (wave 3)

- **Role:** **Stage C v2**, dialect-ranked — **58.14M tokens** total, of which **47.11M gold (≥0.8)** + 4.62M seasoned (0.5–0.8) + 6.42M silver · **Path:** `data/raw/telegram/` (16 channels, 632 MB preview-page JSON) · **Interim:** `data/interim/telegram/{train,val}.jsonl` — 24,996 docs / 190.6M chars, each row carrying `dialect`; val = the **last 2% of docs per channel, contiguous by message id** (adjacent serial-fiction chapters share plot — a random split would leak)
- **License:** channel authors' copyright — **private use only, never redistributed**. Public preview pages, no robots restrictions (`t.me/robots.txt` 404), polite single-threaded crawl.
- **Procured:** `scripts/scrape_telegram.py` — fetches `https://t.me/s/<channel>` (Telegram's public
  web preview, ~20 messages/page of static HTML), paginating with the `?before=<lowest-id>` cursor
  down to message id 1; resumable by page file. The 16 channels (msgs kept after preview
  pagination; **discovery vector:** cross-promo lists posted inside Sudanese channels — the
  directory sites are all 403):
  [nasra8ya](https://t.me/s/nasra8ya) 33,973 · [rwayatSudan](https://t.me/s/rwayatSudan) 46,152 ·
  [novelsforus2](https://t.me/s/novelsforus2) 11,864 · [klam_sudany](https://t.me/s/klam_sudany) 16,167 ·
  [sudan_4g](https://t.me/s/sudan_4g) 22,927 · [w00057777](https://t.me/s/w00057777) 9,802 ·
  [sudanesenovels](https://t.me/s/sudanesenovels) 6,660 · [sudanese_shair](https://t.me/s/sudanese_shair) 3,929 ·
  [comidyann](https://t.me/s/comidyann) 2,410 · [sudanesevip](https://t.me/s/sudanesevip) 2,051 ·
  [nikat7e3n](https://t.me/s/nikat7e3n) 1,362 · [Sd_rewaya3t](https://t.me/s/Sd_rewaya3t) 1,152 ·
  [SudaneseHD](https://t.me/s/SudaneseHD) 805 (أمثال) · [tatwer3](https://t.me/s/tatwer3) 597 (دوبيت) ·
  [sudanes0](https://t.me/s/sudanes0) 499 · [telegraaaammmmmmm](https://t.me/s/telegraaaammmmmmm) 142.
  Diwansha3r is listed in `CHANNELS` but unreachable (web preview disabled → 302, zero widgets).

The purest large dialect source in the project: serial fiction (nasra8ya 29.2M tok, rwayatSudan 9.5M, novelsforus2 8.4M, sudanesenovels 5.4M), poetry/دوبيت, أمثال, ونسة. Processed by `src/preprocessing/telegram.py`: HTML-strip, promo/ad drop (URL-carrying messages with little residual Arabic), per-channel line-dedup ≥12 chars (kills join-us footers — collapsed sudan_4g 22,927 msgs → 447 docs), consecutive messages merged into episode-length docs (~6K chars, split at >200-id gaps). Dialect quartiles **[0.88 / 0.973 / 0.991]** — essentially an all-gold corpus; multiplies the project's gold band ~8× on its own.

**Example (score 0.99, nasra8ya):**
`الرد القبل شوية رديتيهو لي دا بتسميهو شنو ؟ ومن قبيل لاحظت ليك متغيرة حاصل شنو ؟ _ ما حاصل شي خليني أنوم بس مصدعة`

## anasudani.net forum (wave 3)

- **Role:** Stage C v2, dialect-ranked · **Tokens:** 64.6M (gold 9.7M) · **Path:** `data/raw/anasudani/` → `data/interim/anasudani/` · **License:** private use only
- **Procured:** [anasudani.net](http://www.anasudani.net/forum/) — phpBB 3 forum of the Sudanese
  diaspora, frozen ~Jan 2017; **no robots.txt at all** (404 → unrestricted). Crawled complete by
  `scripts/scrape_anasudani.py`: walks `viewforum.php` listings (20 topics/page — pagination step
  MUST be 20, a step of 40 silently halves coverage) to discover topic ids, then fetches each
  `viewtopic.php` page; 404s tombstoned in `dead.txt`, throttling handled by cool-down retries.
  49,559 topics / 68,214 pages. Processed by `src/preprocessing/anasudani.py` (depth-tracking
  `div.content` extraction, BBCode strip, thread-level seen-line dedup).

## alnilin.com articles + comments (wave 3)

- **Role:** comments = Stage C v2 gold (42.6M tok); articles = silver reservoir (251.2M tok) · **Path:** `data/raw/alnilin/` → `data/interim/alnilin_{comments,posts}/` · **License:** private use only
- **Procured:** [alnilin.com](https://www.alnilin.com) — major Sudanese news site with an **open
  WordPress REST API**: `/wp-json/wp/v2/posts` and `/wp-json/wp/v2/comments`, paginated
  `per_page=100&orderby=id&order=asc`, clean JSON, `X-WP-Total` header gives exact scale.
  Crawled complete by `scripts/scrape_alnilin.py` (550,832 articles + 561,752 comments, 2.2 GB).
  Processed by `src/preprocessing/wpjson_clean.py` — comments grouped under their article.

## sudanile.com articles (wave 3)

- **Role:** silver MSA reservoir (188.8M tok) · **Path:** `data/raw/sudanile/` → `data/interim/sudanile_posts/` · **License:** private use only
- **Procured:** [sudanile.com](https://sudanile.com) — news/opinion site, open wp-json (same
  endpoints as alnilin), crawled complete by the generic `scripts/scrape_wpjson.py --name sudanile
  --base https://sudanile.com` (142,731 posts, 0 comments). Columnists drift into dialect,
  hence the small gold band.

## koorasudan.net articles + comments (wave 3)

- **Role:** comments = Stage C v2 football-fan dialect (3.2M tok); articles = silver (49.8M tok) · **Path:** `data/raw/koorasudan/` → `data/interim/koorasudan_{comments,posts}/` · **License:** private use only
- **Procured:** [koorasudan.net](https://koorasudan.net) — Sudanese football news; found in the
  2026-08-31 directory sweep. robots.txt is a bare `User-agent: *` allow-all; open wp-json.
  Crawled complete same day via `scripts/scrape_wpjson.py --name koorasudan` (130,484 posts +
  75,599 comments, 518 MB).

## cover-sd.com articles (wave 3)

- **Role:** silver (0.4M tok) · **Path:** `data/raw/cover_sd/` → `data/interim/cover_sd_posts/` · **License:** private use only
- **Procured:** [cover-sd.com](https://cover-sd.com) (صحيفة كفر ووتر) — small newspaper, clean
  robots, open wp-json; `scripts/scrape_wpjson.py --name cover_sd` (1,847 posts).

## Blogger cluster (waves 2–3)

- **Role:** Stage C v2 — dialect lives in the lyrics/fiction blogs; the delta blogs are MSA (1.3M tok) · **Path:** `data/raw/blogger/` → `data/interim/blogger/` · **License:** private use only
- **Procured:** `scripts/scrape_blogger.py` via the Blogger Atom API —
  `<blog>/feeds/posts/default?max-results=150&start-index=N` returns full post bodies as
  structured XML, no HTML scraping. 12 blogs: originals
  [hageebatalfun](http://hageebatalfun.blogspot.com) (كلمات الحقيبة),
  [sudaneseshortstorieswriters](http://sudaneseshortstorieswriters.blogspot.com),
  [sudanese-novels](http://sudanese-novels.blogspot.com),
  [katabsudsnese](http://katabsudsnese.blogspot.com); 2026-08-31 delta (from
  mtwersd.com/sudanese-blogs/): [unothati](https://unothati.blogspot.com),
  [olive2020](https://olive2020.blogspot.com), [ajba77](https://ajba77.blogspot.com),
  [sudanesemollified](https://sudanesemollified.blogspot.com),
  [salahamza2](https://salahamza2.blogspot.com), [ar-cher](https://ar-cher.blogspot.com),
  [22montser2019](https://22montser2019.blogspot.com),
  [trendsudani](https://trendsudani.blogspot.com). Processed by `src/preprocessing/blogger.py`.

## aghaniwamthal.com (wave 3)

- **Role:** Stage C v2 — proverbs + Haqiba lyrics (0.24M tok) · **Path:** `data/raw/small_sites/aghaniwamthal/` → `data/interim/small_sites/aghaniwamthal/` · **License:** private use only
- **Procured:** [aghaniwamthal.com](https://aghaniwamthal.com) (أجمل الأمثال والأغاني السودانية) —
  same-domain BFS mirror by `scripts/scrape_small_sites.py --site aghaniwamthal --base
  https://aghaniwamthal.com` (2,039 pages / 54 MB; robots permissive). Processed by
  `src/preprocessing/small_sites.py` (document-frequency boilerplate removal: any line on >5%
  of pages is chrome).

## Dead-forum archives — Wayback + Common Crawl (wave 3, crawling)

- **Role:** Stage C v2 once complete · **Path:** `data/raw/wayback/<domain>/`, `data/raw/commoncrawl/<domain>/` → `data/interim/vbarchive/<domain>/` · **License:** private use only
- **Procured:** two miners over the same dead Sudanese forums, cross-source deduped at
  preprocessing. **Wayback** (`scripts/scrape_wayback.py`): CDX enumeration
  (`web.archive.org/cdx/search/cdx?url=<domain>/*&matchType=domain`) then raw snapshot fetch via
  the `id_` endpoint at ~1 req/s. Domains: sudanyat.org, mugrn.net, algorer.net, sudanelite.com,
  alhasahisa.org, hurriyatsudan.com (vBulletin, `archive/index.php/t-N.html`) + sudanesesongs.net
  (IPB lyrics forum, added in the final sweep). **Common Crawl**
  (`scripts/scrape_commoncrawl.py`): per-collection CDX index queries filtered to
  `status:200` (the forums 406'd CC's UA in some years), then byte-range GETs against
  `data.commoncrawl.org`; payloads are windows-1256. wadmadani.com is covered by this route
  (its live crawl was stopped when the site began 403ing our UA). Processed by
  `src/preprocessing/vbarchive.py` (vB archive / vB showthread / IPB post containers,
  cp1256+mojibake repair, domain-level line dedup).

## muzammilsoft/Sudanese_dialect_dataset (wave 3, pending QC)

- **Role:** candidate for the synthetic pool only — **LLM-generated**, never counts as organic · **Path:** `data/raw/hf_muzammilsoft/` (13.6 MB, 2 jsonl files) · **License:** CC-BY-4.0
- **Procured:** [HuggingFace](https://huggingface.co/datasets/muzammilsoft/Sudanese_dialect_dataset),
  published 2026-08-27, downloaded via `hf_hub_download`. Instruction/input/output schema with
  "THINKING" traces; enters training only if it passes our own synthesis QC suite.

## Synthesis pipeline artifacts (v2)

- **Generated by:** `src/synthesis/synth_data.py` (e.g. `synth_data sonnet 1000 gemma3:27b 1000`) —
  persona-card + seed-sampled generation against the models in `MODEL_REGISTRY`; QC'd on demand by
  `src/synthesis/qc.py` (`filtered.jsonl` is the kept pool). Full method record: `data/SYNTHSHEET.md`
  and `src/synthesis/report.md`.
- **Path:** `data/interim/synthetic/` — `blocklist.json` (the DECISIONS.md off-record rule,
  enforced by `src/synthesis/blocklist.py` asserts), `card_inputs/` (42 persona-card
  distillation prompts + owner card — these files are exactly what leaves the machine during
  card compilation), `pseudonym_map.json` (retained but inactive, see below).
- **Naming policy (owner decision 2026-08-23):** cards, seeds and synthetic data carry **real
  names**, including real WhatsApp sender names as speaker labels, so synthetic chats align
  with the real corpus. The earlier pseudonymization layer is kept behind
  `POLICY = "pseudonyms"` in `src/synthesis/pseudonyms.py` but is off. Phones, emails and
  handles are still masked in anything sent to Claude. `elaf-osman` stays excluded everywhere
  regardless of policy.

## Situation bank (v2 synthesis seed artifact)

- **Path:** `data/interim/synthetic/situations.jsonl` (~2,000 entries) · **Built:** 2026-08-27, offline, via Verbalized Sampling (Claude Sonnet, k=8 candidates with probabilities per call)
- **Role:** the only topic source the generators ever see — two-sentence Sudanese situations with a causal hook, rooted in the personas' measured topic distributions × an attribute grid (time-of-day / emotional valence / arc / media). Replaces flat topic nouns; the generator never free-chooses a topic (mode-collapse guard). Regenerable with `python -m src.synthesis.situations build`.

### Acquisition wave 3 — provenance notes

Per-dataset numbers live in the **Summary table** at the top (single source of truth);
per-site crawl verdicts, exclusions (alrakoba, altaghyeer, sudanakhbar, waslat, the
Forumotion namespace — all `ai-train=no` and/or AI-crawler disallows, respected), and the
final-sweep audit trail live in `data/SCRAPESHEET.md`. All wave-3 crawls: single-threaded,
rate-limited, honest UA, robots-respecting; private use only, never redistributed.
Telegram delta detail worth keeping here: 11 channels, 124,150 msgs, every channel
paginated to message id 1 (shortfalls vs id-ceilings are channel deletions, not crawl gaps);
Diwansha3r remains unreachable (web preview disabled).
