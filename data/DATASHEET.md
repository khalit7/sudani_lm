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

## Summary

| Dataset | Role | Local path | Tokens | Size (MB) | License |
|---|---|---|---|---|---|
| [ArabicWeb24](https://huggingface.co/datasets/lightonai/ArabicWeb24) | **Stages A & B** pretraining (5B of 13.49B) **+ 40% replay in Stage C** | `data/raw/arabicweb24/` | **33.9B** | 189,378 | ODC-BY |
| [SmolKalam](https://huggingface.co/datasets/AdaMLLab/SmolKalam-Arabic-Conversational-SFT) | ⚠️ **not used** — only fed the SFT stage the ablation dropped | `data/raw/smolkalam/` | **1.62B** | 3,366 | Apache-2.0 |
| [InstAr-500k](https://huggingface.co/datasets/ClusterlabAi/InstAr-500k) | ⚠️ **not used** — never entered any packed stage | `data/raw/instar500k/` | **147.2M** | 1,043 | Apache-2.0 |
| WhatsApp export (personal) | **Stage C** (all of it) **and Stage D** (owner-heavy chats, masked); held-out chats are the main metric | `data/raw/whatsapp/` | **10.47M** | 138 | private, not redistributable |
| [ArabicMMLU](https://huggingface.co/datasets/MBZUAI/ArabicMMLU) | Eval — MSA forgetting guard in Stages C/D | `data/raw/arabicmmlu/` | **556.4K** | 8 | CC-BY-NC-4.0 |
| [SudSenti](https://github.com/mustafa20999/Sudanese-Arabic-Sentiment-Datasets) | **Stage C**, upsampled ×3 | `data/raw/sudsenti/` | **381.6K** | 2 | unstated (academic release) |
| [Sudanese_Dialect_Tweet_Tele](https://huggingface.co/datasets/arbml/Sudanese_Dialect_Tweet_Tele) | **Stage C**, upsampled ×3 | `data/raw/sudanese_tweets_tele/` | **204.0K** | 1 | unstated |
| [Sudanese_Flores](https://huggingface.co/datasets/McGill-NLP/Sudanese_Flores) | Eval — independent Sudanese signal (DEV tracked; DEVTEST unused, held back) | `data/raw/sudanese_flores/` | **140.9K** | 1 | unstated (FLORES derivative) |
| [Sudanese_Dialect_Tweet](https://huggingface.co/datasets/arbml/Sudanese_Dialect_Tweet) | **Stage C**, upsampled ×3 | `data/raw/sudanese_tweets/` | **50.0K** | 1 | unstated |

**Total on disk: 193,936 MB ≈ 189.4 GB** (`data/raw/`), holding **≈35.7B tokens**.

ArabicWeb24 is 97.6% of that footprint. Everything targeting Sudanese — the three public corpora
plus the personal chat export — is **11.1M tokens**, or 0.03% of the total.

### Data expansion v2 (plan.md Part IV, acquired 2026-08-22)

| Dataset | Role | Local path | Tokens | License |
|---|---|---|---|---|
| [Tarab — Sudanese slice](https://huggingface.co/datasets/drelhaj/Tarab) | **Stage C v2**, capped ~2% of mixture | `data/raw/tarab/` | **428.8K** (post-dedup) | CC-BY-4.0 |
| [Lisan-Sudanese](https://huggingface.co/datasets/AymanMansour/Lisan-Sudanese-TTS-Dataset) (via TTS mirrors) | **Stage C v2** + **Lisan holdout eval** (15% held out) | `data/raw/lisan/` | **47.0K** | CC-BY-4.0 |
| [Organic Sudanese sample](https://huggingface.co/datasets/ebubekr53/organic-sudanese-arabic-dialect-dataset) | **Stage C v2** (via `sudani.py`) | `data/raw/organic_sudanese/` | **3.6K** | CC-BY-4.0 |
| [Alexandria — SD subset](https://huggingface.co/datasets/UBC-NLP/alexandria) | ⚠️ **reserved** — eval/seed material only, not training | `data/raw/alexandria_sd/` | 345 conversations | CC-BY-NC-4.0 |
| oddadmix Sudanese transcripts ([podcast](https://huggingface.co/datasets/oddadmix/arabic-audio-collection-sudanese-sudan-podcast), nuuar, gobara) | **Stage C v2** | `data/raw/oddadmix/` | **2.49M** (255 episodes, 6,545 docs) | unstated (scraped YouTube) — private use only |
| [sudaneseonline.com forum crawl](https://sudaneseonline.com/board/index.htm) | **Stage C v2**: dialect-scored, mixture takes score ≥0.5 ≈ **19.6M tokens** | `data/raw/sudaneseonline/` | **~243M** total (69,633 threads; 99.4% of unique threads) | site content, authors' copyright — private use only |

The v2 rule of thumb: every source lands in `data/raw/<name>/`, gets a preprocessing module
producing `data/interim/<name>/{train,val}.jsonl` split by container (episode / thread / song —
never by post), and enters training only through a mixture manifest (`configs/mixtures/*.yaml`)
that records per-source token counts in the pack's `meta.json`.

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

## Synthesis pipeline artifacts (v2)

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

### Acquisition wave 3 (audited 2026-08-28, crawls in progress)

| Source | Category | Est. volume | Dialect | Status |
|---|---|---|---|---|
| Telegram public channels (5 complete: novelsforus2, klam_sudany, sudanesenovels, Sd_rewaya3t, sudanes0; Diwansha3r unreachable — web preview disabled) | social/serial fiction | 1,953 preview pages, **162 MB** raw | **very high** (pure عامي) | **crawled** (`data/raw/telegram/`, `scripts/scrape_telegram.py`) |
| [alnilin.com](https://www.alnilin.com) via open WP REST | news + **561,752 reader comments** | ~550K articles + comments | articles low (MSA) / comments **high** | crawling (`data/raw/alnilin/`, `scripts/scrape_alnilin.py`) |
| anasudani.net forum (phpBB, frozen 2017, 1.16M posts, no robots restrictions) | forum | ~1.16M posts | high | queued — scraper next |
| Dead vBulletin forums via Wayback (sudanyat, mugrn, algorer, sudanelite, alhasahisa) | forums | unknown | high | queued |
| **Excluded on explicit refusal**: alrakoba.net + vb.alrakoba.net (145K threads), altaghyeer.info, sudanakhbar.com — all carry `Content-Signal: ai-train=no` and/or disallow AI crawlers; respected | — | — | — | excluded |

All wave-3 crawls: single-threaded, rate-limited, honest UA, robots-respecting; private use only, never redistributed.
