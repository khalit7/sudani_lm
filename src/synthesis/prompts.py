"""Prompt templates for the synthesis pipeline, versioned as code (plan.md Part IV, step 2.2-2.4).

Three request kinds share one skeleton: a stable style preamble (cacheable), the persona
material, then the volatile seed + directives. Two rules baked into every template:

  - the model writes plain `NAME: text` lines, never the training template's special tokens —
    rendering through src.tokenizer.special_tokens.render_conversation happens locally, so a
    malformed generation can fail parsing but can never corrupt the token stream;
  - every request carries real seed text: models author Sudanese poorly cold but extend real
    material well, so open-ended generation is banned by construction.

PROMPT_VERSION is recorded with every raw generation so a prompt change never silently mixes
two distributions in one output pool.
"""

# v3 (owner-approved seed redesign, 2026-08-27): situation-grounded seeds from the offline
# bank, n-party chats with empirical participation imbalance, writer-persona monologues with
# a genre × audience grid. v2's length + anti-closure rules carry over.
PROMPT_VERSION = "v3"

STYLE_PREAMBLE = """\
You are a native Sudanese Arabic speaker generating WhatsApp-style conversation data in
authentic Sudanese dialect (اللهجة السودانية). Non-negotiable rules:

- Sudanese, not generic Arabic: زول، شنو، وين، كيفن، ياخ، خلاص، براهو، هسة/هسي، عشان،
  قروش، سمح، شديد — natural Sudanese function words, never MSA (لماذا، سوف، ماذا) and never
  Egyptian/Levantine forms (إزيك، ليش، هيك).
- WhatsApp register: short turns (2–15 words mostly), no punctuation ceremony, emoji where
  natural, occasional Arabizi/English code-switching only where the personas do it.
- Typos, vowel-lengthening (شدييييد), and inconsistent spelling are FEATURES of the register —
  reproduce the habits shown in the material, do not clean them up.
- Real texture: people misunderstand, change subject, answer late, tease, leave things
  unresolved. Avoid tidy assistant-like closure.
- Output format: one turn per line, exactly `NAME: text`. No markdown, no numbering, no
  narration, no translation, nothing but the turns."""

CHAT_TEMPLATE = """{preamble}

=== OWNER CARD (K) ===
{owner_card}

=== PERSONA: {partner_name} ===
{partner_card}

=== REAL EXCERPT (K and {partner_name}, {date}) ===
{seed}

=== TASK ===
Write ONE new WhatsApp conversation between K and {partner_name}: {n_turns} turns, speakers
strictly alternating or naturally bursty like the excerpt. Topic: {topic}. It must be a NEW
scenario — do not retell or continue the excerpt's events; the excerpt is a style anchor only.
K mirrors {partner_name}'s register as the owner card describes.

A conversation this long must behave like a real long chat, not a scene from a script:
- the topic DRIFTS — start on the given topic, wander into 2-3 others the way these two
  actually do, maybe circle back
- include dead air the register allows: a question that never gets answered, a burst of
  three messages before the reply, a one-emoji turn
- NO wrap-up: real chats stop mid-flow (someone stops replying, says طيب or باي mid-topic,
  or just leaves it). Do not write an ending that resolves things.

Begin directly with the first turn."""

# Jais-tuned strict variant (bake-off follow-up): Jais-2-8B wrote decent content but broke
# format 95/98 times — preambles, meta-commentary, described emojis. Shorter chats, and the
# format contract is spelled out as hard prohibitions.
JAIS_CHAT_TEMPLATE = """{preamble}

=== OWNER CARD (K) ===
{owner_card}

=== PERSONA: {partner_name} ===
{partner_card}

=== REAL EXCERPT (K and {partner_name}, {date}) ===
{seed}

=== TASK ===
Write ONE new WhatsApp conversation between K and {partner_name}: {n_turns} turns.
Topic: {topic}. New scenario — the excerpt is a style anchor only.

OUTPUT CONTRACT — violating any rule makes the output worthless:
- Output ONLY the conversation. Nothing before the first turn. Nothing after the last turn.
- Every single line must be exactly `NAME: message` — no other line shape exists.
- No introduction, no summary, no explanation, no commentary, no headings, no translation.
- Emojis are typed as emojis (😂 ❤️ 🌚) inside messages — NEVER described in words, never
  written as (يضحك) or *laughs* or [emoji].
- No narration or stage directions of any kind.

First line = first turn. Last line = last turn. Begin now."""

PAIR_CHAT_TEMPLATE = """{preamble}

=== PERSONA: {name_a} ===
{card_a}

=== HOW {name_a} ACTUALLY WRITES (real excerpt from another chat) ===
{excerpt_a}

=== PERSONA: {name_b} ===
{card_b}

=== HOW {name_b} ACTUALLY WRITES (real excerpt from another chat) ===
{excerpt_b}

=== TASK ===
Write ONE new WhatsApp conversation between {name_a} and {name_b} — K is NOT in this chat:
{n_turns} turns. They know each other{relationship}. Topic: {topic}. Each keeps their OWN
register from their card and excerpt — do not let their voices converge.

A conversation this long must behave like a real long chat, not a scene from a script:
- the topic DRIFTS — start on the given topic, wander into 2-3 others, maybe circle back
- include dead air the register allows: an unanswered question, a burst of three messages
  before the reply, a one-emoji turn
- NO wrap-up: real chats stop mid-flow. Do not write an ending that resolves things.

One turn per line, exactly `NAME: text`, using only these two names. Begin directly."""

GROUP_CHAT_TEMPLATE = """{preamble}

{cards}

=== REAL GROUP-CHAT EXCERPT ("{group_name}", {date}) ===
{seed}

=== TASK ===
Write ONE new conversation in the same WhatsApp group ("{group_name}"): {n_turns} turns,
speakers ONLY from: {speakers}. It must be a NEW scenario — the excerpt is a style and
dynamics anchor only. Keep the group's real texture: people talk over each other, some
members dominate, side-conversations interleave, jokes get dogpiled, someone asks a question
that drowns. Not every listed speaker needs equal turns — mirror group dynamics, not fairness.

NO wrap-up ending; group chats just move on or go quiet.

One turn per line, exactly `NAME: text`. Begin directly."""

MONOLOGUE_TEMPLATE = """{preamble}

=== REAL SUDANESE TEXT (style anchor) ===
{seed}

=== TASK ===
Write ONE piece of connected Sudanese-dialect prose of {n_words} words — {genre}. Same dialect
register as the anchor, but a NEW subject: {topic}. This is discourse, not chat: flowing
paragraphs in the voice of someone talking, like a voice note or a forum post. No headings, no
lists, no MSA news register.

At this length the danger is padding: do not restate the same point in new words, do not
inflate with rhetorical filler. Instead EXPAND the way a storyteller does — more scenes, more
named details, a digression that returns, dialogue snippets inside the telling. Every
paragraph should add something that was not there before. Output only the text."""

TRANSFORM_TEMPLATE = """{preamble}

=== PARALLEL EXAMPLES (MSA → Sudanese) ===
{exemplars}

=== MSA SOURCE ===
{source}

=== TASK ===
Rewrite the MSA source in natural Sudanese dialect, following the parallel examples' register:
not word-by-word substitution but how a Sudanese person would actually say it. Keep the
information; change the voice. Output only the Sudanese text."""

CHAT_V3_TEMPLATE = """{preamble}

{cards}

=== REAL EXCERPT (style anchor ONLY — its topic is irrelevant, do not reuse its events) ===
{excerpt}

=== SITUATION ===
{situation}

=== TASK ===
Write ONE new WhatsApp conversation between {roster}: {n_turns} turns total.
The conversation grows out of the SITUATION above — start inside it, not with greetings
ceremony. The excerpt exists only to show how these people actually type.

{imbalance}Real long-chat texture is mandatory:
- the topic DRIFTS — start on the situation, wander into 2-3 other things, maybe circle back
- dead air is allowed: an unanswered question, a burst of three messages before a reply,
  a one-emoji turn
- NO wrap-up ending: real chats stop mid-flow.

Every line exactly `NAME: message`, using only the listed names. Nothing before the first
turn, nothing after the last. Begin now."""

MONO_V3_TEMPLATE = """{preamble}

{writer_block}=== REAL SUDANESE TEXT (style anchor ONLY — its topic is irrelevant) ===
{anchor}

=== SITUATION ===
{situation}

=== TASK ===
Write ONE piece of connected Sudanese-dialect prose of {n_words} words — {genre},
موجهة لـ{audience}. It grows out of the SITUATION above: tell it, argue it, or riff on it —
a NEW angle, not a retelling of the anchor.{writer_rule}

At this length the danger is padding: never restate a point in new words. EXPAND like a
storyteller — more scenes, named details, a digression that returns, dialogue snippets inside
the telling. No headings, no lists, no MSA news register. Output only the text."""

MONO_AUDIENCES = ["أصحابه القراب", "قروب الدفعة", "ناس المنتدى", "أهل البيت"]

MONO_TOPICS = ["الغربة", "الكهرباء والمويه", "امتحانات الجامعة", "العرس",
               "رمضان في السودان", "الاسعار", "الاهل", "الشغل", "الكورة",
               "المطر والخريف", "الجيران", "المواصلات", "السوق", "الحله زمان",
               "اول يوم في الشغل", "السفر بالبص", "عيد الاضحيه", "الدراسه بره"]

MONOLOGUE_GENRES = [
    "قصة من الذاكرة يحكيها زول لأصحابه",
    "رأي حاد في موضوع الساعة، بأسلوب منشور منتدى",
    "وصف يوم عادي في الخرطوم أو مدينة سودانية",
    "نصيحة من شخص كبير لشخص صغير",
    "حكاية مضحكة حصلت في العرس أو الجامعة أو الشغل",
    "تعليق على مباراة أو مسلسل أو أغنية",
]


def chat_v3_prompt(cards_block, roster, excerpt, situation, n_turns, imbalance=""):
    return CHAT_V3_TEMPLATE.format(
        preamble=STYLE_PREAMBLE, cards=cards_block, roster=roster, excerpt=excerpt,
        situation=situation, n_turns=n_turns,
        imbalance=(imbalance + "\n") if imbalance else "")


def mono_v3_prompt(anchor, situation, genre, audience, n_words, writer_block=""):
    writer_rule = (" Stay strictly in the WRITER's voice as described in their card."
                   if writer_block else "")
    return MONO_V3_TEMPLATE.format(
        preamble=STYLE_PREAMBLE, anchor=anchor, situation=situation, genre=genre,
        audience=audience, n_words=n_words,
        writer_block=(writer_block + "\n\n") if writer_block else "",
        writer_rule=writer_rule)


def chat_prompt(owner_card, partner_name, partner_card, seed_text, date, topic, n_turns):
    return CHAT_TEMPLATE.format(
        preamble=STYLE_PREAMBLE, owner_card=owner_card, partner_name=partner_name,
        partner_card=partner_card, seed=seed_text, date=date,
        topic=topic or "ما يشبه مواضيعهم المعتادة", n_turns=n_turns)


def jais_chat_prompt(owner_card, partner_name, partner_card, seed_text, date, topic, n_turns):
    return JAIS_CHAT_TEMPLATE.format(
        preamble=STYLE_PREAMBLE, owner_card=owner_card, partner_name=partner_name,
        partner_card=partner_card, seed=seed_text, date=date,
        topic=topic or "ما يشبه مواضيعهم المعتادة", n_turns=n_turns)


def pair_chat_prompt(name_a, card_a, excerpt_a, name_b, card_b, excerpt_b,
                     relationship, topic, n_turns):
    return PAIR_CHAT_TEMPLATE.format(
        preamble=STYLE_PREAMBLE, name_a=name_a, card_a=card_a, excerpt_a=excerpt_a,
        name_b=name_b, card_b=card_b, excerpt_b=excerpt_b,
        relationship=f" ({relationship})" if relationship else "",
        topic=topic or "ما يشبه مواضيعهم المعتادة", n_turns=n_turns)


def group_chat_prompt(cards, group_name, date, seed_text, speakers, n_turns):
    return GROUP_CHAT_TEMPLATE.format(
        preamble=STYLE_PREAMBLE, cards=cards, group_name=group_name, date=date,
        seed=seed_text, speakers=", ".join(speakers), n_turns=n_turns)


def monologue_prompt(seed_text, genre, topic, n_words=450):
    return MONOLOGUE_TEMPLATE.format(
        preamble=STYLE_PREAMBLE, seed=seed_text, genre=genre, topic=topic, n_words=n_words)


def transform_prompt(exemplars, source):
    return TRANSFORM_TEMPLATE.format(
        preamble=STYLE_PREAMBLE, exemplars=exemplars, source=source)
