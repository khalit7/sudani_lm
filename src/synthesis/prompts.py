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

PROMPT_VERSION = "v1"

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
K mirrors {partner_name}'s register as the owner card describes. Begin directly with the first
turn."""

MONOLOGUE_TEMPLATE = """{preamble}

=== REAL SUDANESE TEXT (style anchor) ===
{seed}

=== TASK ===
Write ONE piece of connected Sudanese-dialect prose of {n_words} words — {genre}. Same dialect
register as the anchor, but a NEW subject: {topic}. This is discourse, not chat: flowing
paragraphs in the voice of someone talking, like a voice note or a forum post. No headings, no
lists, no MSA news register. Output only the text."""

TRANSFORM_TEMPLATE = """{preamble}

=== PARALLEL EXAMPLES (MSA → Sudanese) ===
{exemplars}

=== MSA SOURCE ===
{source}

=== TASK ===
Rewrite the MSA source in natural Sudanese dialect, following the parallel examples' register:
not word-by-word substitution but how a Sudanese person would actually say it. Keep the
information; change the voice. Output only the Sudanese text."""

MONOLOGUE_GENRES = [
    "قصة من الذاكرة يحكيها زول لأصحابه",
    "رأي حاد في موضوع الساعة، بأسلوب منشور منتدى",
    "وصف يوم عادي في الخرطوم أو مدينة سودانية",
    "نصيحة من شخص كبير لشخص صغير",
    "حكاية مضحكة حصلت في العرس أو الجامعة أو الشغل",
    "تعليق على مباراة أو مسلسل أو أغنية",
]


def chat_prompt(owner_card, partner_name, partner_card, seed_text, date, topic, n_turns):
    return CHAT_TEMPLATE.format(
        preamble=STYLE_PREAMBLE, owner_card=owner_card, partner_name=partner_name,
        partner_card=partner_card, seed=seed_text, date=date,
        topic=topic or "ما يشبه مواضيعهم المعتادة", n_turns=n_turns)


def monologue_prompt(seed_text, genre, topic, n_words=450):
    return MONOLOGUE_TEMPLATE.format(
        preamble=STYLE_PREAMBLE, seed=seed_text, genre=genre, topic=topic, n_words=n_words)


def transform_prompt(exemplars, source):
    return TRANSFORM_TEMPLATE.format(
        preamble=STYLE_PREAMBLE, exemplars=exemplars, source=source)
