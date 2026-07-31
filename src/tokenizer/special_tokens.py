"""Special tokens and the chat template, defined once so every stage renders text identically.

These must exist in the tokenizer *before* pretraining: adding them later invalidates every
embedding. The chat markers are single tokens on purpose — as plain text `[inst]` costs 4 tokens
per turn, which is a large tax when the median chat message is only a handful of tokens long.
"""

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"
PAD = "<pad>"

# Chat structure. One template serves both instruction tuning and real chat, so there is a
# single rendering path everywhere.
CONV_START = "<|conv|>"
TURN_START = "<|turn|>"
TURN_END = "<|end|>"

CORE_SPECIAL_TOKENS = [BOS, EOS, UNK, PAD]
CHAT_SPECIAL_TOKENS = [CONV_START, TURN_START, TURN_END]
ALL_SPECIAL_TOKENS = CORE_SPECIAL_TOKENS + CHAT_SPECIAL_TOKENS

# Speaker labels used when rendering instruction data as a two-turn conversation.
USER_SPEAKER = "USER"
ASSISTANT_SPEAKER = "ASSISTANT"
# Speaker label for the repository owner's own WhatsApp messages.
SELF_SPEAKER = "ME"


def render_turn(speaker: str, text: str) -> str:
    """One conversational turn: <|turn|>SPEAKER: text<|end|>"""
    return f"{TURN_START}{speaker}: {text}{TURN_END}"


def render_conversation(turns, add_bos: bool = True, add_eos: bool = True) -> str:
    """`turns` is a sequence of (speaker, text) pairs."""
    body = CONV_START + "".join(render_turn(speaker, text) for speaker, text in turns)
    return f"{BOS if add_bos else ''}{body}{EOS if add_eos else ''}"
