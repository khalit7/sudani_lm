from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerFast

tokenizer_root = Path("~/sudani_lm/tokenizers").expanduser()

# v2_32k: 32k vocab, byte fallback, trained on Arabic + Sudanese + chat, with the chat markers
# as single tokens. The packed data under data/packed/ is built with this vocabulary and its
# fingerprint is asserted at load time, so the two cannot drift apart.
#
# The previous 8k "init_tokenizer" is still on disk to reproduce pre-v2 runs, but nothing in the
# training path should use it: its ids do not match the packed corpus.
DEFAULT_TOKENIZER = "v2_32k"

_tokenizer: PreTrainedTokenizerFast | None = None


def get_tokenizer(name: str = DEFAULT_TOKENIZER) -> PreTrainedTokenizerFast:
    global _tokenizer
    if _tokenizer is None or getattr(_tokenizer, "_sudani_name", None) != name:
        _tokenizer = AutoTokenizer.from_pretrained(tokenizer_root / name)
        _tokenizer._sudani_name = name
    return _tokenizer
