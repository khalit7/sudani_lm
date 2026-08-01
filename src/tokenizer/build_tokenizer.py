"""Build the v2 tokenizer: 32k vocab, byte fallback, trained on Arabic + Sudanese + chat.

Replaces build_init_tokenizer.py (8k, ArabicWeb24 only, no byte fallback). Three things drove
the rebuild:

  * 15.6% of the target domain (WhatsApp) is Latin-script Arabizi, which an Arabic-only vocab
    shatters — "kaif al7al ya zol" costs 11 tokens under the old tokenizer.
  * The chat markers the model must learn cost 4 tokens each as plain text.
  * No byte fallback meant emoji could become <unk>, and emoji are frequent in chat.

The public Sudanese corpora are upsampled heavily. At 0.64M tokens against ~2.9 GB of MSA they
would otherwise contribute no dialect merges at all.

This is irreversible after pretraining — changing the vocabulary invalidates every embedding —
so run the gate at the bottom before packing anything.

Usage:  python -m src.tokenizer.build_tokenizer
"""

import csv
import glob
import json
import random
from pathlib import Path

from datasets import Dataset
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from src.tokenizer.special_tokens import (
    ALL_SPECIAL_TOKENS,
    BOS,
    EOS,
    PAD,
    UNK,
)

csv.field_size_limit(10**9)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
SAVE_DIR = REPO_ROOT / "tokenizers" / "v2_32k"

SEED = 67
VOCAB_SIZE = 32_000

# Corpus balance is the lever that decides whether dialect and Arabizi earn vocab slots.
#
# A first attempt used 2,500 docs/shard (2.9 GB) with x50/x20 upsampling. That left chat at only
# ~13% of training characters — and since just 15.6% of WhatsApp characters are Latin, effective
# Arabizi was ~2% of the corpus. No Latin merges formed: "kaif al7al ya zol" still cost 9 tokens
# against a target of 5.
#
# So the Arabic sample is cut (1.45 GB is still ample for MSA merges at 32k) and chat upsampling
# raised, putting chat at ~45% of characters. MSA remains the substrate the model pretrains on,
# but chat is the deliverable, and the tokenizer should be sized for what it must compress well.
#
# Stratified across shards rather than taken from the first few: the corpus is ordered by crawl,
# so consecutive shards skew toward the same domains.
ARABIC_DOCS_PER_SHARD = 1_250
SUDANESE_UPSAMPLE = 150
WHATSAPP_UPSAMPLE = 60

BYTE_TOKENS = [f"<0x{i:02X}>" for i in range(256)]


def _log(msg):
    print(msg, flush=True)


def iter_arabicweb24():
    shards = sorted(
        glob.glob(str(DATA_RAW / "arabicweb24" / "**" / "arabic_web24-train-*-of-*.arrow"),
                  recursive=True)
    )
    if not shards:
        raise FileNotFoundError(f"no ArabicWeb24 shards under {DATA_RAW/'arabicweb24'}")
    _log(f"[arabicweb24] {len(shards)} shards x {ARABIC_DOCS_PER_SHARD} docs")
    rng = random.Random(SEED)
    for n, path in enumerate(shards, 1):
        ds = Dataset.from_file(path)
        take = min(ARABIC_DOCS_PER_SHARD, len(ds))
        for i in rng.sample(range(len(ds)), take):
            text = ds[i]["text"]
            if text:
                yield text
        if n % 50 == 0:
            _log(f"[arabicweb24] {n}/{len(shards)} shards")


def _sudanese_texts():
    texts = []
    for name, column in (("sudanese_tweets", "Tweet"),
                         ("sudanese_tweets_tele", "Tweet_Text")):
        files = glob.glob(str(DATA_RAW / name / "**" / "*.arrow"), recursive=True)
        if not files:
            _log(f"[sudanese] WARNING: {name} not found, skipping")
            continue
        ds = Dataset.from_file(files[0])
        texts += [str(t) for t in ds[column] if t]
    for path in sorted(glob.glob(str(DATA_RAW / "sudsenti" / "*-Tweets.txt"))):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                # SudSenti lines are "text<TAB>label"; keep only the text.
                text = line.split("\t")[0].strip()
                if text:
                    texts.append(text)
    return texts


def _whatsapp_texts():
    path = DATA_RAW / "whatsapp" / "whatsapp.csv"
    if not path.exists():
        _log("[whatsapp] WARNING: whatsapp.csv not found, skipping")
        return []
    texts = []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            if row.get("Type") in ("Incoming", "Outgoing"):
                text = (row.get("Text") or "").strip()
                if text:
                    texts.append(text)
    return texts


def build_corpus_iterator():
    """Yields every training line. Upsampling is done by repetition."""
    sudanese = _sudanese_texts()
    whatsapp = _whatsapp_texts()
    _log(f"[sudanese] {len(sudanese):,} texts x{SUDANESE_UPSAMPLE}")
    _log(f"[whatsapp] {len(whatsapp):,} messages x{WHATSAPP_UPSAMPLE}")

    def generator():
        yield from iter_arabicweb24()
        for _ in range(SUDANESE_UPSAMPLE):
            yield from sudanese
        for _ in range(WHATSAPP_UPSAMPLE):
            yield from whatsapp

    return generator


def train():
    # SentencePiece-style BPE, matching the v1 pipeline (NFKC + Metaspace) but with byte
    # fallback added so no character can ever become <unk>.
    tokenizer = Tokenizer(models.BPE(unk_token=UNK, byte_fallback=True))
    # NFKC is deliberately lossy: it folds Arabic presentation forms and compatibility
    # characters onto their canonical spellings, which stops the vocab wasting slots on
    # variants. It also rewrites a few characters outright (… -> ...), so encode/decode is
    # identity only up to NFKC, not byte-for-byte.
    tokenizer.normalizer = normalizers.NFKC()
    # prepend_scheme="first", not the default "always": "always" prepends the metaspace marker
    # to the segment after *every* special token, so "<|turn|>ME:" decodes as "<|turn|> ME:" —
    # a phantom space injected into every single turn of every conversation.
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(prepend_scheme="first")
    # A Metaspace *decoder* cannot be used here: ByteFallback has to run before the pieces are
    # fused into one string, and once fused Metaspace no longer sees token boundaries, which
    # silently drops every space. Replace/Fuse/Strip is the pipeline Llama uses for the same
    # reason. (Leading whitespace is not preserved — standard sentencepiece behaviour.)
    tokenizer.decoder = decoders.Sequence([
        decoders.Replace("▁", " "),
        decoders.ByteFallback(),   # <0xF0><0x9F>… -> raw bytes -> characters
        decoders.Fuse(),
        decoders.Strip(" ", 1, 0),  # drop the space Metaspace prepends
    ])

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        # Byte tokens must live in the model vocab for byte_fallback to resolve. They are
        # passed here to reserve ids, then demoted out of `added_tokens` below so that
        # decode(skip_special_tokens=True) does not swallow them.
        special_tokens=ALL_SPECIAL_TOKENS + BYTE_TOKENS,
        show_progress=True,
    )

    _log(f"training BPE: vocab_size={VOCAB_SIZE}")
    tokenizer.train_from_iterator(build_corpus_iterator()(), trainer=trainer)
    return tokenizer


def demote_byte_tokens(tokenizer: Tokenizer) -> Tokenizer:
    """Keep the 256 byte tokens in the model vocab but stop marking them special.

    Left as special tokens they are stripped by decode(skip_special_tokens=True), which would
    silently delete every emoji and any other byte-fallback character.
    """
    payload = json.loads(tokenizer.to_str())
    byte_set = set(BYTE_TOKENS)
    before = len(payload.get("added_tokens", []))
    payload["added_tokens"] = [
        t for t in payload.get("added_tokens", []) if t["content"] not in byte_set
    ]
    _log(f"demoted {before - len(payload['added_tokens'])} byte tokens out of added_tokens")
    return Tokenizer.from_str(json.dumps(payload))


def main():
    random.seed(SEED)
    tokenizer = demote_byte_tokens(train())

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=BOS,
        eos_token=EOS,
        unk_token=UNK,
        pad_token=PAD,
    )
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(SAVE_DIR)
    _log(f"saved to {SAVE_DIR}  (vocab {len(fast)})")
    return fast


if __name__ == "__main__":
    main()
