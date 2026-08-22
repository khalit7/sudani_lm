"""Gate for offline packing (Part I, Step 2).

Most of these build their own tiny pack in tmp_path so they run in milliseconds and do not
depend on the multi-GB artifact. The last group validates the real artifact when it exists.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.dataset.packed import PackedDataset, PackedDatasetModule
from src.preprocessing.pack import DTYPE, tokenizer_fingerprint

PACKED_DIR = Path(__file__).resolve().parents[1] / "data" / "packed" / "pretrain"


@pytest.fixture
def tiny_pack(tmp_path):
    """A deterministic 5,000-token stream, so expected values can be computed by hand."""
    tokens = np.arange(5000, dtype=DTYPE)
    path = tmp_path / "train.bin"
    tokens.tofile(path)
    return path, tokens


# --- block geometry -------------------------------------------------------------------------

def test_labels_are_inputs_shifted_by_one(tiny_pack):
    path, _ = tiny_pack
    ds = PackedDataset(path, block_size=128)
    x, y = ds[0]
    assert torch.equal(x[1:], y[:-1]), "labels must be the input window shifted one position"


def test_blocks_do_not_overlap(tiny_pack):
    path, tokens = tiny_pack
    block = 128
    ds = PackedDataset(path, block_size=block)
    for i in (0, 1, 7, len(ds) - 1):
        x, _ = ds[i]
        expected = tokens[i * block : (i + 1) * block].astype(np.int64)
        assert np.array_equal(x.numpy(), expected)


def test_block_count_leaves_room_for_the_label_shift(tiny_pack):
    path, tokens = tiny_pack
    block = 128
    ds = PackedDataset(path, block_size=block)
    assert len(ds) == (len(tokens) - 1) // block
    # the final block must be able to read one token past its end without running off the file
    x, y = ds[len(ds) - 1]
    assert x.shape == (block,) and y.shape == (block,)


def test_every_token_is_covered_exactly_once(tiny_pack):
    path, tokens = tiny_pack
    block = 100
    ds = PackedDataset(path, block_size=block)
    seen = np.concatenate([ds[i][0].numpy() for i in range(len(ds))])
    assert len(seen) == len(ds) * block
    assert len(np.unique(seen)) == len(seen), "no token may appear in two blocks"


def test_rejects_a_stream_too_short_for_one_block(tmp_path):
    path = tmp_path / "tiny.bin"
    np.arange(10, dtype=DTYPE).tofile(path)
    with pytest.raises(ValueError, match="too few"):
        PackedDataset(path, block_size=1024)


# --- dtype contract -------------------------------------------------------------------------

def test_storage_is_uint16_but_batches_are_int64(tiny_pack):
    path, _ = tiny_pack
    ds = PackedDataset(path, block_size=128)
    assert ds.tokens.dtype == np.uint16, "the file must stay 2 bytes/token"
    x, y = ds[0]
    # torch embeddings index with int64, and torch.from_numpy rejects uint16 outright
    assert x.dtype == torch.int64 and y.dtype == torch.int64


def test_uint16_covers_the_vocab():
    assert np.iinfo(DTYPE).max >= 32_000 - 1, "vocab no longer fits in the packed dtype"


# --- the fingerprint guard -------------------------------------------------------------------

def test_stale_pack_is_refused(tmp_path, monkeypatch):
    """A pack is meaningless without the vocabulary that produced it — ids are otherwise
    arbitrary. Loading one built by a different tokenizer must fail loudly."""
    stage_dir = tmp_path / "packed" / "pretrain"
    stage_dir.mkdir(parents=True)
    np.arange(5000, dtype=DTYPE).tofile(stage_dir / "train.bin")
    (stage_dir / "meta.json").write_text(json.dumps({
        "splits": {"train": 5000}, "tokenizer_fingerprint": "deadbeefdeadbeef",
    }))
    monkeypatch.setattr("src.dataset.packed.data_root", tmp_path)

    with pytest.raises(ValueError, match="Repack before training"):
        PackedDatasetModule(stage="pretrain", block_size=128)


def test_matching_fingerprint_is_accepted(tmp_path, monkeypatch):
    stage_dir = tmp_path / "packed" / "pretrain"
    stage_dir.mkdir(parents=True)
    np.arange(5000, dtype=DTYPE).tofile(stage_dir / "train.bin")
    (stage_dir / "meta.json").write_text(json.dumps({
        "splits": {"train": 5000}, "tokenizer_fingerprint": tokenizer_fingerprint(),
    }))
    monkeypatch.setattr("src.dataset.packed.data_root", tmp_path)

    module = PackedDatasetModule(stage="pretrain", block_size=128)
    assert len(module.build_dataset("train")) == (5000 - 1) // 128


def test_missing_pack_names_the_command(tmp_path, monkeypatch):
    monkeypatch.setattr("src.dataset.packed.data_root", tmp_path)
    with pytest.raises(FileNotFoundError, match="src.preprocessing.pack"):
        PackedDatasetModule(stage="pretrain")


# --- collate --------------------------------------------------------------------------------

def test_collate_produces_fixed_shapes_and_no_padding(tiny_pack, tmp_path, monkeypatch):
    path, _ = tiny_pack
    ds = PackedDataset(path, block_size=128)
    module = PackedDatasetModule.__new__(PackedDatasetModule)
    module.block_size = 128

    X, labels = module.colllate_fn([ds[0], ds[1], ds[2]])
    assert X["input_ids"].shape == (3, 128)
    assert labels.shape == (3, 128)
    # attention_mask must be None, not a tensor of ones. Both are numerically identical, but
    # None is what lets the model take SDPA's is_causal flash path instead of materializing an
    # (batch, heads, seq, seq) score matrix — an all-ones mask would silently cost the speedup.
    assert X["attention_mask"] is None


# --- the real artifact, when it exists ---------------------------------------------------------

real_pack = pytest.mark.skipif(
    not (PACKED_DIR / "meta.json").exists(),
    reason="pretraining pack not built — run python -m src.preprocessing.pack pretrain",
)


@real_pack
def test_real_pack_matches_its_metadata():
    meta = json.loads((PACKED_DIR / "meta.json").read_text())
    for split, expected in meta["splits"].items():
        actual = np.memmap(PACKED_DIR / f"{split}.bin", dtype=DTYPE, mode="r").shape[0]
        assert actual == expected, f"{split}.bin holds {actual} tokens, meta claims {expected}"


@real_pack
def test_real_pack_ids_are_within_vocab():
    meta = json.loads((PACKED_DIR / "meta.json").read_text())
    tokens = np.memmap(PACKED_DIR / "train.bin", dtype=DTYPE, mode="r")
    assert int(tokens[:5_000_000].max()) < meta["vocab_size"]


@real_pack
def test_real_pack_has_document_separators():
    meta = json.loads((PACKED_DIR / "meta.json").read_text())
    tokens = np.memmap(PACKED_DIR / "train.bin", dtype=DTYPE, mode="r")
    sample = np.asarray(tokens[:2_000_000])
    assert (sample == meta["eos_id"]).sum() > 100, "documents are not being separated by </s>"


# --- mixture manifests (data expansion v2, plan.md Part IV) -----------------------------------

import inspect

from src.preprocessing.pack import (
    ME_SHARE_THRESHOLD,
    STAGE_C_ME_THRESHOLD,
    _warn_if_threshold_changed,
    pack_manifest,
    pack_stage_c,
    pack_stage_d,
)


def test_stage_defaults_encode_the_ablation_winner():
    """The shipped stage_c pack came from an ad-hoc --me-threshold 1.01 override; the default
    must now say the same thing, or a rebuild silently reverts to the rejected mixture."""
    assert inspect.signature(pack_stage_c).parameters["threshold"].default == STAGE_C_ME_THRESHOLD
    assert STAGE_C_ME_THRESHOLD > 1.0, "stage C must see every chat"
    assert inspect.signature(pack_stage_d).parameters["threshold"].default == ME_SHARE_THRESHOLD


def test_threshold_change_warns(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("src.preprocessing.pack.DATA_PACKED", tmp_path)
    stage_dir = tmp_path / "stage_c"
    stage_dir.mkdir()
    (stage_dir / "meta.json").write_text(json.dumps({"me_share_threshold": 1.01}))

    _warn_if_threshold_changed("stage_c", 0.20)
    assert "WARNING" in capsys.readouterr().out

    _warn_if_threshold_changed("stage_c", 1.01)
    assert "WARNING" not in capsys.readouterr().out


@pytest.fixture
def manifest_env(tmp_path, monkeypatch):
    """Two tiny jsonl sources, a fake pretrain pack for replay, and a manifest factory."""
    monkeypatch.setattr("src.preprocessing.pack.DATA_PACKED", tmp_path / "packed")

    pretrain_dir = tmp_path / "packed" / "pretrain"
    pretrain_dir.mkdir(parents=True)
    np.arange(50_000, dtype=DTYPE).tofile(pretrain_dir / "train.bin")

    def write_jsonl(name, texts):
        path = tmp_path / name
        path.write_text("\n".join(json.dumps({"text": t}) for t in texts) + "\n")
        return path

    a = write_jsonl("a.jsonl", [f"وثيقة تجريبية رقم {i} عن السودان" for i in range(30)])
    b = write_jsonl("b.jsonl", [f"مستند آخر {i} بنص مختلف تماما" for i in range(20)])

    def make(stage, sources, arabic_replay=0.0, val_sources=()):
        import yaml
        path = tmp_path / f"{stage}.yaml"
        path.write_text(yaml.safe_dump({
            "stage": stage, "sources": sources, "arabic_replay": arabic_replay,
            "val_sources": list(val_sources),
        }))
        return path

    return tmp_path, a, b, make


def test_manifest_repeat_multiplies_the_token_stream(manifest_env):
    tmp_path, a, _, make = manifest_env
    once = pack_manifest(make("m1", [{"path": str(a), "name": "a", "repeat": 1}]))
    twice = pack_manifest(make("m2", [{"path": str(a), "name": "a", "repeat": 2}]))
    assert twice["sources"][0]["tokens"] == 2 * once["sources"][0]["tokens"]
    assert twice["splits"]["train"] == 2 * once["splits"]["train"]


def test_manifest_replay_hits_the_requested_fraction(manifest_env):
    tmp_path, a, b, make = manifest_env
    meta = pack_manifest(make(
        "m3",
        [{"path": str(a), "name": "a"}, {"path": str(b), "name": "b", "repeat": 3}],
        arabic_replay=0.4,
    ))
    assert meta["replay_fraction_actual"] == pytest.approx(0.4, abs=0.01)
    assert meta["primary_tokens"] + meta["replay_tokens"] == meta["splits"]["train"]
    # per-source counts are the record of what the mixture actually was
    assert sum(s["tokens"] for s in meta["sources"]) == meta["primary_tokens"]
    actual = np.memmap(tmp_path / "packed" / "m3" / "train.bin", dtype=DTYPE, mode="r")
    assert len(actual) == meta["splits"]["train"]


def test_manifest_val_boundaries_are_contiguous(manifest_env):
    tmp_path, a, b, make = manifest_env
    meta = pack_manifest(make(
        "m4", [{"path": str(a), "name": "a"}],
        val_sources=[{"path": str(a), "name": "a"}, {"path": str(b), "name": "b"}],
    ))
    val = meta["val_sources"]
    assert val[0]["start"] == 0
    assert val[1]["start"] == val[0]["tokens"], "boundaries must tile val.bin with no gaps"
    assert val[0]["tokens"] + val[1]["tokens"] == meta["splits"]["val"]


@real_pack
def test_real_pack_decodes_to_readable_arabic():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        Path(__file__).resolve().parents[1] / "tokenizers" / "v2_32k"
    )
    ds = PackedDataset(PACKED_DIR / "train.bin", block_size=1024)
    for i in (0, len(ds) // 2, len(ds) - 1):
        text = tok.decode(ds[i][0][:200].tolist(), skip_special_tokens=True)
        arabic = sum("؀" <= c <= "ۿ" for c in text)
        assert arabic > 0.3 * max(len(text), 1), f"block {i} does not look like Arabic"
