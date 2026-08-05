"""Gate for chat and Sudanese preprocessing (Part I, Step 6).

The load-bearing test is test_no_chat_appears_in_both_splits: holding out random *messages*
instead of whole *chats* would leak running jokes, nicknames and recurring phrases across the
split, and the resulting perplexity is the number that steers Stages C and D.
"""

import json
from datetime import datetime, timedelta

import pytest

from src.preprocessing.sudani import TELE_YA_FIX, clean as clean_sudani
from src.preprocessing.whatsapp import (
    MAX_TURNS_PER_CONVERSATION,
    build_conversations,
    chunk_conversation,
    clean_text,
    merge_consecutive,
    segment_sessions,
    speaker_for,
    split_by_chat,
)
from src.tokenizer.special_tokens import CONV_START, SELF_SPEAKER, TURN_END, TURN_START

BASE = datetime(2025, 1, 1, 12, 0, 0)


def msg(minutes, speaker, text):
    return (BASE + timedelta(minutes=minutes), speaker, text)


# --- speaker labelling -----------------------------------------------------------------------

def test_outgoing_is_labelled_me_despite_empty_sender_name():
    """Every one of the 412,830 Outgoing rows has a blank Sender Name in this export, so the
    owner's own messages must be identified from Type."""
    row = {"Type": "Outgoing", "Sender Name": ""}
    assert speaker_for(row, "some chat") == SELF_SPEAKER


def test_incoming_uses_sender_name():
    assert speaker_for({"Type": "Incoming", "Sender Name": "Mukh"}, "chat") == "Mukh"


def test_incoming_without_a_name_falls_back_to_the_chat():
    """6,251 incoming rows carry no sender name; in a 1:1 chat the chat name identifies them."""
    assert speaker_for({"Type": "Incoming", "Sender Name": ""}, "Reem") == "Reem"


# --- session segmentation --------------------------------------------------------------------

def test_long_pause_starts_a_new_session():
    messages = [msg(0, "A", "hi"), msg(5, "B", "hey"), msg(60 * 7, "A", "next day")]
    sessions = segment_sessions(messages, gap_hours=6)
    assert len(sessions) == 2
    assert sessions[0] == [("A", "hi"), ("B", "hey")]
    assert sessions[1] == [("A", "next day")]


def test_short_pause_keeps_one_session():
    messages = [msg(0, "A", "hi"), msg(60 * 5, "B", "still here")]
    assert len(segment_sessions(messages, gap_hours=6)) == 1


# --- burst merging ----------------------------------------------------------------------------

def test_consecutive_messages_from_one_sender_merge():
    """People send four messages where one paragraph was meant. Unmerged, the model learns to
    emit a turn marker every few words."""
    merged = merge_consecutive([("A", "one"), ("A", "two"), ("B", "reply"), ("A", "three")])
    assert merged == [("A", "one two"), ("B", "reply"), ("A", "three")]


def test_merging_preserves_alternation():
    merged = merge_consecutive([("A", "x"), ("B", "y"), ("A", "z")])
    assert [s for s, _ in merged] == ["A", "B", "A"]


# --- chunking ----------------------------------------------------------------------------------

def test_long_sessions_are_chunked():
    turns = [("A" if i % 2 == 0 else "B", f"msg{i}") for i in range(80)]
    chunks = chunk_conversation(turns)
    assert len(chunks) > 1
    assert all(len(c) <= MAX_TURNS_PER_CONVERSATION for c in chunks)


def test_single_turn_conversations_are_dropped():
    """A lone message is not a dialogue and teaches nothing about replying."""
    assert chunk_conversation([("A", "just one")]) == []


# --- rendering -----------------------------------------------------------------------------------

def test_rendered_conversation_uses_the_template():
    by_chat = {"chat": [msg(0, "Mukh", "hi"), msg(1, SELF_SPEAKER, "tamam")]}
    text = build_conversations(by_chat)["chat"][0]
    assert text.startswith(f"<s>{CONV_START}")
    assert text.endswith("</s>")
    assert f"{TURN_START}Mukh: hi{TURN_END}" in text
    assert f"{TURN_START}{SELF_SPEAKER}: tamam{TURN_END}" in text
    assert text.count(TURN_START) == text.count(TURN_END) == 2


def test_newlines_inside_a_message_are_collapsed():
    """A turn is one utterance; the template already delimits turns."""
    assert clean_text("line one\n  line two") == "line one line two"


def test_pseudonymize_masks_phone_numbers_when_enabled():
    text = "call me on +249 91 234 5678 ok"
    assert "+249" in clean_text(text, pseudonymize=False)
    assert "<phone>" in clean_text(text, pseudonymize=True)


# --- the holdout gate -----------------------------------------------------------------------------

def test_no_chat_appears_in_both_splits():
    conversations = {f"chat{i}": [f"c{i}-{j}" for j in range(10)] for i in range(40)}
    train, val = split_by_chat(conversations, val_fraction=0.2)
    assert set(train).isdisjoint(set(val)), "a chat leaked across the split"
    assert set(train) | set(val) == set(conversations)


def test_holdout_is_whole_chats_not_sampled_messages():
    conversations = {f"chat{i}": [f"c{i}-{j}" for j in range(10)] for i in range(40)}
    train, val = split_by_chat(conversations, val_fraction=0.2)
    for chat, items in val.items():
        assert items == conversations[chat], "a held-out chat was only partially held out"


def test_holdout_split_is_deterministic():
    conversations = {f"chat{i}": ["x"] * (i + 1) for i in range(30)}
    a = split_by_chat(conversations, seed=7)[1]
    b = split_by_chat(conversations, seed=7)[1]
    assert set(a) == set(b)


def test_holdout_reaches_roughly_the_requested_share():
    conversations = {f"chat{i}": ["x"] * 10 for i in range(100)}
    _, val = split_by_chat(conversations, val_fraction=0.1)
    share = sum(len(v) for v in val.values()) / 1000
    assert 0.08 < share < 0.20


# --- real artifacts, when they exist ---------------------------------------------------------------

from pathlib import Path  # noqa: E402

INTERIM = Path(__file__).resolve().parents[1] / "data" / "interim"
real = pytest.mark.skipif(
    not (INTERIM / "whatsapp" / "val.jsonl").exists(),
    reason="run python -m src.preprocessing.whatsapp first",
)


@real
def test_real_split_shares_no_chat():
    train = {json.loads(l)["chat"] for l in open(INTERIM / "whatsapp" / "train.jsonl", encoding="utf-8")}
    val = {json.loads(l)["chat"] for l in open(INTERIM / "whatsapp" / "val.jsonl", encoding="utf-8")}
    assert train.isdisjoint(val)


@real
def test_real_conversations_are_well_formed():
    for i, line in enumerate(open(INTERIM / "whatsapp" / "train.jsonl", encoding="utf-8")):
        if i >= 2000:
            break
        text = json.loads(line)["text"]
        assert text.startswith(f"<s>{CONV_START}") and text.endswith("</s>")
        assert text.count(TURN_START) == text.count(TURN_END) >= 2


# --- Sudanese corpora ------------------------------------------------------------------------------

def test_tele_ya_substitution_is_reversed():
    """The Telegram corpus was collected with every ي rewritten as ى. Left alone it teaches a
    spelling that does not occur in natural Sudanese writing."""
    assert clean_sudani("قلت لى اقفل التلفون تانى", fix_ya=True) == "قلت لي اقفل التلفون تاني"
    # and only for that corpus — ى is a legitimate letter elsewhere
    assert "ى" in clean_sudani("مصطفى", fix_ya=False)


def test_sudani_strips_urls_mentions_and_bom():
    assert "http" not in clean_sudani("شوف https://example.com/x دي")
    assert "@" not in clean_sudani("سلام @someone كيفك")
    assert clean_sudani("﻿نص").startswith("نص")


def test_ya_table_only_touches_alef_maqsura():
    assert TELE_YA_FIX == str.maketrans({"ى": "ي"})
