"""Gate for the chat REPL (Part I, Step 7 / Stage E).

The load-bearing property is that the prompt the REPL builds matches the format the model was
trained on. Any other shape is off-distribution, and the failure is silent — the model still
replies, just worse.
"""

import pytest
import torch

from src.chat import ChatSession
from src.tokenizer.special_tokens import (
    BOS,
    CONV_START,
    SELF_SPEAKER,
    TURN_END,
    TURN_START,
    render_conversation,
)
from src.tokenizer.utils import get_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer()


class EchoModel(torch.nn.Module):
    """Returns a fixed continuation, so session mechanics can be tested without a checkpoint."""

    max_seq_len = 1024

    def __init__(self, reply_ids):
        super().__init__()
        self.reply_ids = reply_ids
        self.prompts = []

    def generate(self, input_ids, **kwargs):
        self.prompts.append(input_ids)
        tail = torch.tensor([self.reply_ids], device=input_ids.device)
        return torch.cat([input_ids, tail], dim=-1)


def make_session(tokenizer, reply="تمام", **kwargs):
    ids = tokenizer.encode(reply + TURN_END, add_special_tokens=False)
    return ChatSession(EchoModel(ids), tokenizer, "cpu", **kwargs)


# --- prompt format ---------------------------------------------------------------------------

def test_prompt_matches_the_training_template(tokenizer):
    session = make_session(tokenizer, speaker="Mukh")
    session.turns = [("Mukh", "هلا"), (SELF_SPEAKER, "اهلين")]
    prompt = session.build_prompt()

    assert prompt.startswith(f"{BOS}{CONV_START}")
    assert f"{TURN_START}Mukh: هلا{TURN_END}" in prompt
    # and it ends with an *open* owner turn for the model to complete
    assert prompt.endswith(f"{TURN_START}{SELF_SPEAKER}:")
    assert not prompt.endswith(TURN_END)


def test_prompt_prefix_matches_a_rendered_conversation(tokenizer):
    """The history portion must be byte-identical to what training saw."""
    session = make_session(tokenizer)
    session.turns = [("Ali", "سلام"), (SELF_SPEAKER, "هلا")]
    trained = render_conversation(session.turns, add_bos=True, add_eos=False)
    assert session.build_prompt().startswith(trained)


def test_empty_history_still_produces_a_valid_prompt(tokenizer):
    assert make_session(tokenizer).build_prompt() == f"{BOS}{CONV_START}{TURN_START}{SELF_SPEAKER}:"


# --- context window --------------------------------------------------------------------------

def test_oldest_turns_are_dropped_when_the_window_fills(tokenizer):
    """The model has a fixed 1024-token window. Dropping from the front keeps the most recent
    context, which is the part that matters."""
    session = make_session(tokenizer, max_new_tokens=64)
    session.turns = [("Ali", "كلام طويل " * 40) for _ in range(60)]
    prompt = session.build_prompt()
    n = len(tokenizer.encode(prompt, add_special_tokens=False))
    assert n <= session.max_seq_len - session.max_new_tokens - 8


def test_truncation_keeps_the_most_recent_turn(tokenizer):
    session = make_session(tokenizer, max_new_tokens=64)
    session.turns = [("Ali", "قديم " * 200) for _ in range(20)]
    session.turns.append(("Ali", "الرسالة الاخيرة"))
    assert "الرسالة الاخيرة" in session.build_prompt()


def test_a_single_oversized_turn_does_not_loop_forever(tokenizer):
    """The drop loop must terminate even when one turn alone exceeds the budget."""
    session = make_session(tokenizer, max_new_tokens=64)
    session.turns = [("Ali", "طويل " * 5000)]
    assert isinstance(session.build_prompt(), str)


# --- reply handling ---------------------------------------------------------------------------

def test_reply_is_trimmed_at_the_turn_boundary(tokenizer):
    """With sampling the model runs on and starts speaking for the other person."""
    session = make_session(tokenizer)
    text = session._decode_reply(
        tokenizer.encode(f"تمام{TURN_END}{TURN_START}Ali: زيادة", add_special_tokens=False)
    )
    assert text == "تمام"
    assert TURN_START not in text and TURN_END not in text


def test_reply_trimmed_at_end_of_conversation(tokenizer):
    session = make_session(tokenizer)
    assert session._decode_reply(
        tokenizer.encode("خلاص</s>راجع", add_special_tokens=False)
    ) == "خلاص"


def test_turns_accumulate_in_order(tokenizer):
    session = make_session(tokenizer, speaker="Mukh")
    session.reply("اول")
    session.reply("تاني")
    speakers = [s for s, _ in session.turns]
    assert speakers == ["Mukh", SELF_SPEAKER, "Mukh", SELF_SPEAKER]


def test_reply_conditions_on_the_growing_history(tokenizer):
    session = make_session(tokenizer, speaker="Mukh")
    session.reply("اول")
    session.reply("تاني")
    first, second = session.model.prompts
    assert second.shape[1] > first.shape[1], "the second prompt must carry the first exchange"


def test_reset_clears_history(tokenizer):
    session = make_session(tokenizer)
    session.reply("هلا")
    assert session.turns
    session.reset()
    assert session.turns == []
    assert session.build_prompt() == f"{BOS}{CONV_START}{TURN_START}{SELF_SPEAKER}:"


# --- stop tokens --------------------------------------------------------------------------------

def test_stop_ids_resolve_to_real_tokens(tokenizer):
    """A turn ends at <|end|>; without a valid id generation runs to max_new_tokens every time."""
    session = make_session(tokenizer)
    assert tokenizer.convert_tokens_to_ids(TURN_END) in session.stop_ids
    assert tokenizer.eos_token_id in session.stop_ids
    assert tokenizer.unk_token_id not in session.stop_ids


def test_generate_stops_on_any_listed_token():
    """Both terminators must stop generation, not just the first one."""
    from src.models.decoder import DecoderLMHeadModel

    torch.manual_seed(0)
    model = DecoderLMHeadModel(
        {"vocab_size": 64, "d_model": 32, "num_layers": 1, "num_heads": 4, "max_seq_len": 32}
    ).eval()
    ids = torch.randint(0, 64, (1, 3))
    # every token is a stop token, so generation must halt after exactly one step
    out = model.generate(ids, max_new_tokens=10, temperature=0.0,
                         stop_token_ids=list(range(64)))
    assert out.shape[1] == 4


def test_transcript_renders_both_speakers(tokenizer):
    session = make_session(tokenizer, speaker="Mukh")
    session.reply("هلا")
    transcript = session.transcript()
    assert "Mukh: هلا" in transcript
    assert f"{SELF_SPEAKER}:" in transcript
