"""Hold a conversation with a trained checkpoint.

Separate from `inference.py` so the session logic is importable and testable without a terminal.

The model was trained on whole rendered conversations, so generation has to speak the same
format: history in the chat template, then an open `<|turn|>ME:` for the model to complete. Any
other prompt shape is off-distribution and the replies degrade accordingly.
"""

from pathlib import Path

import torch

from src.models.decoder import DecoderLMHeadModel
from src.tokenizer.special_tokens import (
    BOS,
    CONV_START,
    SELF_SPEAKER,
    TURN_END,
    TURN_START,
    render_turn,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "checkpoints" / "sudani_llm_pretraining" / "stage_d" / "best.pt"
)


def load_model(checkpoint_path=DEFAULT_CHECKPOINT, device=None):
    """Load a trained checkpoint. The model config travels inside the checkpoint."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = DecoderLMHeadModel(checkpoint["config"]["model"]["config"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, device, checkpoint


class ChatSession:
    """Multi-turn conversation state and prompt rendering."""

    def __init__(self, model, tokenizer, device, max_new_tokens=64, temperature=0.8,
                 top_k=50, top_p=0.95, speaker="Friend"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.speaker = speaker
        self.turns: list[tuple[str, str]] = []

        self.max_seq_len = getattr(model, "max_seq_len", 1024)
        # Stop at the end of a turn, not only at end of conversation: the model closes its reply
        # with <|end|> and would otherwise keep inventing the other side of the dialogue.
        self.stop_ids = [
            i for i in (tokenizer.convert_tokens_to_ids(TURN_END), tokenizer.eos_token_id)
            if i is not None
        ]

    def reset(self):
        self.turns.clear()

    def build_prompt(self) -> str:
        """History plus an open ME turn for the model to complete.

        Oldest turns are dropped until the prompt leaves room for the reply — the model has a
        fixed 1024-token window and silently truncating from the wrong end would cut the most
        recent context, which is the part that matters most.
        """
        budget = self.max_seq_len - self.max_new_tokens - 8
        kept = list(self.turns)
        while True:
            body = "".join(render_turn(s, t) for s, t in kept)
            prompt = f"{BOS}{CONV_START}{body}{TURN_START}{SELF_SPEAKER}:"
            if len(self.tokenizer.encode(prompt, add_special_tokens=False)) <= budget or not kept:
                return prompt
            kept.pop(0)

    def reply(self, message: str) -> str:
        self.turns.append((self.speaker, message.strip()))
        prompt = self.build_prompt()
        ids = torch.tensor(
            [self.tokenizer.encode(prompt, add_special_tokens=False)], device=self.device
        )
        out = self.model.generate(
            ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k if self.temperature > 0 else None,
            top_p=self.top_p if self.temperature > 0 else None,
            stop_token_ids=self.stop_ids,
        )
        text = self._decode_reply(out[0, ids.shape[1]:].tolist())
        self.turns.append((SELF_SPEAKER, text))
        return text

    def _decode_reply(self, token_ids) -> str:
        text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        # Trim anything past the turn boundary: with sampling the model sometimes runs on and
        # starts speaking for the other person.
        for marker in (TURN_END, TURN_START, "</s>"):
            if marker in text:
                text = text.split(marker)[0]
        return text.strip()

    def transcript(self) -> str:
        return "\n".join(f"{speaker}: {text}" for speaker, text in self.turns)
