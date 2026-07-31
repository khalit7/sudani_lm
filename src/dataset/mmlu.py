import random
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from src.dataset.base import BaseDatasetModule
from src.tokenizer.utils import get_tokenizer

data_root = Path("~/sudani_lm/data").expanduser()

# Arabic letters used to label the options, in order.
OPTION_LETTERS = ["أ", "ب", "ج", "د", "ه"]
MAX_OPTIONS = len(OPTION_LETTERS)

# Cue appended after the options so that an option letter is the natural next token.
# Without it the prompt ends on the last option's text and the model is never actually
# asked for an answer.
ANSWER_CUE = "\nالإجابة:"


def get_answer_token_ids(tokenizer, letters=OPTION_LETTERS, cue=ANSWER_CUE):
    """Token id the model would actually emit for each option letter, in context.

    `convert_tokens_to_ids("أ")` returns the id of the bare letter, which is *not* what a
    BPE tokenizer produces after "الإجابة:" — there it is the space-prefixed variant. Scoring
    the bare ids means scoring tokens the model essentially never emits, which pins accuracy
    at chance. So encode the cue with and without the letter and take the first new token.
    """
    base = tokenizer.encode(cue, add_special_tokens=False)
    ids = []
    for letter in letters:
        full = tokenizer.encode(cue + " " + letter, add_special_tokens=False)
        if full[: len(base)] != base or len(full) <= len(base):
            raise ValueError(
                f"tokenization of {letter!r} is not a clean suffix of the answer cue; "
                "the option-letter ids cannot be derived this way for this tokenizer"
            )
        ids.append(full[len(base)])
    return ids


class MMLU(Dataset):
    def __init__(self, dataset, tokenizer) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.options_tokens = OPTION_LETTERS
        self.options_ids = get_answer_token_ids(tokenizer)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        datapoint = self.dataset[index]

        # Option count varies across the benchmark (4 options for 10,120 questions, 3 for
        # 2,121, 2 for 1,874 and 5 for only 340), so render exactly the ones that exist.
        # Rendering a fixed 5 injects a literal "None" and scores a 5-way space for a
        # question that only has 4 answers.
        present = [
            datapoint[f"Option {i}"]
            for i in range(1, MAX_OPTIONS + 1)
            if datapoint.get(f"Option {i}") not in (None, "")
        ]
        n_options = len(present)

        # Seeded per index so the metric is comparable across steps and identical in every
        # dataloader worker. An unseeded shuffle makes the score wander run to run.
        order = list(range(n_options))
        random.Random(index).shuffle(order)

        options_str = "\n".join(
            f"{OPTION_LETTERS[position]} {present[original]}"
            for position, original in enumerate(order)
        )

        answer_idx = ord(datapoint["Answer Key"].strip().lower()) - ord("a")
        shuffled_answer_idx = order.index(answer_idx)

        question = datapoint["Question"]
        context = datapoint.get("Context") or ""
        question_block = f"{context}\n{question}" if context else question

        text_input = f"[inst] {question_block} [/inst]\n{options_str}{ANSWER_CUE}"

        return text_input, shuffled_answer_idx, n_options


class ArabicMMLUDatasetModule(BaseDatasetModule):
    def __init__(self):
        self.tokenizer = get_tokenizer()

    def build_dataset(self, split) -> torch.utils.data.Dataset:
        dataset = load_dataset(
            "MBZUAI/ArabicMMLU", "All", cache_dir=data_root / "raw" / "arabicmmlu"
        )[split]
        return MMLU(dataset, self.tokenizer)

    def colllate_fn(self, batch) -> Any:
        X = self.tokenizer(
            [x[0] for x in batch],
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )
        Y = torch.tensor([x[1] for x in batch], dtype=torch.long)
        # Carried through so the evaluator can mask option slots this question does not have.
        n_options = torch.tensor([x[2] for x in batch], dtype=torch.long)
        return X, (Y, n_options)
