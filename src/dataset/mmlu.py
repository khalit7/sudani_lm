import random
from typing import Any
import torch
from datasets import load_dataset
from torch.nn.modules import padding
from torch.utils.data import Dataset
from src.dataset.base import BaseDatasetModule
from src.tokenizer.utils import get_tokenizer




class MMLU(Dataset):

    def __init__(self,dataset,tokenizer) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.options_tokens = [
                "أ",
                "ب",
                "ج",
                "د",
                "ه"
                ]
        self.options_ids     = self.tokenizer.convert_tokens_to_ids(self.options_tokens)
        print("THOSE ARE THE TOKENS")
        print(self.options_tokens)
        print("THOSE ARE THE TOKENS IDS")
        print(self.options_ids)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        idxs = list(range(1,5+1))
        random.shuffle(idxs)

        datapoint = self.dataset[index]

        question = datapoint["Question"]
        context   = datapoint["Context"]
        options  = [ f"{self.options_tokens[letter_idx]} {datapoint[f'Option {option_idx}']}" for letter_idx,option_idx in enumerate(idxs) ]
        options_str = "\n".join(options)
        answer_idx = ord(datapoint["Answer Key"].lower()) - ord("a")
        shuffled_answer_idx = idxs.index(answer_idx+1)


        text_input = \
f"""
[inst] {question} [/inst]\n {options_str}
"""

        return (text_input,shuffled_answer_idx)




class ArabicMMLUDatasetModule(BaseDatasetModule):
    def __init__(self):
        self.tokenizer = get_tokenizer()
    
    def build_dataset(self, split) -> torch.utils.data.Dataset:
        dataset = load_dataset("MBZUAI/ArabicMMLU", "All")[split]
        return MMLU(dataset,self.tokenizer)


    def colllate_fn(self, batch) -> Any:
        X = self.tokenizer(
                [x[0] for x in batch],
                padding = True,
                truncation = True,
                max_length = 1024,
                return_tensors = "pt"
                )
        Y = [x[1] for x in batch]
        return X,Y
