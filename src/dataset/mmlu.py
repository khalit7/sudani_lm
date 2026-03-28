from typing import Any
import torch
from datasets import load_dataset
from torch.nn.modules import padding
from torch.utils.data import Dataset
from src.dataset.base import BaseDatasetModule
from data.src.tokenizer.utils import get_tokenizer




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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        datapoint = self.dataset[index]

        question = datapoint["Question"]
        context   = datapoint["Context"]
        options  = [ f"{self.options_tokens[i-1]}. {datapoint[f'Option {i}']}" for i in range(1,5+1) ]
        options_str = "\n   ".join(options)
        answer_idx = ord(datapoint["Answer Key"].lower()) - ord("a")


        text_input = \
        f"""
        اسمع يا زول، انا حاسالك سؤال و بديك خمسه خيارات، دايرك تجاوب الاجابه الصاح ب انك تختار الحرف بس! ما تحاول تكتب الكلام، اختار الحرف بتاع الاجابه الصاح بس. 
        السؤال:
        {question}
        الخيارات: 
        {options_str}
        الاجابه:"""

        return (text_input,answer_idx)




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
