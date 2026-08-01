import torch
from torch.utils.data import Dataset,DataLoader
from src.tokenizer.utils import get_tokenizer
from src import dataset
from src.dataset.base import BaseDatasetModule
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from datasets import load
from typing import Any

data_root = Path("~/sudani_lm/data").expanduser()

class ArabicIFTDataset(Dataset):
    def __init__(self,dataset,tokenizer) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        datapoint = self.dataset[index]
        instruction = datapoint["instruction"]
        output = datapoint["output"]

        instruction = f"[inst] {instruction} [/inst]\n"

        return instruction,output

    def __len__(self):
        return len(self.dataset)


class ArabicIFTDatasetModule(BaseDatasetModule):
    
    def __init__(self) -> None:
        self.tokenizer = get_tokenizer()

    def build_dataset(self, split) -> torch.utils.data.Dataset:
        dataset = load.load_from_disk(data_root/"raw"/"instar500k"/split)
        return ArabicIFTDataset(dataset=dataset,tokenizer=self.tokenizer)

    def colllate_fn(self, batch) -> Any:
        X = []
        Y = []
        for instruction,output in batch:
            instruction_ids = self.tokenizer.encode(instruction,add_special_tokens=False)
            output_ids      = self.tokenizer.encode(output,add_special_tokens=False)

            combined_input_ids  = [self.tokenizer.bos_token_id] + instruction_ids + output_ids
            combined_output_ids = [-100]*len(instruction_ids) + output_ids + [self.tokenizer.eos_token_id]

            X.append(torch.tensor(combined_input_ids)[:1024])
            Y.append(torch.tensor(combined_output_ids)[:1024])

        X = pad_sequence(X,batch_first=True,padding_value=self.tokenizer.pad_token_id)
        Y = pad_sequence(Y,batch_first=True,padding_value=-100)

        assert X.shape == Y.shape

        attention_mask = X.ne(self.tokenizer.pad_token_id).long()

        return {"input_ids":X,"attention_mask":attention_mask},Y

            
