from transformers import AutoTokenizer,PreTrainedTokenizerFast
from datasets import load 
from torch.utils.data import Dataset,DataLoader
from pathlib import Path

data_root = Path("~/sudani_lm/data").expanduser()
tokenizer_root = Path("~/sudani_lm/tokenizers").expanduser()

tokenizer : AutoTokenizer|None = None
def get_tokenizer()->PreTrainedTokenizerFast:
    global tokenizer
    if tokenizer is None:
       tokenizer = AutoTokenizer.from_pretrained(tokenizer_root/"init_tokenizer") 
    return tokenizer


class ArabicPretrainingDataset(Dataset):

    def __init__(self,hf_dataset) -> None:
        self.dataset = hf_dataset
        self.tokenzier = get_tokenizer()


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        eos = self.tokenzier.eos_token
        bos = self.tokenzier.bos_token
        try:
            return bos + self.dataset[idx]["text"] , self.dataset[idx]["text"] + eos
        except:
            print("WTF!!")
            print(eos)
            print(bos)
            print(self.dataset[idx])




def collate_fn(batch:list[tuple[str,str]]):
    tokenizer = get_tokenizer()
    X = tokenizer(\
        [x[0] for x in batch], 
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt"
        )
    Y = tokenizer(\
        [x[1] for x in batch], 
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt"
        )
    return X,Y["input_ids"].flatten()


def get_data_loader(split,**kwargs):
    data_path = data_root/"arab"/"processed"/split

    dataset = load.load_from_disk(data_path)
    dataset = ArabicPretrainingDataset(dataset)
    return DataLoader(dataset,collate_fn=collate_fn,**kwargs)


