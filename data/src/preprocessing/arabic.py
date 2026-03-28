from datasets import load_dataset
from pathlib import Path


seed = 67
data_root = Path("~/sudani_lm/data").expanduser()

data_path = data_root/"arab"/"raw"
full_dataset = load_dataset('lightonai/ArabicWeb24', data_files='ArabicWeb24/**/*.arrow', split='train',cache_dir=data_path)
full_dataset = full_dataset.train_test_split(test_size=0.1,seed=seed)["test"]

print("dataset size after dropout",len(full_dataset))

def is_valid_entry(example):
    text = example["text"]
    if not isinstance(text,str):
        return False
    if len(text.split())<20:
        return False

    return True

full_dataset = full_dataset.filter(
        is_valid_entry,
        num_proc=16
        )

splitted_dataset = full_dataset.train_test_split(test_size=0.02)

train_dataset = splitted_dataset["train"]
val_dataset   = splitted_dataset["test"]
print("train size:",len(train_dataset))
print("val size:",len(val_dataset))

train_dataset.save_to_disk(data_root/"arab"/"processed/train")
val_dataset.save_to_disk(data_root/"arab"/"processed/val")
