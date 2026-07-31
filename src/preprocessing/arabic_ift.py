from datasets import load_dataset
from pathlib import Path


data_root = Path("~/sudani_lm/data").expanduser()

ds = load_dataset("ClusterlabAi/InstAr-500k")["train"]
split_dataset = ds.train_test_split(test_size=0.1,seed=67)

split_dataset["train"].save_to_disk(data_root/"raw"/"instar500k"/"train")
split_dataset["test"].save_to_disk(data_root/"raw"/"instar500k"/"test")
print(ds)
