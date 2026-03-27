import random
from pathlib import Path
from datasets import load_dataset
from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast

random.seed(67)

data_root = Path("~/sudani_lm/data")
data_path = data_root/"arab"/"raw"

dataset = load_dataset('lightonai/ArabicWeb24', data_files='ArabicWeb24/**/*.arrow', split='train',cache_dir=data_path.expanduser())

# build the tokenzier

SAVE_DIR = Path("~/sudani_lm/tokenizers/init_tokenizer/")
percentage_of_data_to_train_tokenizer = 0.01
vocab_size = 8 *10**3

num_datapoints = int(len(dataset)*percentage_of_data_to_train_tokenizer)
random_ids = random.sample( range(0,len(dataset)),num_datapoints )
print(f"will sample {num_datapoints} datapoints to build tokenizer")

tokenizer = SentencePieceBPETokenizer()
special_tokens = [
    "<s>",   # bos_token: Beginning of sequence
    "</s>",  # eos_token: End of sequence
    "<unk>", # unk_token: Unknown (fallback for characters not in vocab)
    "<pad>", # pad_token: Padding (to fix the error you had earlier!)
    "<mask>" # mask_token: Optional, used if you ever do fill-mask tasks
]
tokenizer.train_from_iterator(
dataset["text"][random_ids],
vocab_size=vocab_size,
special_tokens=special_tokens,
show_progress=True
        )

tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>"
)
tokenizer.save_pretrained(SAVE_DIR.expanduser())
print(tokenizer)
print(type(tokenizer))
