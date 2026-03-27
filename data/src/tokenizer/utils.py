from transformers import AutoTokenizer, PreTrainedTokenizerFast
from pathlib import Path

tokenizer_root = Path("~/sudani_lm/tokenizers").expanduser()
tokenizer : AutoTokenizer|None = None
def  get_tokenizer()->PreTrainedTokenizerFast:
    global tokenizer
    if tokenizer is None:
       tokenizer = AutoTokenizer.from_pretrained(tokenizer_root/"init_tokenizer") 
    return tokenizer

