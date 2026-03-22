from pandas.core.base import Shape
from torch._C import device
from torch.mps import is_available
from src.dataset.utils import get_tokenizer 
import torch

class Generator:

    def __init__(self,model) -> None:
        self.model = model
        self.tokenizer = get_tokenizer() 
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.mps.is_available():
            self.device = "mps"


    def generate(self,prompt="<s>",max_tokens=50,temperature=1,top_p=None,top_k=None):
        input_ids = self.tokenizer.encode(prompt,return_tensors="pt").to(self.device)
        with torch.no_grad():
            while input_ids.shape[-1] < max_tokens and input_ids[...,-1].item() != self.tokenizer.eos_token_id:
                logits = self.model(**{"input_ids":input_ids ,"attention_mask":torch.ones(input_ids.shape,device=self.device)})
                logits = logits[-1,...].flatten() # get the logits of only the final token
                if temperature == 0:
                    token_id = logits.argmax().unsqueeze(0)
                else:
                    prob = torch.nn.functional.softmax(logits/temperature,dim=-1)
                    token_id = torch.multinomial(prob,num_samples=1)
                token_id = token_id.unsqueeze(0)
                input_ids = torch.cat([input_ids,token_id],dim=-1) 


        return " ".join(self.tokenizer.decode(input_ids)[0].split())

    # TODO: implement kv caching
