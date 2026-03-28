from abc import ABC, abstractmethod

import torch
from tqdm import tqdm
import wandb

from sklearn.metrics import classification_report

class Evaluator(ABC):
    def __init__(self,model,device,frequency,run_at_0,dataloader,eval_name) -> None:
        self.model = model
        self.device = device
        self.frequency = frequency
        self.run_at_0 = run_at_0
        self.dataloader = dataloader
        self.eval_name = eval_name

    @abstractmethod
    def eval(self,wandb_run,step,**kwargs) -> str|None:
        pass

    def run_eval(self,wandb_run,step,**kwargs) -> str|None:
        self.model.eval()
        with torch.no_grad():
            checkpoint_name = self.eval(wandb_run,step,**kwargs)
        self.model.train()

        return checkpoint_name

class ValidationEvaluator(Evaluator):

    def eval(self,wandb_run,step,ignore_index):
        loss_fn = torch.nn.functional.cross_entropy
        total_loss = 0
        for X,Y in tqdm(self.dataloader):
            X = {k:v.to(self.device) for k,v in X.items()}
            Y = Y.to(self.device).flatten()
            output = self.model(**X)
            loss   = loss_fn(output.view(X["input_ids"].shape[0]*X["input_ids"].shape[1],-1),Y, ignore_index=ignore_index)
            total_loss += loss.item()
        avg_loss = total_loss/len(self.dataloader)
        wandb_run.log({"val_loss":avg_loss},step=step)

        min_val_loss = wandb_run.summary.get("val_loss",{}).get("min",float("inf"))
        if avg_loss < min_val_loss:
            return "best.pt"
        else:
            return None

class GenerationEvaluator(Evaluator):

    def __init__(self, model, device, frequency,run_at_0, dataloader, eval_name,prompts,temperatures,max_tokens=50):
        super().__init__(model, device, frequency,run_at_0, dataloader, eval_name)
        self.prompts = prompts
        self.temperatures = temperatures
        self.max_tokens = max_tokens
        self.wandb_table = wandb.Table(columns=["step","prompt","temperature","generation"],log_mode="MUTABLE")

    def eval(self,wandb_run,step,tokenizer):
        for prompt in self.prompts:
            for temperature in self.temperatures:
                generation = self._generate(prompt,temperature,self.max_tokens,tokenizer)
                self.wandb_table.add_data(step,prompt,temperature,generation)

        wandb_run.log({"generation":self.wandb_table},step=step)

        return None

    def _generate(self,prompt,temperature,max_tokens,tokenizer):
        input_ids = tokenizer.encode(prompt,return_tensors="pt").to(self.device)
        with torch.no_grad():
            while input_ids.shape[-1] < max_tokens and input_ids[...,-1].item() != tokenizer.eos_token_id:
                logits = self.model(**{"input_ids":input_ids ,"attention_mask":torch.ones(input_ids.shape,device=self.device)})
                logits = logits[-1,-1,...].flatten() # get the logits of only the final token
                if temperature == 0:
                    token_id = logits.argmax().unsqueeze(0)
                else:
                    prob = torch.nn.functional.softmax(logits/temperature,dim=-1)
                    token_id = torch.multinomial(prob,num_samples=1)
                token_id = token_id.unsqueeze(0)
                input_ids = torch.cat([input_ids,token_id],dim=-1) 

        return " ".join(tokenizer.decode(input_ids[0]).split())

class MMLUEvaluator(Evaluator):

    def eval(self,wandb_run,step):

        num_correct = 0
        for X,Y in tqdm(self.dataloader):
            X = { k:v.to(self.device) for k,v in X.items()}
            output = self.model(**X)# has shape (batch_size,seq_len,vocab_size)
            next_token_logits = output[:,-1,:].squeeze() # has shape (batch_size,vocab_size)
            filtered_logits = next_token_logits[:,self.dataloader.dataset.options_ids] # has shape (batch_size,num_options)

        clf_report = classification_report( Y, filtered_logits.argmax(dim=-1).cpu().numpy(), output_dict=True,zero_division=0)
         
        wandb_run.log( {
            "mmlu_acc":clf_report["accuracy"],
            "mmlu_weighted_precision":clf_report["weighted avg"]["precision"],
            "mmlu_weighted_recall"   :clf_report["weighted avg"]["recall"],
            "mmlu_weighted_f1"       :clf_report["weighted avg"]["f1-score"],
            "mmlu_weighted_precision":clf_report["weighted avg"]["precision"],
            "mmlu_weighted_recall"   :clf_report["weighted avg"]["recall"],
            "mmlu_weighted_f1"       :clf_report["weighted avg"]["f1-score"]




            },step=step)
