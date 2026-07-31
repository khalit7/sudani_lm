from abc import ABC, abstractmethod
from collections import Counter

import torch
from tqdm import tqdm
import wandb

from sklearn.metrics import classification_report

def last_real_token_logits(output, attention_mask):
    """Logits at each row's final non-padding position.

    `output[:, -1, :]` is the last position of the *padded* batch, which is a <pad> token for
    every row shorter than the batch maximum, so it reads the model's state at padding rather
    than at the end of the prompt.
    """
    last_idx = attention_mask.sum(dim=-1) - 1                      # (batch,)
    return output[torch.arange(output.shape[0], device=output.device), last_idx]


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

    def __init__(self,*args,**kwargs) -> None:
        super().__init__(*args,**kwargs)
        # Tracked here rather than read back from the wandb summary: the metric is logged as
        # "loss/val_loss", so looking up "val_loss" always missed and compared against inf,
        # which meant every single validation wrote "best.pt".
        self.best_val_loss = float("inf")

    def eval(self,wandb_run,step):
        loss_fn = torch.nn.functional.cross_entropy
        total_loss = 0.0
        total_tokens = 0
        for X,Y in tqdm(self.dataloader):
            X = {k:v.to(self.device) for k,v in X.items()}
            Y = Y.to(self.device).flatten()
            output = self.model(**X)
            # Summed, not averaged, so batches are weighted by their real token count instead
            # of every batch counting equally regardless of how much padding it carried.
            loss = loss_fn(output.view(Y.shape[0],-1),Y,ignore_index=-100,reduction="sum")
            total_loss += loss.item()
            total_tokens += int((Y != -100).sum().item())
        avg_loss = total_loss/max(total_tokens,1)
        wandb_run.log({"loss/val_loss":avg_loss},step=step)

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            return "best.pt"
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

        options_ids = torch.tensor(self.dataloader.dataset.options_ids,device=self.device)
        y_pred = []
        y_true = []
        chance = 0.0
        for X,(Y,n_options) in tqdm(self.dataloader):
            X = { k:v.to(self.device) for k,v in X.items()}
            n_options = n_options.to(self.device)
            output = self.model(**X)                                   # (batch,seq,vocab)
            next_token_logits = last_real_token_logits(output,X["attention_mask"])
            filtered_logits = next_token_logits[:,options_ids]          # (batch,MAX_OPTIONS)

            # Mask option slots this question does not have, so a 3-option question can never
            # be answered "د" or "ه".
            slots = torch.arange(filtered_logits.shape[1],device=self.device)
            filtered_logits = filtered_logits.masked_fill(
                slots.unsqueeze(0) >= n_options.unsqueeze(1), float("-inf")
            )

            y_pred.extend(filtered_logits.argmax(dim=-1).cpu().tolist())
            y_true.extend(Y.tolist())
            chance += (1.0/n_options).sum().item()

        clf_report = classification_report(y_true,y_pred,output_dict=True,zero_division=0.0)

        # Logged alongside accuracy because this benchmark's chance level is not a constant —
        # option counts vary from 2 to 5 — so "above 0.2" is not the right reference line.
        pred_hist = Counter(y_pred)
        wandb_run.log( {
            "mmlu/mmlu_acc":clf_report["accuracy"],
            "mmlu/mmlu_chance":chance/max(len(y_true),1),
            "mmlu/mmlu_weighted_precision":clf_report["weighted avg"]["precision"],
            "mmlu/mmlu_weighted_recall"   :clf_report["weighted avg"]["recall"],
            "mmlu/mmlu_weighted_f1"       :clf_report["weighted avg"]["f1-score"],
            "mmlu/mmlu_macro_precision":clf_report["macro avg"]["precision"],
            "mmlu/mmlu_macro_recall"   :clf_report["macro avg"]["recall"],
            "mmlu/mmlu_macro_f1"       :clf_report["macro avg"]["f1-score"],
            # Surfaces the "model always answers أ" failure mode directly.
            **{f"mmlu/pred_frac_{i}":pred_hist.get(i,0)/max(len(y_pred),1) for i in range(5)},
            },step=step)
