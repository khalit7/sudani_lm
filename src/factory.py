import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR,CosineAnnealingLR,SequentialLR

from src.evaluator import Evaluator, GenerationEvaluator, MMLUEvaluator, ValidationEvaluator
from src.models.decoder import DecoderLMHeadModel
from src.dataset import ArabicPretrainingDatasetModule


class Factory:
    def __init__(self,config) -> None:
        self.config = config


    def get_model(self) -> nn.Module :
        
        model_name   = self.config["model"]["name"]
        model_config = self.config["model"]["config"]
        if model_name == "init_decoder":
            return DecoderLMHeadModel(model_config) 
        else:
            raise Exception("Model name not recognised")

    def get_optimiser(self,model_parameters) -> torch.optim.Optimizer :

        optimiser_name   = self.config["train"]["optimiser"]["name"]
        optimiser_config = self.config["train"]["optimiser"]["config"]
        if optimiser_name == "adam":
            return Adam(model_parameters,**optimiser_config)
        else:
            raise Exception("Optimiser name not recognised")

    def get_scheduler(self,total_training_steps,optimiser):

       scheduler_name   = self.config["train"]["scheduler"]["name"]
       scheduler_config = self.config["train"]["scheduler"]["config"]

       if scheduler_name == "warmup_cos":
          warmup_percentage  = scheduler_config["warmup_percentage"]
          warmup_start_factor = scheduler_config["warmup_start_factor"]

          warmup_steps = int(total_training_steps*warmup_percentage)
          remaining_steps = total_training_steps - warmup_steps

          linear_lr = LinearLR(optimiser,start_factor=warmup_start_factor,end_factor=1,total_iters=warmup_steps)
          cosine_lr = CosineAnnealingLR(optimiser,T_max=remaining_steps)

          return SequentialLR(optimiser,schedulers=[linear_lr,cosine_lr],milestones=[warmup_steps])

       else:
           raise Exception("scheduler name not recognised")
    
    def get_dataloader(self,dataloader_config):
        if dataloader_config is None:
            return None
        dataloader_name   = dataloader_config["name"]
        split             = dataloader_config["split"]
        dataloader_params = dataloader_config["config"]

        if dataloader_name == "arabic":
            dataset = ArabicPretrainingDatasetModule()
            return dataset.build_dataloader(split,**dataloader_params)
        else:
            raise Exception("dataloader name not recognised")
        

    def _construct_eval(self,eval_name,eval_config,model,device) -> Evaluator:
        frequency = eval_config["freq"]
        run_at_0  = eval_config["run_at_0"]
        dataloader_config = eval_config.get("dataloader")
        dataloader = self.get_dataloader(dataloader_config)
        if eval_name == "validation":
            return ValidationEvaluator(model,device,frequency,run_at_0,dataloader,eval_name)
        elif eval_name == "generation":
            return GenerationEvaluator(
                model,
                device,
                frequency,
                run_at_0,
                dataloader,
                eval_name,
                prompts=eval_config["prompts"],
                temperatures=eval_config["temperatures"],
                max_tokens=eval_config.get("max_tokens",50))
        elif eval_name == "mmlu":
            return MMLUEvaluator(model,device,frequency,run_at_0,dataloader,eval_name)
        else:
            raise Exception("eval name not recognised")
        
    def get_evals(self,model,device) -> list[Evaluator]:
        eval_dict = self.config["eval"]
        evals = []
        for eval_name,eval_config in eval_dict.items():
            evals.append(self._construct_eval(eval_name,eval_config,model,device))
        return evals

            
        
