import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR,CosineAnnealingLR,SequentialLR

from models.decoder import DecoderLMHeadModel


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

    def get_optimiser(self):

        optimiser_name   = self.config["train"]["optimiser"]["name"]
        optimiser_config = self.config["train"]["optimiser"]["config"]
        if optimiser_name == "adam":
            return Adam(optimiser_config)
        else:
            raise Exception("Optimiser name not recognised")

    def get_scheduler(self,total_training_steps):
       optimiser = self.get_optimiser()

       scheduler_name   = self.config["train"]["scheduler"]["name"]
       scheduler_config = self.config["train"]["scheduler"]["config"]

       if scheduler_name == "warmup_cos":
          warmup_percentage  = scheduler_config["warmup_percentage"]
          warmup_start_factor = scheduler_configp["warmup_start_factor"]

          warmup_steps = int(total_training_steps*warmup_percentage)
          remaining_steps = total_training_steps - warmup_steps

          linear_lr = LinearLR(optimiser,start_factor=warmup_start_factor,end_factor=1,total_iters=warmup_steps)
          cosine_lr = CosineAnnealingLR(optimiser,T_max=remaining_steps)

          return SequentialLR(optimiser,schedulers=[linear_lr,cosine_lr],milestones=[warmup_steps])

       else:
           raise Exception("scheduler name not recognised")
