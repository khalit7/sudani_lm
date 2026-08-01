import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LinearLR,CosineAnnealingLR,SequentialLR
from torch.utils import data

from src.dataset.arabic_ift import ArabicIFTDatasetModule
from src.dataset.packed import PackedDatasetModule
from src.models.decoder import DecoderLMHeadModel
from src.dataset import ArabicPretrainingDatasetModule,ArabicMMLUDatasetModule
from src.tokenizer.utils import get_tokenizer

class Factory:
    def __init__(self,config) -> None:
        self.config = config

    def get_tokenizer(self):
        return get_tokenizer()

    def get_model(self) -> nn.Module :
        tokenizer = self.get_tokenizer()
        self.config["model"]["config"]["vocab_size"] = len(tokenizer)
        model_name   = self.config["model"]["name"]
        model_config = self.config["model"]["config"]
        if model_name == "init_decoder":
            return DecoderLMHeadModel(model_config) 
        else:
            raise Exception("Model name not recognised")

    def get_packed_module(self, stage: str, block_size: int) -> PackedDatasetModule:
        return PackedDatasetModule(stage=stage, block_size=block_size)

    def get_optimiser(self,model) -> torch.optim.Optimizer :

        optimiser_name   = self.config["train"]["optimiser"]["name"]
        optimiser_config = dict(self.config["train"]["optimiser"]["config"])
        if optimiser_name == "adam":
            # Legacy path: plain Adam over one undifferentiated parameter group.
            return Adam(model.parameters(), **optimiser_config)
        if optimiser_name != "adamw":
            raise Exception("Optimiser name not recognised")

        weight_decay = optimiser_config.pop("weight_decay", 0.1)
        # Decay only matrices. Applying it to norms and biases (all 1-D) shrinks parameters
        # that have no scale redundancy, which hurts rather than regularises.
        decay, no_decay = [], []
        for param in model.parameters():
            if not param.requires_grad:
                continue
            (decay if param.dim() >= 2 else no_decay).append(param)

        fused = torch.cuda.is_available()
        return AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            fused=fused,
            **optimiser_config,
        )

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

        if dataloader_name == "packed":
            # Offline-packed token stream: no tokenization in the hot loop, no padding, fixed
            # shapes. This is the path pretraining should use.
            dataset = PackedDatasetModule(
                stage=dataloader_config.get("stage", "pretrain"),
                block_size=dataloader_config.get("block_size", 1024),
            )
            return dataset.build_dataloader(split,**dataloader_params)
        elif dataloader_name == "arabic":
            # Legacy path: tokenizes twice per example inside the collate function and truncates
            # documents at 1024 tokens. Superseded by "packed"; kept only to reproduce old runs.
            dataset = ArabicPretrainingDatasetModule()
            return dataset.build_dataloader(split,**dataloader_params)
        elif dataloader_name == "mmlu":
            dataset = ArabicMMLUDatasetModule()
            return dataset.build_dataloader(split,**dataloader_params)
        elif dataloader_name == "arabic_ift":
            dataset = ArabicIFTDatasetModule()
            return dataset.build_dataloader(split,**dataloader_params)
        else:
            raise Exception("dataloader name not recognised")
        

    def build_evaluators(self) -> list:
        """Instantiate the evaluators declared under config["eval"].

        Returns [] when the section is absent, so a pure-throughput run needs no eval config.
        """
        from src.evaluator import (
            FloresPerplexityEvaluator,
            GenerationEvaluator,
            MMLULetterEvaluator,
            MMLULoglikelihoodEvaluator,
        )

        evaluators = []
        for name, cfg in (self.config.get("eval") or {}).items():
            cfg = dict(cfg)
            frequency = cfg.pop("freq", 500)
            run_at_0 = cfg.pop("run_at_0", True)

            if name == "mmlu_loglikelihood":
                module = ArabicMMLUDatasetModule()
                evaluators.append(MMLULoglikelihoodEvaluator(
                    module.build_dataset(cfg.pop("split", "test")),
                    frequency=frequency, run_at_0=run_at_0, **cfg,
                ))
            elif name == "mmlu_letter":
                dataloader = self.get_dataloader({
                    "name": "mmlu", "split": cfg.pop("split", "test"),
                    "config": cfg.pop("dataloader", {"batch_size": 32, "shuffle": False}),
                })
                evaluators.append(MMLULetterEvaluator(
                    dataloader, frequency=frequency, run_at_0=run_at_0))
            elif name == "flores":
                evaluators.append(FloresPerplexityEvaluator(
                    frequency=frequency, run_at_0=run_at_0, **cfg))
            elif name == "generation":
                evaluators.append(GenerationEvaluator(
                    frequency=frequency, run_at_0=run_at_0, **cfg))
            else:
                raise Exception(f"eval name not recognised: {name}")
        return evaluators
