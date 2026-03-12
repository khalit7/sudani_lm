import torch
import wandb
from tqdm import tqdm

from pathlib import Path
from src.models.decoder import Decoder
from src.dataset.arabic import get_data_loader, get_tokenizer
from src.trainer import Trainer
import yaml

config_root = Path("~/sudani_lm/configs").expanduser()
config_name = "init_config.yaml"

# reading config file
with open(config_root/config_name,'r') as f:
    config = yaml.safe_load(f)

# getting tokenizer using a singelton pattern
tokenizer = get_tokenizer()

train_dataloader = get_data_loader(split="train",**config["train_dataloader"])
val_dataloader   = get_data_loader(split="val",**config["val_dataloader"])

config["model"]["vocab_size"] = tokenizer.vocab_size
config["train_dataloader"]["steps"] = len(train_dataloader)
config["val_dataloader"]["steps"]   = len(val_dataloader)

# print("---------- tokenizer info -----------")
# print("special tokens:")
# speical_tookens = tokenizer.all_special_tokens
# print(speical_tookens)
# print(tokenizer.convert_tokens_to_ids(speical_tookens))
# print()
#

# print("----------- initializing the model -------------")
model = Decoder(config["model"])
model_stats = model.get_model_stats()
# dummy_input = {
#         "input_ids":torch.randint(low=0,high=10,size=(config["train_dataloader"]["batch_size"],1024)),
#         "attention_mask":torch.ones((config["train_dataloader"]["batch_size"],1024))}
#
# model.profile_model(dummy_input)
config["model"]["stats"] = model_stats

print("----------- initializing the trainer -----------")
wandb_run = wandb.init(
project = "arabic_decoder",
config=config,
name = config_name
        )

trainer = Trainer(
        model=model,
        wandb_run=wandb_run,
        config=config["trainer"]
        )

# print("----------- starting training ------------------")
trainer.train(train_dataloader,val_dataloader)
wandb_run.finish()


