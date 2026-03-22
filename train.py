import sys
import torch
import wandb
from tqdm import tqdm

from pathlib import Path
from src.models.decoder import Decoder
from src.dataset.arabic import get_data_loader
from src.dataset.utils import get_tokenizer
from src.trainer import Trainer
from src.generator import Generator
import yaml

is_profile_run = False

config_root = Path("~/sudani_lm/configs").expanduser()
config_name = "pretraining/init_config.yaml"

# reading config file
with open(config_root/config_name,'r') as f:
    config = yaml.safe_load(f)

print("----------- getting tokenizer -------------")
tokenizer = get_tokenizer()

print("----------- getting dataloaders-------------")
train_dataloader = get_data_loader(split="train",**config["train_dataloader"])
val_dataloader   = get_data_loader(split="val",**config["val_dataloader"])

config["model"]["vocab_size"] = tokenizer.vocab_size
config["train_dataloader"]["steps"] = len(train_dataloader)
config["val_dataloader"]["steps"]   = len(val_dataloader)

print("----------- initializing the model -------------")
model = Decoder(config["model"])
model_stats = model.get_model_stats()
config["model"]["stats"] = model_stats
if is_profile_run:
    dummy_input = {
             "input_ids":torch.randint(low=0,high=10,size=(config["train_dataloader"]["batch_size"],1024)),
             "attention_mask":torch.ones((config["train_dataloader"]["batch_size"],1024))}
    
    model.profile_model(dummy_input)
    sys.exit()


print("----------- initializing the trainer -----------")
wandb_run = wandb.init(
project = config["project"],
name = config["run"],
config=config,
        )

trainer = Trainer(
        model=model,
        wandb_run=wandb_run,
        config=config["trainer"]
        )

print("----------- starting training ------------------")
trainer.train(train_dataloader,val_dataloader)
wandb_run.finish()

# from pathlib import Path
# checkpoint_root = Path("~/sudani_lm/checkpoints/Decoder/init_config").expanduser()
# checkpoint_dict = torch.load(checkpoint_root/"best.pt")
# model.load_state_dict(checkpoint_dict["model_state_dict"])
# generator = Generator(model.to("cuda"))
# text = generator.generate(prompt="<s> english")
# print(text)
