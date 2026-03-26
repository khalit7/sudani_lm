import sys
import torch
import wandb
from tqdm import tqdm

from pathlib import Path
from src.models.decoder import DecoderLMHeadModel
from src.dataset.arabic import get_data_loader
from src.dataset.utils import get_tokenizer
from src.trainer import Trainer
from src.generator import Generator
import yaml

is_profile_run = False 

config_root = Path("~/sudani_lm/configs").expanduser()
config_name = "pretraining/config.yaml"

# reading config file
with open(config_root/config_name,'r') as f:
    config = yaml.safe_load(f)

if config["trainer"]["num_epochs"] != 1:
    raise Exception("Codebase assumes num epochs = 1! I know, stupid, submit a pr if you want to fix or use num_epochs = 1 :)")

print("----------- getting tokenizer -------------")
tokenizer = get_tokenizer()

print("----------- getting dataloaders-------------")
train_dataloader = get_data_loader(split="train",**config["train_dataloader"])
val_dataloader   = get_data_loader(split="val",**config["val_dataloader"])

config["model"]["vocab_size"] = tokenizer.vocab_size
config["train_dataloader"]["num_examples"] = len(train_dataloader.dataset)
config["val_dataloader"]["num_examples"]   = len(val_dataloader.dataset)

print("----------- initializing the model -------------")
model = DecoderLMHeadModel(config["model"])
model_stats = model.get_model_stats()
config["model"]["name"] = model.__class__.__name__
config["model"]["stats"] = model_stats
if is_profile_run:
    dummy_input_train = {
             "input_ids":torch.randint(low=0,high=10,size=(config["train_dataloader"]["batch_size"],1024)),
             "attention_mask":torch.ones((config["train_dataloader"]["batch_size"],1024))}

    dummy_input_val= {
             "input_ids":torch.randint(low=0,high=10,size=(config["val_dataloader"]["batch_size"],1024)),
             "attention_mask":torch.ones((config["val_dataloader"]["batch_size"],1024))}
    

    train_profile,val_profile = model.profile_model(dummy_input_train,dummy_input_val)
    print("TRAIN PROFILE: \n \n ")
    print(train_profile)
    print()
    print("VAL PROFILE: \n \n ")
    print(val_profile)
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
