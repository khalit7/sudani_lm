from pathlib import Path
from src.trainer import Trainer
import yaml


config_root = Path("~/sudani_lm/configs").expanduser()
config_name = "pretraining.yaml"
 
# reading config file
with open(config_root/config_name,'r') as f:
    config = yaml.safe_load(f)

trainer = Trainer(config)
trainer.train()
