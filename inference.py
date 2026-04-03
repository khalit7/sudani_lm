from collections import Counter,defaultdict
import torch
from torch.mps import is_available
import yaml
from src.factory import Factory
from pathlib import Path


config_root = Path("~/sudani_lm/configs").expanduser()
config_name = "pretraining.yaml" 
# reading config file
with open(config_root/config_name,'r') as f:
    config = yaml.safe_load(f)


factory = Factory(config)

model = factory.get_model()
dataloader = factory.get_dataloader(config["eval"]["mmlu"]["dataloader"])

# load checkpoint
checkpoint = torch.load("checkpoints/sudani_llm_pretraining/base_dataset_4_layer_mmlu/best.pt")

model.load_state_dict(checkpoint["model_state_dict"])

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"

for name,param in model.named_parameters():
    print(name)
    print(param)

# model.to(device)
# for _ in range(5):
#
#     overal_cnt = defaultdict(int)
#     with torch.no_grad():
#         for X,Y in dataloader:
#             X = { k:v.to(device) for k,v in X.items()}
#             output = model(**X)# has shape (batch_size,seq_len,vocab_size)
#             next_token_logits = output[:,-1,:]# has shape (batch_size,vocab_size)
#             filtered_logits = next_token_logits[:,dataloader.dataset.options_ids] # has shape (batch_size,num_options)
#             filtered_logits = filtered_logits.argmax(dim=-1).cpu().tolist()
#             ctr = Counter(filtered_logits)
#             for k,v in ctr.items():
#                 overal_cnt[k] += v
#             # y_pred.extend( filtered_logits.argmax(dim=-1).cpu().tolist())
#             # y_true.extend(Y)
#
#     print(overal_cnt)
#     print(len(dataloader.dataset))
