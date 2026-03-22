from pathlib import Path
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR,CosineAnnealingLR,SequentialLR
from torch.nn import CrossEntropyLoss
import wandb

from .generator import Generator

from src.dataset.utils import get_tokenizer

class Trainer:
    def __init__(self,model,wandb_run,config) -> None:
        self.model = model
        self.wandb_run = wandb_run
        self.config = config
        tokenizer = get_tokenizer()
        self.loss_fn = CrossEntropyLoss(ignore_index = tokenizer.pad_token_id )
        self.optimizer = Adam(self.model.parameters(),lr=config["learning_rate"])

        self.wandb_run.define_metric("val_loss",summary="min")
        self.generation_table = wandb.Table(columns=["step","prompt","generation"],log_mode="MUTABLE")
        self.generator = Generator(self.model)

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.mps.is_available():
            self.device = "mps"

        self.checkpoints_root = Path("~/sudani_lm/checkpoints").expanduser()/ self.model.__class__.__name__ / Path(self.wandb_run.name).stem 
        self.checkpoints_root.mkdir(parents=True,exist_ok=True)
        (self.checkpoints_root/"last_x").mkdir(parents=True,exist_ok=True)

    def train(self,train_dataloader,val_dataloader):
        self.model.to(self.device)
        self.model.train()
        warmup_steps = len(train_dataloader)*self.config["warmup_percentage"]
        linear_lr_scheduler = LinearLR(self.optimizer,start_factor=self.config["warmup_start_factor"],end_factor=1,total_iters=warmup_steps)
        cosine_lr_scheduler = CosineAnnealingLR(self.optimizer,T_max=len(train_dataloader)-warmup_steps)
        self.lr_scheduler = SequentialLR(self.optimizer,[linear_lr_scheduler,cosine_lr_scheduler],milestones=[warmup_steps])
        print("Starting training on device = ",self.device)

        for epoch in range(self.config["num_epochs"]):
            for step,(X,Y) in tqdm(enumerate(train_dataloader)):

                # run evaluation
                if step%self.config["eval_every"] == 0:
                    self.eval(val_dataloader,epoch,step)

                # run training step
                self.train_step(X,Y,step)



    def train_step(self,X,Y,step):
        X = { k:v.to(self.device) for k,v in X.items() }
        Y = Y.to(self.device)
        output = self.model(**X)
        loss = self.loss_fn(output,Y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        self.wandb_run.log(
                {
                    "train_loss":loss.detach().item(),
                    "learning_rate":self.optimizer.param_groups[0]["lr"]
                },
                step=step
                )


    def eval(self,dataloader,epoch,step):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X,Y in tqdm(dataloader):
                X = {k:v.to(self.device) for k,v in X.items()}
                Y = Y.to(self.device)
                output = self.model(**X)
                loss   = self.loss_fn(output,Y) 
                total_loss += loss.item()
        avg_loss = total_loss/len(dataloader)
        self.wandb_run.log({"val_loss":avg_loss},step=step)

        min_val_loss = self.wandb_run.summary.get("val_loss",{}).get("min",float("inf"))
        if avg_loss < min_val_loss:
            print(f"found new best model!")
            print("old val loss = ", min_val_loss)
            print("new val loss = ", avg_loss)
            self._save_checkpoint(epoch=epoch,step=step,checkpoint_name="best.pt")
        self._save_checkpoint(epoch=epoch,step=step,checkpoint_name=f"last_x/checkpoint_{step}.pt")

        prompt = "<s>"
        generated_text = self.generator.generate(prompt=prompt)
        self.generation_table.add_data(step,prompt,generated_text)
        self.wandb_run.log({"generation":self.generation_table},step=step)

        self.model.train()

    def _save_checkpoint(self,epoch,step,checkpoint_name):
        checkpoint_str = \
        {
        "epoch":epoch,
        "step":step,
        "model_state_dict":self.model.state_dict(),
        "optimiser_state_dict":self.optimizer.state_dict(),
        "lr_scheduler_state_dict":self.lr_scheduler.state_dict(),
        }

        torch.save(checkpoint_str,self.checkpoints_root/checkpoint_name)

        # Only clean up routine checkpoints, not the "best" checkpoint
        if checkpoint_name != "best":
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Delete old routine checkpoints, keeping only the highest n_checkpoints step numbers."""
        checkpoint_dir = self.checkpoints_root/"last_x"
        # Get all checkpoint files except "best.pth"
        checkpoint_files = [
            f for f in checkpoint_dir.glob("checkpoint_*.pth")
        ]

        # Sort by step number extracted from filename (highest first)
        def get_step_number(filepath):
            # Extract step number from "checkpoint_{step}.pth"
            stem = filepath.stem  # "checkpoint_{step}"
            return int(stem.split('_')[1])

        checkpoint_files.sort(key=get_step_number, reverse=True)

        # Delete old checkpoints beyond n_models_to_save
        for old_checkpoint in checkpoint_files[self.config["n_checkpoints"]:]:
            print(f"Deleting old checkpoint: {old_checkpoint.name}")
            old_checkpoint.unlink()
